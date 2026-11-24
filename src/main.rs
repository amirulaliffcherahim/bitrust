mod ffi;
mod speculative;

use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        Html, IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use clap::{Parser, Subcommand};
use ffi::{BitNetContext, BitNetModel};
use serde::{Deserialize, Serialize};
use speculative::{generate, GenConfig, GenStats};
use std::convert::Infallible;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

#[derive(Parser, Debug)]
#[command(name = "bitrust", about = "bitrust-zen3: High-throughput Rust FFI inference harness for BitNet 1.58-bit models")]
struct Cli {
    /// Path to GGUF model
    #[arg(short, long)]
    model: String,

    /// Context size
    #[arg(short = 'c', long, default_value_t = 4096)]
    ctx_size: u32,

    /// Generation threads (physical cores)
    #[arg(short = 't', long, default_value_t = 4)]
    threads: i32,

    /// Prompt/batch threads (physical cores)
    #[arg(long, default_value_t = 8)]
    threads_batch: i32,

    /// Physical CPU core list for sched_setaffinity (comma-separated)
    #[arg(long, default_value = "0,2,4,6,8,10,12,14")]
    cores: String,

    /// Enable mlock
    #[arg(long, default_value_t = false)]
    mlock: bool,

    /// Max draft tokens for prompt-lookup speculative decoding (0 to disable)
    #[arg(long, default_value_t = 5)]
    draft: usize,

    /// Verbose llama.cpp logs
    #[arg(long, default_value_t = false)]
    verbose: bool,

    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run speed benchmark comparing standard vs prompt-lookup speculative decoding
    Bench {
        #[arg(short = 'n', long, default_value_t = 128)]
        max_tokens: usize,
        #[arg(short = 'p', long)]
        prompt: Option<String>,
    },
    /// Start OpenAI-compatible HTTP/SSE inference server
    Serve {
        #[arg(long, default_value = "0.0.0.0")]
        host: String,
        #[arg(long, default_value_t = 8080)]
        port: u16,
    },
}

struct EngineState {
    model: Arc<BitNetModel>,
    ctx: Arc<Mutex<BitNetContext>>,
    model_name: String,
    default_draft: usize,
    total_requests: AtomicU64,
    total_tokens: AtomicU64,
    last_tok_s_x100: AtomicU64,
    last_accept_x100: AtomicU64,
}

#[derive(Deserialize)]
struct ChatCompletionReq {
    #[allow(dead_code)]
    model: Option<String>,
    messages: Vec<ChatMsg>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    stream: Option<bool>,
}

#[derive(Deserialize, Serialize, Clone)]
struct ChatMsg {
    role: String,
    content: String,
}

fn parse_cores(s: &str) -> Vec<usize> {
    s.split(',')
        .filter_map(|x| x.trim().parse::<usize>().ok())
        .collect()
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

async fn index_html() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

async fn health(State(st): State<Arc<EngineState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "engine": "bitrust-zen3",
        "model": st.model_name,
        "total_requests": st.total_requests.load(Ordering::Relaxed),
        "total_tokens": st.total_tokens.load(Ordering::Relaxed),
        "last_tok_per_sec": (st.last_tok_s_x100.load(Ordering::Relaxed) as f64) / 100.0,
        "last_spec_accept_pct": (st.last_accept_x100.load(Ordering::Relaxed) as f64) / 100.0,
    }))
}

async fn list_models(State(st): State<Arc<EngineState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": st.model_name,
            "object": "model",
            "created": 1730000000,
            "owned_by": "bitrust"
        }]
    }))
}

async fn chat_completions(
    State(st): State<Arc<EngineState>>,
    Json(req): Json<ChatCompletionReq>,
) -> Response {
    let pairs: Vec<(String, String)> = req
        .messages
        .iter()
        .map(|m| (m.role.clone(), m.content.clone()))
        .collect();
    let prompt_text = st.model.apply_chat_template(&pairs, true);
    let prompt_tokens = match st.model.tokenize(&prompt_text, true, true) {
        Ok(t) => t,
        Err(e) => return (StatusCode::BAD_REQUEST, e).into_response(),
    };

    let cfg = GenConfig {
        max_tokens: req.max_tokens.unwrap_or(256).min(2048),
        temperature: req.temperature.unwrap_or(0.0),
        top_p: req.top_p.unwrap_or(0.95),
        speculative: st.default_draft > 0,
        max_draft: st.default_draft,
        seed: now_unix().wrapping_mul(6364136223846793005).wrapping_add(1),
    };
    let req_id = format!("chatcmpl-{}", now_unix());
    let created = now_unix();
    let model_name = st.model_name.clone();

    if req.stream.unwrap_or(false) {
        let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(64);
        let st_clone = st.clone();
        let model_name_c = model_name.clone();
        let req_id_c = req_id.clone();

        tokio::task::spawn_blocking(move || {
            let mut ctx = st_clone.ctx.lock().unwrap();
            let mut utf8_buf = Vec::<u8>::new();
            let res = generate(&st_clone.model, &mut ctx, &prompt_tokens, &cfg, |_tok, bytes| {
                utf8_buf.extend_from_slice(bytes);
                if let Ok(s) = std::str::from_utf8(&utf8_buf) {
                    let chunk = serde_json::json!({
                        "id": req_id_c,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name_c,
                        "choices": [{
                            "index": 0,
                            "delta": { "content": s },
                            "finish_reason": serde_json::Value::Null
                        }]
                    });
                    utf8_buf.clear();
                    tx.blocking_send(Ok(Event::default().data(chunk.to_string()))).is_ok()
                } else {
                    true
                }
            });
            if let Ok(stats) = res {
                record_stats(&st_clone, &stats);
            }
            let done_chunk = serde_json::json!({
                "id": req_id_c,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name_c,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            });
            let _ = tx.blocking_send(Ok(Event::default().data(done_chunk.to_string())));
            let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
        });

        let stream = ReceiverStream::new(rx);
        return Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response();
    }

    let st_clone = st.clone();
    let out = tokio::task::spawn_blocking(move || {
        let mut ctx = st_clone.ctx.lock().unwrap();
        let mut bytes_out = Vec::<u8>::new();
        let stats = generate(&st_clone.model, &mut ctx, &prompt_tokens, &cfg, |_tok, b| {
            bytes_out.extend_from_slice(b);
            true
        })?;
        record_stats(&st_clone, &stats);
        Ok::<(String, GenStats), String>((String::from_utf8_lossy(&bytes_out).into_owned(), stats))
    })
    .await;

    match out {
        Ok(Ok((text, stats))) => Json(serde_json::json!({
            "id": req_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": text },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": stats.prompt_tokens,
                "completion_tokens": stats.generated_tokens,
                "total_tokens": stats.prompt_tokens + stats.generated_tokens
            },
            "timings": {
                "prompt_tok_per_sec": (stats.prompt_tok_per_sec() * 100.0).round() / 100.0,
                "gen_tok_per_sec": (stats.tok_per_sec() * 100.0).round() / 100.0,
                "spec_draft_proposed": stats.draft_proposed,
                "spec_draft_accepted": stats.draft_accepted,
                "spec_accept_pct": (stats.accept_rate() * 100.0).round() / 100.0
            }
        }))
        .into_response(),
        Ok(Err(e)) => (StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

fn record_stats(st: &EngineState, stats: &GenStats) {
    st.total_requests.fetch_add(1, Ordering::Relaxed);
    st.total_tokens.fetch_add(stats.generated_tokens as u64, Ordering::Relaxed);
    st.last_tok_s_x100
        .store((stats.tok_per_sec() * 100.0) as u64, Ordering::Relaxed);
    st.last_accept_x100
        .store((stats.accept_rate() * 100.0) as u64, Ordering::Relaxed);
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let cores = parse_cores(&cli.cores);
    ffi::pin_to_cores(&cores);
    ffi::init_backend(cli.verbose);

    let model = Arc::new(BitNetModel::load(&cli.model, cli.mlock)?);
    let mut ctx = model.new_context(cli.ctx_size, 512, cli.threads, cli.threads_batch)?;

    match cli.cmd {
        Command::Bench { max_tokens, prompt } => {
            let prompts: Vec<(&str, String)> = if let Some(p) = prompt {
                vec![("custom", p)]
            } else {
                vec![
                    (
                        "general",
                        "Explain what a 1.58-bit large language model is and why ternary weights speed up CPU inference.".to_string(),
                    ),
                    (
                        "code-edit",
                        "Rewrite this Rust function and add documentation comments:\n```rust\nfn fibonacci(n: u64) -> u64 {\n    let mut a = 0;\n    let mut b = 1;\n    for _ in 0..n {\n        let tmp = a + b;\n        a = b;\n        b = tmp;\n    }\n    a\n}\n```\nReturn the full updated Rust function with doc comments and unit tests.".to_string(),
                    ),
                ]
            };

            for (label, text) in prompts {
                let formatted = model.apply_chat_template(&[("user".to_string(), text)], true);
                let toks = model.tokenize(&formatted, true, true)?;
                println!("\n=== BENCH: {} (prompt={} tok, max={}) ===", label, toks.len(), max_tokens);

                for &(mode, spec) in &[("standard", false), ("spec-lookup", true)] {
                    let cfg = GenConfig {
                        max_tokens,
                        temperature: 0.0,
                        top_p: 0.95,
                        speculative: spec,
                        max_draft: cli.draft,
                        seed: 42,
                    };
                    let mut out_bytes = Vec::new();
                    let stats = generate(&model, &mut ctx, &toks, &cfg, |_, b| {
                        out_bytes.extend_from_slice(b);
                        true
                    })?;
                    println!(
                        "[{:11}] gen={:3} tok | {:6.2} tok/s | pp={:6.1} tok/s | fwd={} | draft={}/{} ({:.1}%)",
                        mode,
                        stats.generated_tokens,
                        stats.tok_per_sec(),
                        stats.prompt_tok_per_sec(),
                        stats.forward_passes,
                        stats.draft_accepted,
                        stats.draft_proposed,
                        stats.accept_rate(),
                    );
                    if mode == "spec-lookup" {
                        let preview = String::from_utf8_lossy(&out_bytes);
                        let short: String = preview.chars().take(160).collect();
                        println!("  output: {}", short.replace('\n', " "));
                    }
                }
            }
        }
        Command::Serve { host, port } => {
            let model_name = std::path::Path::new(&cli.model)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("bitnet-b1.58-2B-4T")
                .to_string();
            let state = Arc::new(EngineState {
                model,
                ctx: Arc::new(Mutex::new(ctx)),
                model_name,
                default_draft: cli.draft,
                total_requests: AtomicU64::new(0),
                total_tokens: AtomicU64::new(0),
                last_tok_s_x100: AtomicU64::new(0),
                last_accept_x100: AtomicU64::new(0),
            });

            let app = Router::new()
                .route("/", get(index_html))
                .route("/health", get(health))
                .route("/v1/models", get(list_models))
                .route("/v1/chat/completions", post(chat_completions))
                .with_state(state);

            let addr = format!("{}:{}", host, port);
            let listener = tokio::net::TcpListener::bind(&addr).await?;
            eprintln!(
                "bitrust-zen3 listening on http://{} (t={}, tb={}, draft={}, cores={})",
                addr, cli.threads, cli.threads_batch, cli.draft, cli.cores
            );
            axum::serve(listener, app).await?;
        }
    }
    Ok(())
}
