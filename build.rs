fn main() {
    let bitnet_root = std::env::var("BITNET_ROOT")
        .unwrap_or_else(|_| "../BitNet".to_string());
    let build_dir = std::env::var("BITNET_BUILD_DIR")
        .unwrap_or_else(|_| format!("{}/build-znver3", bitnet_root));
    let target_arch = std::env::var("BITRUST_MARCH")
        .unwrap_or_else(|_| "znver3".to_string());

    let llama_inc = format!("{}/3rdparty/llama.cpp/include", bitnet_root);
    let ggml_inc = format!("{}/3rdparty/llama.cpp/ggml/include", bitnet_root);
    let llama_lib = format!("{}/3rdparty/llama.cpp/src", build_dir);
    let ggml_lib = format!("{}/3rdparty/llama.cpp/ggml/src", build_dir);

    cc::Build::new()
        .file("csrc/shim.c")
        .include(&llama_inc)
        .include(&ggml_inc)
        .flag("-O3")
        .flag(format!("-march={}", target_arch))
        .compile("bitnet_shim");

    println!("cargo:rustc-link-search=native={}", llama_lib);
    println!("cargo:rustc-link-search=native={}", ggml_lib);
    println!("cargo:rustc-link-lib=dylib=llama");
    println!("cargo:rustc-link-lib=dylib=ggml");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", llama_lib);
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", ggml_lib);
    println!("cargo:rerun-if-changed=csrc/shim.c");
}
