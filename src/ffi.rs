use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::ptr::NonNull;

#[repr(C)]
pub struct LlamaModel {
    _private: [u8; 0],
}

#[repr(C)]
pub struct LlamaContext {
    _private: [u8; 0],
}

extern "C" {
    fn bn_init(verbose: c_int);
    fn bn_prewarm_hugepages(path: *const c_char);
    fn bn_load_model(path: *const c_char, use_mmap: c_int, use_mlock: c_int) -> *mut LlamaModel;
    fn bn_free_model(model: *mut LlamaModel);
    fn bn_new_context(
        model: *mut LlamaModel,
        n_ctx: u32,
        n_batch: u32,
        n_threads: i32,
        n_threads_batch: i32,
    ) -> *mut LlamaContext;
    fn bn_free_context(ctx: *mut LlamaContext);
    fn bn_n_vocab(model: *const LlamaModel) -> i32;
    fn bn_is_eog(model: *const LlamaModel, token: i32) -> i32;
    fn bn_tokenize(
        model: *const LlamaModel,
        text: *const c_char,
        text_len: i32,
        out_tokens: *mut i32,
        max_tokens: i32,
        add_special: c_int,
        parse_special: c_int,
    ) -> i32;
    fn bn_token_to_piece(
        model: *const LlamaModel,
        token: i32,
        buf: *mut c_char,
        buf_len: i32,
    ) -> i32;
    fn bn_kv_clear(ctx: *mut LlamaContext);
    fn bn_kv_rm_from(ctx: *mut LlamaContext, pos: i32) -> i32;
    fn bn_decode_tokens(
        ctx: *mut LlamaContext,
        tokens: *const i32,
        start_pos: i32,
        n_tokens: i32,
        all_logits: c_int,
    ) -> i32;
    fn bn_get_logits_ith(ctx: *mut LlamaContext, idx: i32) -> *const f32;
    fn bn_apply_chat_template(
        model: *const LlamaModel,
        roles: *const *const c_char,
        contents: *const *const c_char,
        n_msg: usize,
        add_ass: c_int,
        buf: *mut c_char,
        buf_len: i32,
    ) -> i32;
}

pub fn init_backend(verbose: bool) {
    unsafe { bn_init(if verbose { 1 } else { 0 }) };
}

pub fn pin_to_cores(cores: &[usize]) {
    if cores.is_empty() {
        return;
    }
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        for &c in cores {
            libc::CPU_SET(c, &mut set);
        }
        let _ = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

pub struct BitNetModel {
    ptr: NonNull<LlamaModel>,
    n_vocab: usize,
}

unsafe impl Send for BitNetModel {}
unsafe impl Sync for BitNetModel {}

impl BitNetModel {
    pub fn load(path: &str, use_mlock: bool) -> Result<Self, String> {
        let c_path = CString::new(path).map_err(|e| e.to_string())?;
        unsafe {
            bn_prewarm_hugepages(c_path.as_ptr());
            let ptr = bn_load_model(c_path.as_ptr(), 1, if use_mlock { 1 } else { 0 });
            let ptr = NonNull::new(ptr).ok_or_else(|| format!("Failed to load model: {}", path))?;
            let n_vocab = bn_n_vocab(ptr.as_ptr()) as usize;
            Ok(Self { ptr, n_vocab })
        }
    }

    pub fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    pub fn is_eog(&self, token: i32) -> bool {
        unsafe { bn_is_eog(self.ptr.as_ptr(), token) != 0 }
    }

    pub fn tokenize(&self, text: &str, add_special: bool, parse_special: bool) -> Result<Vec<i32>, String> {
        let max_len = text.len().saturating_add(32) as i32;
        let mut out = vec![0i32; max_len as usize];
        let n = unsafe {
            bn_tokenize(
                self.ptr.as_ptr(),
                text.as_ptr() as *const c_char,
                text.len() as i32,
                out.as_mut_ptr(),
                max_len,
                if add_special { 1 } else { 0 },
                if parse_special { 1 } else { 0 },
            )
        };
        if n < 0 {
            let need = (-n) as usize;
            out.resize(need, 0);
            let n2 = unsafe {
                bn_tokenize(
                    self.ptr.as_ptr(),
                    text.as_ptr() as *const c_char,
                    text.len() as i32,
                    out.as_mut_ptr(),
                    need as i32,
                    if add_special { 1 } else { 0 },
                    if parse_special { 1 } else { 0 },
                )
            };
            if n2 < 0 {
                return Err("Tokenization failed".to_string());
            }
            out.truncate(n2 as usize);
        } else {
            out.truncate(n as usize);
        }
        Ok(out)
    }

    pub fn token_to_bytes(&self, token: i32) -> Vec<u8> {
        let mut buf = [0u8; 128];
        let n = unsafe {
            bn_token_to_piece(
                self.ptr.as_ptr(),
                token,
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as i32,
            )
        };
        if n > 0 {
            buf[..n as usize].to_vec()
        } else if n < 0 {
            let mut big = vec![0u8; (-n) as usize];
            let n2 = unsafe {
                bn_token_to_piece(
                    self.ptr.as_ptr(),
                    token,
                    big.as_mut_ptr() as *mut c_char,
                    big.len() as i32,
                )
            };
            if n2 > 0 {
                big.truncate(n2 as usize);
                big
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        }
    }

    pub fn apply_chat_template(&self, messages: &[(String, String)], add_assistant: bool) -> String {
        let c_roles: Vec<CString> = messages
            .iter()
            .map(|(r, _)| CString::new(r.as_str()).unwrap_or_else(|_| CString::new("user").unwrap()))
            .collect();
        let c_contents: Vec<CString> = messages
            .iter()
            .map(|(_, c)| CString::new(c.as_str()).unwrap_or_else(|_| CString::new("").unwrap()))
            .collect();
        let role_ptrs: Vec<*const c_char> = c_roles.iter().map(|s| s.as_ptr()).collect();
        let content_ptrs: Vec<*const c_char> = c_contents.iter().map(|s| s.as_ptr()).collect();

        let mut buf = vec![0u8; 8192];
        let n = unsafe {
            bn_apply_chat_template(
                self.ptr.as_ptr(),
                role_ptrs.as_ptr(),
                content_ptrs.as_ptr(),
                messages.len(),
                if add_assistant { 1 } else { 0 },
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as i32,
            )
        };
        if n > 0 && (n as usize) < buf.len() {
            buf.truncate(n as usize);
            return String::from_utf8_lossy(&buf).into_owned();
        }
        if n > 0 {
            buf.resize(n as usize + 1, 0);
            let n2 = unsafe {
                bn_apply_chat_template(
                    self.ptr.as_ptr(),
                    role_ptrs.as_ptr(),
                    content_ptrs.as_ptr(),
                    messages.len(),
                    if add_assistant { 1 } else { 0 },
                    buf.as_mut_ptr() as *mut c_char,
                    buf.len() as i32,
                )
            };
            if n2 > 0 {
                buf.truncate(n2 as usize);
                return String::from_utf8_lossy(&buf).into_owned();
            }
        }
        // Fallback Llama-3 style prompt
        let mut s = String::new();
        for (r, c) in messages {
            s.push_str(&format!("<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>", r, c));
        }
        if add_assistant {
            s.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        }
        s
    }

    pub fn new_context(
        &self,
        n_ctx: u32,
        n_batch: u32,
        n_threads: i32,
        n_threads_batch: i32,
    ) -> Result<BitNetContext, String> {
        let ptr = unsafe {
            bn_new_context(
                self.ptr.as_ptr(),
                n_ctx,
                n_batch,
                n_threads,
                n_threads_batch,
            )
        };
        let ptr = NonNull::new(ptr).ok_or_else(|| "Failed to create llama context".to_string())?;
        Ok(BitNetContext { ptr })
    }
}

impl Drop for BitNetModel {
    fn drop(&mut self) {
        unsafe { bn_free_model(self.ptr.as_ptr()) };
    }
}

pub struct BitNetContext {
    ptr: NonNull<LlamaContext>,
}

unsafe impl Send for BitNetContext {}

impl BitNetContext {
    pub fn kv_clear(&mut self) {
        unsafe { bn_kv_clear(self.ptr.as_ptr()) };
    }

    pub fn kv_rm_from(&mut self, pos: i32) {
        unsafe {
            let _ = bn_kv_rm_from(self.ptr.as_ptr(), pos);
        }
    }

    pub fn decode(&mut self, tokens: &[i32], start_pos: i32, all_logits: bool) -> Result<(), String> {
        let rc = unsafe {
            bn_decode_tokens(
                self.ptr.as_ptr(),
                tokens.as_ptr(),
                start_pos,
                tokens.len() as i32,
                if all_logits { 1 } else { 0 },
            )
        };
        if rc != 0 {
            Err(format!("llama_decode failed with code {}", rc))
        } else {
            Ok(())
        }
    }

    pub fn logits_ith(&self, idx: i32, n_vocab: usize) -> &[f32] {
        unsafe {
            let ptr = bn_get_logits_ith(self.ptr.as_ptr(), idx);
            std::slice::from_raw_parts(ptr, n_vocab)
        }
    }
}

impl Drop for BitNetContext {
    fn drop(&mut self) {
        unsafe { bn_free_context(self.ptr.as_ptr()) };
    }
}

#[allow(dead_code)]
fn _cstr_unused(_: &CStr) {}
