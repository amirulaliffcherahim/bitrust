#define _GNU_SOURCE
#include "llama.h"
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static void bn_log_silent(enum ggml_log_level level, const char *text, void *user_data) {
    (void)level;
    (void)text;
    (void)user_data;
}

void bn_init(int verbose) {
    if (!verbose) {
        llama_log_set(bn_log_silent, NULL);
    }
    llama_backend_init();
}

void bn_free_backend(void) {
    llama_backend_free();
}

void bn_prewarm_hugepages(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return;
    struct stat st;
    if (fstat(fd, &st) == 0 && st.st_size > 0) {
        void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
        if (p != MAP_FAILED) {
#ifdef MADV_HUGEPAGE
            madvise(p, (size_t)st.st_size, MADV_HUGEPAGE);
#endif
            madvise(p, (size_t)st.st_size, MADV_WILLNEED);
            munmap(p, (size_t)st.st_size);
        }
    }
    close(fd);
}

struct llama_model *bn_load_model(const char *path, int use_mmap, int use_mlock) {
    struct llama_model_params params = llama_model_default_params();
    params.use_mmap = use_mmap != 0;
    params.use_mlock = use_mlock != 0;
    return llama_load_model_from_file(path, params);
}

void bn_free_model(struct llama_model *model) {
    if (model) llama_free_model(model);
}

struct llama_context *bn_new_context(
    struct llama_model *model,
    uint32_t n_ctx,
    uint32_t n_batch,
    int32_t n_threads,
    int32_t n_threads_batch
) {
    struct llama_context_params params = llama_context_default_params();
    params.n_ctx = n_ctx;
    params.n_batch = n_batch;
    params.n_ubatch = n_batch;
    params.n_threads = n_threads;
    params.n_threads_batch = n_threads_batch;
    params.flash_attn = true;
    return llama_new_context_with_model(model, params);
}

void bn_free_context(struct llama_context *ctx) {
    if (ctx) llama_free(ctx);
}

int32_t bn_n_vocab(const struct llama_model *model) {
    return llama_n_vocab(model);
}

int32_t bn_is_eog(const struct llama_model *model, int32_t token) {
    return llama_token_is_eog(model, (llama_token)token) ? 1 : 0;
}

int32_t bn_tokenize(
    const struct llama_model *model,
    const char *text,
    int32_t text_len,
    int32_t *out_tokens,
    int32_t max_tokens,
    int add_special,
    int parse_special
) {
    return llama_tokenize(model, text, text_len, (llama_token *)out_tokens, max_tokens, add_special != 0, parse_special != 0);
}

int32_t bn_token_to_piece(
    const struct llama_model *model,
    int32_t token,
    char *buf,
    int32_t buf_len
) {
    return llama_token_to_piece(model, (llama_token)token, buf, buf_len, 0, false);
}

void bn_kv_clear(struct llama_context *ctx) {
    llama_kv_cache_clear(ctx);
}

int32_t bn_kv_rm_from(struct llama_context *ctx, int32_t pos) {
    return llama_kv_cache_seq_rm(ctx, 0, (llama_pos)pos, -1) ? 1 : 0;
}

int32_t bn_decode_tokens(
    struct llama_context *ctx,
    const int32_t *tokens,
    int32_t start_pos,
    int32_t n_tokens,
    int all_logits
) {
    struct llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (int32_t i = 0; i < n_tokens; i++) {
        batch.token[i] = (llama_token)tokens[i];
        batch.pos[i] = (llama_pos)(start_pos + i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (all_logits || i == n_tokens - 1) ? 1 : 0;
    }
    int32_t rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return rc;
}

const float *bn_get_logits_ith(struct llama_context *ctx, int32_t idx) {
    return llama_get_logits_ith(ctx, idx);
}

int32_t bn_apply_chat_template(
    const struct llama_model *model,
    const char **roles,
    const char **contents,
    size_t n_msg,
    int add_ass,
    char *buf,
    int32_t buf_len
) {
    struct llama_chat_message *msgs = (struct llama_chat_message *)calloc(n_msg, sizeof(struct llama_chat_message));
    if (!msgs) return -1;
    for (size_t i = 0; i < n_msg; i++) {
        msgs[i].role = roles[i];
        msgs[i].content = contents[i];
    }
    int32_t res = llama_chat_apply_template(model, NULL, msgs, n_msg, add_ass != 0, buf, buf_len);
    free(msgs);
    return res;
}
