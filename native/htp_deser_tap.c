// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Chromatic and contributors

#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/*
 * Universal QNN HTP V73 Deserializer & Execution Layer-Tap Hook (LD_PRELOAD).
 *
 * Targets libQnnHtp.so (QNN 2.39 x86_64 host-emu runtime) via dlopen + PLT interposition:
 *  - Handles RTLD_LOCAL by intercepting dlopen("...libQnnHtp.so", ...) and resolving
 *    internal symbols from the captured handle.
 *  - Intercepts hnnx::Deserializer const-extent & Graph auxdata calls -> extents.jsonl.
 *  - Intercepts Graph::do_execute -> dumps ordered runlist & tensor metadata -> runlist.jsonl (+ consts.bin).
 *  - Intercepts continue_execution_fast & continue_execution_bkgrnd_thread -> untiles & dumps
 *    every op's output activation immediately after execution -> taps_index.jsonl + taps.bin.
 */

typedef struct {
    uint32_t rank;
    uint32_t dtype;
    uint64_t dims[8];
    int32_t zero_offset;
    float scale;
} OutputDef;

typedef struct {
    uint64_t n_in;
    uint64_t n_out;
} IoCounts;

typedef struct {
    int (*fn)(void *obj, void *graph, uint32_t oss);
    void *obj;
    volatile int *dep_count;
    volatile int *notify_ptr;
} ExecType;

typedef struct {
    uint32_t *beg;
    uint32_t *end;
    uint32_t *cap;
} VecU32;

typedef struct {
    void *op_ptr;
    int idx;
} OpEntry;

typedef struct {
    void *exec_obj;
    void *op_ptr;
    int idx;
} ExecMapEntry;

static pthread_mutex_t g_init_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_log_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_tap_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_exec_map_mu = PTHREAD_MUTEX_INITIALIZER;

static void *(*real_dlopen)(const char *, int) = NULL;
static void *g_htp_handle = NULL;

/* Real libQnnHtp.so entry points resolved via dlsym(g_htp_handle, ...) */
static void *(*real_extract_const_extent_table)(void *out_vec, void *deser, unsigned long off) = NULL;
static int (*real_auxdata_read_const_extent_descriptor)(void *deser, unsigned int nwords) = NULL;
static int (*real_handle_auxdata_deser)(void *graph, void *deser, unsigned int a, unsigned int b) = NULL;
static int (*real_extract_const_extent_data_dma)(void *deser, unsigned long a, unsigned long b, void *dst, void *dma) = NULL;
static int (*real_extract_const_extent_data_mm)(void *deser, unsigned long a, unsigned long b, void *dst, unsigned long c, unsigned long d) = NULL;
static void (*real_compile_exec_list)(void *graph, void *rl, int ltype, int start, int end) = NULL;
static int (*real_do_execute)(void *graph, void *env) = NULL;
static int (*real_continue_fast)(void *graph, void *exec_info, ExecType *list, uint32_t num_ops,
                                 uint32_t *idx_ptr, int *abort_flag, uint32_t *wait_cycles,
                                 uint32_t thread_id, uint64_t oss) = NULL;
static int (*real_continue_bkgrnd)(void *graph, void *exec_info, ExecType *list, uint32_t num_ops,
                                   uint32_t *idx_ptr, int *abort_flag, uint32_t thread_id, uint64_t oss) = NULL;

/* Inspection helpers inside libQnnHtp.so */
static const void *(*fn_get_extra_info)(const void *graph, const void *op) = NULL;
static const char *(*fn_op_true_name)(const void *op) = NULL;
static const char *(*fn_op_true_func)(const void *op) = NULL;
static uint8_t (*fn_op_is_const)(const void *op) = NULL;
static void (*fn_tensor_gen_output_def)(OutputDef *out, const void *tensor) = NULL;

/* Output files & configuration */
static int g_inited = 0;
static char g_tap_dir[512] = "";
static FILE *g_extents_fp = NULL;
static FILE *g_taps_idx_fp = NULL;
static FILE *g_taps_bin_fp = NULL;
static uint64_t g_taps_bin_off = 0;
static FILE *g_consts_idx_fp = NULL;
static int g_consts_bin_fd = -1;
static uint64_t g_consts_bin_off = 0;

static int g_dump_consts = 0;
static int g_skip_const_taps = 1;
static int g_skip_repeat_exec = 0;
static int g_skip_exec = 0;
static int g_mode_deser_only = 0;
static uint64_t g_max_tap_bytes = 0; /* 0 = unlimited */

/* Execution capture state */
static volatile int g_capturing = 0;
static int g_exec_count = 0;
static void *g_cap_graph = NULL;
static OpEntry *g_op_entries = NULL;
static size_t g_n_ops = 0;
static ExecMapEntry *g_exec_map = NULL;
static size_t g_exec_map_n = 0;
static size_t g_exec_map_cap = 0;
static ExecMapEntry *g_exec_lookup = NULL;
static size_t g_exec_lookup_n = 0;
static uint64_t g_tap_seq = 0;

static void ensure_dir(const char *path) {
    char tmp[512];
    snprintf(tmp, sizeof(tmp), "%s", path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            mkdir(tmp, 0755);
            *p = '/';
        }
    }
    mkdir(tmp, 0755);
}

static void init_config(void) {
    pthread_mutex_lock(&g_init_mu);
    if (!g_inited) {
        const char *d = getenv("HTP_TAP_DIR");
        if (d && d[0]) {
            snprintf(g_tap_dir, sizeof(g_tap_dir), "%s", d);
            ensure_dir(g_tap_dir);
        }
        const char *m = getenv("HTP_TAP_MODE");
        if (m && (strcmp(m, "deser") == 0 || strcmp(m, "none") == 0)) {
            g_mode_deser_only = 1;
        }
        const char *dc = getenv("HTP_TAP_DUMP_CONSTS");
        if (dc && strcmp(dc, "1") == 0) g_dump_consts = 1;
        const char *sc = getenv("HTP_TAP_SKIP_CONST");
        if (sc && strcmp(sc, "0") == 0) g_skip_const_taps = 0;
        const char *sr = getenv("HTP_TAP_SKIP_REPEAT");
        if (sr && strcmp(sr, "1") == 0) g_skip_repeat_exec = 1;
        const char *se = getenv("HTP_TAP_SKIP_EXEC");
        if (se && strcmp(se, "1") == 0) g_skip_exec = 1;
        const char *mb = getenv("HTP_TAP_MAX_TAP_BYTES");
        if (mb && mb[0]) g_max_tap_bytes = strtoull(mb, NULL, 0);

        if (g_tap_dir[0]) {
            char p[600];
            snprintf(p, sizeof(p), "%s/extents.jsonl", g_tap_dir);
            g_extents_fp = fopen(p, "w");
            if (!g_mode_deser_only) {
                snprintf(p, sizeof(p), "%s/taps_index.jsonl", g_tap_dir);
                g_taps_idx_fp = fopen(p, "w");
                if (g_taps_idx_fp) setvbuf(g_taps_idx_fp, NULL, _IOFBF, 1 << 20);
                snprintf(p, sizeof(p), "%s/taps.bin", g_tap_dir);
                g_taps_bin_fp = fopen(p, "wb");
                if (g_taps_bin_fp) setvbuf(g_taps_bin_fp, NULL, _IOFBF, 4 << 20);
            }
            if (g_dump_consts) {
                snprintf(p, sizeof(p), "%s/consts_index.jsonl", g_tap_dir);
                g_consts_idx_fp = fopen(p, "w");
                snprintf(p, sizeof(p), "%s/consts.bin", g_tap_dir);
                g_consts_bin_fd = open(p, O_WRONLY | O_CREAT | O_TRUNC, 0644);
            }
        }
        g_inited = 1;
    }
    pthread_mutex_unlock(&g_init_mu);
}

static void write_all_fd(int fd, const void *buf, size_t len) {
    const uint8_t *p = (const uint8_t *)buf;
    size_t left = len;
    while (left > 0) {
        ssize_t w = write(fd, p, left);
        if (w < 0) {
            if (errno == EINTR) continue;
            break;
        }
        p += w;
        left -= (size_t)w;
    }
}

static void fprint_json_str(FILE *fp, const char *s) {
    fputc('"', fp);
    if (s) {
        for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
            unsigned char c = *p;
            if (c == '"') fputs("\\\"", fp);
            else if (c == '\\') fputs("\\\\", fp);
            else if (c == '\n') fputs("\\n", fp);
            else if (c == '\r') fputs("\\r", fp);
            else if (c == '\t') fputs("\\t", fp);
            else if (c < 0x20) fprintf(fp, "\\u%04x", c);
            else fputc(c, fp);
        }
    }
    fputc('"', fp);
}

static size_t dtype_elem_bytes(uint32_t dt) {
    switch (dt) {
        case 1: return 1; /* QUint8 */
        case 2: return 2; /* QUint16 */
        case 3: return 2; /* QInt16 */
        case 4: return 4; /* Float32 */
        case 5: return 4; /* Int32 / QFloat */
        case 6: return 4; /* QInt32 */
        case 7: return 1; /* QInt8 */
        case 8: return 2; /* Float16 */
        case 9: return 8; /* Int64 */
        case 10: return 2; /* BFloat16 */
        default: return 0;
    }
}

static const char *dtype_name(uint32_t dt) {
    switch (dt) {
        case 1: return "QUint8";
        case 2: return "QUint16";
        case 3: return "QInt16";
        case 4: return "Float32";
        case 5: return "Int32";
        case 6: return "QInt32";
        case 7: return "QInt8";
        case 8: return "Float16";
        case 9: return "Int64";
        case 10: return "BFloat16";
        case 0xfe: return "None";
        default: return "Unknown";
    }
}

static void resolve_htp_symbols(void *h) {
    real_extract_const_extent_table = dlsym(h, "_ZN4hnnx12Deserializer26extract_const_extent_tableEm");
    real_auxdata_read_const_extent_descriptor = dlsym(h, "_ZN4hnnx12Deserializer36auxdata_read_const_extent_descriptorEj");
    real_handle_auxdata_deser = dlsym(h, "_ZN5Graph20handle_auxdata_deserERN4hnnx12DeserializerEjj");
    real_extract_const_extent_data_dma = dlsym(h, "_ZN4hnnx12Deserializer25extract_const_extent_dataEmmPvPNS_11DMA_ManagerE");
    real_extract_const_extent_data_mm = dlsym(h, "_ZN4hnnx12Deserializer25extract_const_extent_dataEmmPvmm");
    real_compile_exec_list = dlsym(h, "_ZN5Graph17compile_exec_listERN9GraphData10RunListSetENS0_8ListTypeEii");
    real_do_execute = dlsym(h, "_ZN5Graph10do_executeER12HexagonNNEnv");
    real_continue_fast = dlsym(h, "continue_execution_fast");
    real_continue_bkgrnd = dlsym(h, "continue_execution_bkgrnd_thread");

    fn_get_extra_info = dlsym(h, "_ZNK5Graph14get_extra_infoEPK2Op");
    fn_op_true_name = dlsym(h, "_ZNK2Op9true_nameEv");
    fn_op_true_func = dlsym(h, "_ZNK2Op9true_funcEv");
    fn_op_is_const = dlsym(h, "_ZNK2Op8is_constEv");
    fn_tensor_gen_output_def = dlsym(h, "_ZNK6Tensor14gen_output_defEv");
}

void *dlopen(const char *filename, int flags) {
    if (!real_dlopen) {
        real_dlopen = (void *(*)(const char *, int))dlsym(RTLD_NEXT, "dlopen");
    }
    init_config();
    void *h = real_dlopen(filename, flags);
    if (h && filename && strstr(filename, "libQnnHtp.so") && !g_htp_handle) {
        g_htp_handle = h;
        resolve_htp_symbols(h);
    }
    return h;
}

/* --- Deserializer PLT hooks --- */

void *_ZN4hnnx12Deserializer26extract_const_extent_tableEm(void *out_vec, void *deser, unsigned long off) {
    init_config();
    if (!real_extract_const_extent_table) return out_vec;
    void *ret = real_extract_const_extent_table(out_vec, deser, off);
    VecU32 *v = (VecU32 *)out_vec;
    if (g_extents_fp && v && v->beg && v->end && v->end >= v->beg + 4) {
        size_t nwords = (size_t)(v->end - v->beg);
        uint32_t magic = v->beg[0];
        uint32_t hdr1 = v->beg[1];
        uint32_t n_groups = v->beg[2] & 0xffffffu;
        uint32_t n_extents = v->beg[3] & 0xffffffu;
        uint32_t hdr_items = hdr1 >> 24;
        pthread_mutex_lock(&g_log_mu);
        fprintf(g_extents_fp, "{\"event\":\"const_extent_table\",\"off\":%lu,\"nwords\":%zu,\"magic\":\"0x%08x\",\"hdr1\":\"0x%08x\",\"hdr_items\":%u,\"n_groups\":%u,\"n_extents\":%u,\"groups\":[",
                off, nwords, magic, hdr1, hdr_items, n_groups, n_extents);
        size_t g_start = (size_t)hdr_items * 4;
        for (uint32_t g = 0; g < n_groups && g_start + (g + 1) * 4 <= nwords; g++) {
            const uint32_t *p = &v->beg[g_start + g * 4];
            fprintf(g_extents_fp, "%s[%u,%u,%u,%u]", g ? "," : "", p[0], p[1], p[2], p[3]);
        }
        fprintf(g_extents_fp, "],\"extents\":[");
        size_t e_start = g_start + (size_t)n_groups * 4;
        for (uint32_t e = 0; e < n_extents && e_start + (e + 1) * 4 <= nwords; e++) {
            const uint32_t *p = &v->beg[e_start + e * 4];
            fprintf(g_extents_fp, "%s[%u,%u,%u,%u]", e ? "," : "", p[0], p[1], p[2], p[3]);
        }
        fprintf(g_extents_fp, "]}\n");
        fflush(g_extents_fp);
        pthread_mutex_unlock(&g_log_mu);
    }
    return ret;
}

int _ZN4hnnx12Deserializer36auxdata_read_const_extent_descriptorEj(void *deser, unsigned int nwords) {
    init_config();
    char name_buf[64];
    name_buf[0] = '\0';
    int have_name = 0;
    if (deser && nwords >= 12) {
        const uint8_t *base = *(const uint8_t *const *)((const uint8_t *)deser + 0x50);
        const uint8_t *cur = *(const uint8_t *const *)((const uint8_t *)deser + 0x60);
        const uint8_t *end = *(const uint8_t *const *)((const uint8_t *)deser + 0x58);
        if (base && cur && end && cur >= base) {
            size_t rel = (size_t)(cur - base) & ~3ULL;
            const uint8_t *aligned = base + rel;
            if (aligned + (size_t)nwords * 4 <= end) {
                uint32_t marker = *(const uint32_t *)(aligned + ((size_t)nwords - 1) * 4);
                if (marker == 0xebbe003eu) {
                    const char *src = (const char *)(aligned + (size_t)nwords * 4 - 0x30);
                    memcpy(name_buf, src, 45);
                    name_buf[45] = '\0';
                    have_name = 1;
                }
            }
        }
    }
    int rc = real_auxdata_read_const_extent_descriptor ? real_auxdata_read_const_extent_descriptor(deser, nwords) : -1;
    if (g_extents_fp) {
        pthread_mutex_lock(&g_log_mu);
        fprintf(g_extents_fp, "{\"event\":\"const_extent_desc\",\"nwords\":%u,\"rc\":%d,\"name\":", nwords, rc);
        if (have_name) fprint_json_str(g_extents_fp, name_buf);
        else fputs("null", g_extents_fp);
        fprintf(g_extents_fp, "}\n");
        fflush(g_extents_fp);
        pthread_mutex_unlock(&g_log_mu);
    }
    return rc;
}

int _ZN5Graph20handle_auxdata_deserERN4hnnx12DeserializerEjj(void *graph, void *deser, unsigned int a, unsigned int b) {
    init_config();
    int rc = real_handle_auxdata_deser ? real_handle_auxdata_deser(graph, deser, a, b) : -1;
    if (g_extents_fp) {
        pthread_mutex_lock(&g_log_mu);
        fprintf(g_extents_fp, "{\"event\":\"auxdata_deser\",\"graph\":\"%p\",\"a\":%u,\"b\":%u,\"rc\":%d}\n", graph, a, b, rc);
        fflush(g_extents_fp);
        pthread_mutex_unlock(&g_log_mu);
    }
    return rc;
}

int _ZN4hnnx12Deserializer25extract_const_extent_dataEmmPvPNS_11DMA_ManagerE(
    void *deser, unsigned long a, unsigned long b, void *dst, void *dma) {
    init_config();
    int rc = real_extract_const_extent_data_dma ? real_extract_const_extent_data_dma(deser, a, b, dst, dma) : -1;
    if (g_extents_fp) {
        pthread_mutex_lock(&g_log_mu);
        fprintf(g_extents_fp, "{\"event\":\"extract_const_extent_data_dma\",\"a\":%lu,\"b\":%lu,\"dst\":\"%p\",\"dma\":\"%p\",\"rc\":%d}\n",
                a, b, dst, dma, rc);
        fflush(g_extents_fp);
        pthread_mutex_unlock(&g_log_mu);
    }
    return rc;
}

int _ZN4hnnx12Deserializer25extract_const_extent_dataEmmPvmm(
    void *deser, unsigned long a, unsigned long b, void *dst, unsigned long c, unsigned long d) {
    init_config();
    int rc = real_extract_const_extent_data_mm ? real_extract_const_extent_data_mm(deser, a, b, dst, c, d) : -1;
    if (g_extents_fp) {
        pthread_mutex_lock(&g_log_mu);
        fprintf(g_extents_fp, "{\"event\":\"extract_const_extent_data_mm\",\"a\":%lu,\"b\":%lu,\"dst\":\"%p\",\"c\":%lu,\"d\":%lu,\"rc\":%d}\n",
                a, b, dst, c, d, rc);
        fflush(g_extents_fp);
        pthread_mutex_unlock(&g_log_mu);
    }
    return rc;
}

/* --- compile_exec_list PLT hook: tracks ExecType.obj -> Op* --- */

static void record_exec_op(void *exec_obj, void *op_ptr) {
    if (!exec_obj || !op_ptr) return;
    pthread_mutex_lock(&g_exec_map_mu);
    if (g_exec_map_n >= g_exec_map_cap) {
        size_t ncap = g_exec_map_cap ? g_exec_map_cap * 2 : 4096;
        ExecMapEntry *narr = (ExecMapEntry *)realloc(g_exec_map, ncap * sizeof(ExecMapEntry));
        if (!narr) {
            pthread_mutex_unlock(&g_exec_map_mu);
            return;
        }
        g_exec_map = narr;
        g_exec_map_cap = ncap;
    }
    g_exec_map[g_exec_map_n].exec_obj = exec_obj;
    g_exec_map[g_exec_map_n].op_ptr = op_ptr;
    g_exec_map[g_exec_map_n].idx = -1;
    g_exec_map_n++;
    pthread_mutex_unlock(&g_exec_map_mu);
}

void _ZN5Graph17compile_exec_listERN9GraphData10RunListSetENS0_8ListTypeEii(
    void *graph, void *rl, int ltype, int start, int end) {
    init_config();
    if (!real_compile_exec_list) return;
    if (!rl || start >= end) {
        real_compile_exec_list(graph, rl, ltype, start, end);
        return;
    }
    ExecType **p_exec_beg = (ExecType **)((uint8_t *)rl + 0x00);
    ExecType **p_exec_end = (ExecType **)((uint8_t *)rl + 0x08);
    void ***p_op_beg = (void ***)((uint8_t *)rl + 0x18);
    void ***p_op_end = (void ***)((uint8_t *)rl + 0x20);

    for (int i = start; i < end; i++) {
        void **op_beg = *p_op_beg;
        void **op_end = *p_op_end;
        size_t n_ops_in_rl = (op_beg && op_end && op_end > op_beg) ? (size_t)(op_end - op_beg) : 0;
        void *op_ptr = (i >= 0 && (size_t)i < n_ops_in_rl) ? op_beg[i] : NULL;

        ExecType *eb0 = *p_exec_beg;
        ExecType *ee0 = *p_exec_end;
        size_t prev_cnt = (eb0 && ee0 && ee0 >= eb0) ? (size_t)(ee0 - eb0) : 0;

        real_compile_exec_list(graph, rl, ltype, i, i + 1);

        ExecType *eb1 = *p_exec_beg;
        ExecType *ee1 = *p_exec_end;
        size_t cur_cnt = (eb1 && ee1 && ee1 >= eb1) ? (size_t)(ee1 - eb1) : 0;
        if (op_ptr && cur_cnt == prev_cnt + 1) {
            void *exec_obj = eb1[prev_cnt].obj;
            if (exec_obj && exec_obj != op_ptr) {
                record_exec_op(exec_obj, op_ptr);
            }
        }
    }
}

/* --- Runlist & Tensor Untiling Helpers --- */

static int cmp_op_entry(const void *a, const void *b) {
    const OpEntry *ea = (const OpEntry *)a;
    const OpEntry *eb = (const OpEntry *)b;
    uintptr_t pa = (uintptr_t)ea->op_ptr;
    uintptr_t pb = (uintptr_t)eb->op_ptr;
    return (pa > pb) - (pa < pb);
}

static int find_op_idx(const void *op_ptr) {
    if (!g_op_entries || g_n_ops == 0) return -1;
    OpEntry key = { (void *)op_ptr, 0 };
    OpEntry *found = (OpEntry *)bsearch(&key, g_op_entries, g_n_ops, sizeof(OpEntry), cmp_op_entry);
    return found ? found->idx : -1;
}

static int cmp_exec_entry(const void *a, const void *b) {
    const ExecMapEntry *ea = (const ExecMapEntry *)a;
    const ExecMapEntry *eb = (const ExecMapEntry *)b;
    uintptr_t pa = (uintptr_t)ea->exec_obj;
    uintptr_t pb = (uintptr_t)eb->exec_obj;
    return (pa > pb) - (pa < pb);
}

static const void *find_op_for_exec(const void *exec_obj, int *out_idx) {
    if (g_exec_lookup && g_exec_lookup_n > 0) {
        ExecMapEntry key = { (void *)exec_obj, NULL, -1 };
        ExecMapEntry *found = (ExecMapEntry *)bsearch(&key, g_exec_lookup, g_exec_lookup_n, sizeof(ExecMapEntry), cmp_exec_entry);
        if (found) {
            *out_idx = found->idx;
            return found->op_ptr;
        }
    }
    int idx = find_op_idx(exec_obj);
    if (idx >= 0) {
        *out_idx = idx;
        return exec_obj;
    }
    *out_idx = -1;
    return NULL;
}

/*
 * Extract tensor bytes into a newly malloc'd buffer (caller frees).
 * Returns byte count written to *out_buf, or 0 if unavailable / FakeTensor.
 * *out_encoding receives a short static label describing the buffer layout.
 */
static size_t extract_tensor_bytes(const void *tensor, const OutputDef *od, const char *tname,
                                   uint32_t fmt_code, uint32_t tile_bits,
                                   uint8_t **out_buf, const char **out_encoding) {
    *out_buf = NULL;
    *out_encoding = "none";
    if (!tensor || !tname || strcmp(tname, "FakeTensor") == 0 || od->dtype == 0xfe) {
        return 0;
    }
    void **tvt = *(void ***)tensor;
    size_t sbytes = ((size_t (*)(const void *))tvt[23])(tensor);
    void *rdata = ((void *(*)(const void *))tvt[20])(tensor);
    if (!rdata || sbytes == 0) return 0;

    size_t ebytes = dtype_elem_bytes(od->dtype);
    uint64_t numel = 1;
    for (uint32_t d = 0; d < od->rank && d < 8; d++) numel *= od->dims[d];
    if (od->rank == 0) numel = 1;

    uint32_t layout_byte = (fmt_code >> 16) & 0xffu;
    int is_crouton = (strstr(tname, "Crouton") != NULL) || (layout_byte >= 0x1c && layout_byte <= 0x2f);

    if (!is_crouton) {
        /* Flat / Plain / Quant non-crouton tensor */
        size_t copy_bytes = sbytes;
        if (ebytes > 0 && numel > 0 && numel * ebytes <= sbytes) {
            copy_bytes = (size_t)(numel * ebytes);
        }
        uint8_t *buf = (uint8_t *)malloc(copy_bytes);
        if (!buf) return 0;
        memcpy(buf, rdata, copy_bytes);
        *out_buf = buf;
        *out_encoding = "flat_rowmajor";
        return copy_bytes;
    }

    /* Crouton layouts with TensorShape at +0x18 and blocktab at +0x28 */
    const uint8_t *sh = *(const uint8_t *const *)((const uint8_t *)tensor + 0x18);
    void *const *btab = *(void *const *const *)((const uint8_t *)tensor + 0x28);
    int is_special_crouton = (strchr(tname, '#') != NULL || strstr(tname, "2x2") != NULL ||
                              strstr(tname, "4x1") != NULL || strstr(tname, "Wide") != NULL ||
                              strstr(tname, "Batched") != NULL);
    int is_ar4 = (strcmp(tname, "a;4H") == 0 || strcmp(tname, "A;4H") == 0 || strstr(tname, "AR4") != NULL);
    int is_ar8 = (strcmp(tname, "a;8H") == 0 || strcmp(tname, "A;8H") == 0 || strstr(tname, "AR8") != NULL);

    if (sh && btab && !is_special_crouton && od->rank == 4 && ebytes > 0) {
        uint64_t B = od->dims[0], H = od->dims[1], W = od->dims[2], C = od->dims[3];
        uint64_t max_h = *(const uint64_t *)(sh + 0x40);
        uint64_t max_w = *(const uint64_t *)(sh + 0x48);
        uint64_t max_c = *(const uint64_t *)(sh + 0x50);
        uint8_t pad_b = *(const uint8_t *)(sh + 0x58);
        uint8_t pad_h = *(const uint8_t *)(sh + 0x59);
        uint8_t pad_w = *(const uint8_t *)(sh + 0x5a);
        uint8_t pad_c = *(const uint8_t *)(sh + 0x5b);

        if (B > 0 && H > 0 && W > 0 && C > 0 && max_h > 0 && max_w > 0 && max_c > 0) {
            size_t out_bytes = (size_t)(B * H * W * C * ebytes);
            if (ebytes == 1 && (sbytes % 2048 == 0)) {
                size_t nblk = sbytes / 2048;
                uint64_t mh = max_h >> 3, mw = max_w >> 3, mc = max_c >> 5;
                uint64_t max_blk = (((B - 1 + pad_b) * mh + ((H - 1 + pad_h) >> 3)) * mw + ((W - 1 + pad_w) >> 3)) * mc + ((C - 1 + pad_c) >> 5);
                if (mh > 0 && mw > 0 && mc > 0 && max_blk < nblk) {
                    uint8_t *buf = (uint8_t *)malloc(out_bytes);
                    if (!buf) return 0;
                    for (uint64_t b = 0; b < B; b++) {
                        uint64_t bb = b + pad_b;
                        for (uint64_t h = 0; h < H; h++) {
                            uint64_t hh = h + pad_h;
                            for (uint64_t w = 0; w < W; w++) {
                                uint64_t ww = w + pad_w;
                                uint64_t blk_base = ((bb * mh + (hh >> 3)) * mw + (ww >> 3)) * mc;
                                uint64_t off_hw = ((hh & 7) << 8) | ((ww & 7) << 5);
                                uint8_t *dst = buf + ((b * H + h) * W + w) * C;
                                for (uint64_t c = 0; c < C; c++) {
                                    uint64_t cc = c + pad_c;
                                    const uint8_t *bp = (const uint8_t *)btab[blk_base + (cc >> 5)];
                                    dst[c] = bp ? bp[off_hw | (cc & 31)] : 0;
                                }
                            }
                        }
                    }
                    *out_buf = buf;
                    *out_encoding = "untiled_nhwc";
                    return out_bytes;
                }
            } else if (ebytes == 2 && is_ar4) {
                uint64_t mw = max_w >> 2;
                if (mw > 0) {
                    uint16_t *buf = (uint16_t *)malloc(out_bytes);
                    if (!buf) return 0;
                    for (uint64_t b = 0; b < B; b++) {
                        uint64_t bb = b + pad_b;
                        for (uint64_t h = 0; h < H; h++) {
                            uint64_t hh = h + pad_h;
                            for (uint64_t w = 0; w < W; w++) {
                                uint64_t ww = w + pad_w;
                                uint64_t blk = (bb * max_h + hh) * mw + (ww >> 2);
                                const uint16_t *bp = (const uint16_t *)btab[blk];
                                uint64_t off_w = ((ww & 2) << 5) | (ww & 1);
                                uint16_t *dst = buf + ((b * H + h) * W + w) * C;
                                for (uint64_t c = 0; c < C; c++) {
                                    uint64_t cc = c + pad_c;
                                    dst[c] = bp ? bp[((cc & ~31ULL) << 2) | off_w | ((cc & 31) << 1)] : 0;
                                }
                            }
                        }
                    }
                    *out_buf = (uint8_t *)buf;
                    *out_encoding = "untiled_nhwc";
                    return out_bytes;
                }
            } else if (ebytes == 2 && is_ar8) {
                uint64_t mw = max_w >> 3;
                if (mw > 0) {
                    uint16_t *buf = (uint16_t *)malloc(out_bytes);
                    if (!buf) return 0;
                    for (uint64_t b = 0; b < B; b++) {
                        uint64_t bb = b + pad_b;
                        for (uint64_t h = 0; h < H; h++) {
                            uint64_t hh = h + pad_h;
                            for (uint64_t w = 0; w < W; w++) {
                                uint64_t ww = w + pad_w;
                                uint64_t blk = (bb * max_h + hh) * mw + (ww >> 3);
                                const uint16_t *bp = (const uint16_t *)btab[blk];
                                uint64_t off_w = ((ww & 6) << 5) | (ww & 1);
                                uint16_t *dst = buf + ((b * H + h) * W + w) * C;
                                for (uint64_t c = 0; c < C; c++) {
                                    uint64_t cc = c + pad_c;
                                    dst[c] = bp ? bp[((cc & ~31ULL) << 3) | off_w | ((cc & 31) << 1)] : 0;
                                }
                            }
                        }
                    }
                    *out_buf = (uint8_t *)buf;
                    *out_encoding = "untiled_nhwc";
                    return out_bytes;
                }
            } else if (ebytes == 2 && (sbytes % 2048 == 0)) {
                size_t nblk = sbytes / 2048;
                uint64_t mh = max_h >> 3, mw = max_w >> 2, mc = max_c >> 5;
                uint64_t max_blk = (((B - 1 + pad_b) * mh + ((H - 1 + pad_h) >> 3)) * mw + ((W - 1 + pad_w) >> 2)) * mc + ((C - 1 + pad_c) >> 5);
                if (mh > 0 && mw > 0 && mc > 0 && max_blk < nblk) {
                    uint16_t *buf = (uint16_t *)malloc(out_bytes);
                    if (!buf) return 0;
                    for (uint64_t b = 0; b < B; b++) {
                        uint64_t bb = b + pad_b;
                        for (uint64_t h = 0; h < H; h++) {
                            uint64_t hh = h + pad_h;
                            for (uint64_t w = 0; w < W; w++) {
                                uint64_t ww = w + pad_w;
                                uint64_t blk_base = ((bb * mh + (hh >> 3)) * mw + (ww >> 2)) * mc;
                                uint64_t off_hw = ((hh & 7) << 7) | ((ww & 2) << 5) | (ww & 1);
                                uint16_t *dst = buf + ((b * H + h) * W + w) * C;
                                for (uint64_t c = 0; c < C; c++) {
                                    uint64_t cc = c + pad_c;
                                    const uint16_t *bp = (const uint16_t *)btab[blk_base + (cc >> 5)];
                                    dst[c] = bp ? bp[off_hw | ((cc & 31) << 1)] : 0;
                                }
                            }
                        }
                    }
                    *out_buf = (uint8_t *)buf;
                    *out_encoding = "untiled_nhwc";
                    return out_bytes;
                }
            } else if (ebytes == 4 && (sbytes % 2048 == 0)) {
                size_t nblk = sbytes / 2048;
                uint64_t mh = max_h >> 3, mw = max_w >> 1, mc = max_c >> 5;
                uint64_t max_blk = (((B - 1 + pad_b) * mh + ((H - 1 + pad_h) >> 3)) * mw + ((W - 1 + pad_w) >> 1)) * mc + ((C - 1 + pad_c) >> 5);
                if (mh > 0 && mw > 0 && mc > 0 && max_blk < nblk) {
                    uint32_t *buf = (uint32_t *)malloc(out_bytes);
                    if (!buf) return 0;
                    for (uint64_t b = 0; b < B; b++) {
                        uint64_t bb = b + pad_b;
                        for (uint64_t h = 0; h < H; h++) {
                            uint64_t hh = h + pad_h;
                            for (uint64_t w = 0; w < W; w++) {
                                uint64_t ww = w + pad_w;
                                uint64_t blk_base = ((bb * mh + (hh >> 3)) * mw + (ww >> 1)) * mc;
                                uint64_t off_hw = ((hh & 7) << 6) | ((ww & 1) << 5);
                                uint32_t *dst = buf + ((b * H + h) * W + w) * C;
                                for (uint64_t c = 0; c < C; c++) {
                                    uint64_t cc = c + pad_c;
                                    const uint32_t *bp = (const uint32_t *)btab[blk_base + (cc >> 5)];
                                    dst[c] = bp ? bp[off_hw | (cc & 31)] : 0;
                                }
                            }
                        }
                    }
                    *out_buf = (uint8_t *)buf;
                    *out_encoding = "untiled_nhwc";
                    return out_bytes;
                }
            }
        }
    }

    /* Fallback: dump raw 2048-byte crouton blocks sequentially */
    if (btab && (sbytes % 2048 == 0) && tile_bits != 0) {
        size_t nblk = sbytes / 2048;
        uint8_t *buf = (uint8_t *)malloc(sbytes);
        if (!buf) return 0;
        for (size_t k = 0; k < nblk; k++) {
            if (btab[k]) memcpy(buf + k * 2048, btab[k], 2048);
            else memset(buf + k * 2048, 0, 2048);
        }
        *out_buf = buf;
        *out_encoding = "crouton_raw_blocks";
        return sbytes;
    }

    return 0;
}

static void write_tensor_json(FILE *fp, const void *tensor, int dump_const_to_file) {
    if (!tensor) {
        fputs("null", fp);
        return;
    }
    void **tvt = *(void ***)tensor;
    const char *tname = ((const char *(*)(const void *))tvt[1])(tensor);
    uint32_t fmt = ((uint32_t (*)(const void *))tvt[14])(tensor);
    uint32_t tbits = ((uint32_t (*)(const void *))tvt[30])(tensor);

    OutputDef od;
    memset(&od, 0, sizeof(od));
    if (fn_tensor_gen_output_def) fn_tensor_gen_output_def(&od, tensor);

    int is_fake = (!tname || strcmp(tname, "FakeTensor") == 0 || od.dtype == 0xfe);
    size_t sbytes = 0;
    void *rdata = NULL;
    if (!is_fake) {
        sbytes = ((size_t (*)(const void *))tvt[23])(tensor);
        rdata = ((void *(*)(const void *))tvt[20])(tensor);
    }

    uint64_t const_off = 0, const_len = 0;
    const char *const_enc = "none";
    if (dump_const_to_file && !is_fake && g_consts_bin_fd >= 0 && rdata && sbytes > 0) {
        uint8_t *buf = NULL;
        size_t sz = extract_tensor_bytes(tensor, &od, tname, fmt, tbits, &buf, &const_enc);
        if (buf && sz > 0) {
            pthread_mutex_lock(&g_tap_mu);
            const_off = g_consts_bin_off;
            const_len = (uint64_t)sz;
            write_all_fd(g_consts_bin_fd, buf, sz);
            g_consts_bin_off += sz;
            if (g_consts_idx_fp) {
                fprintf(g_consts_idx_fp, "{\"tensor\":\"%p\",\"tname\":", tensor);
                fprint_json_str(g_consts_idx_fp, tname);
                fprintf(g_consts_idx_fp, ",\"off\":%llu,\"len\":%llu,\"enc\":",
                        (unsigned long long)const_off, (unsigned long long)const_len);
                fprint_json_str(g_consts_idx_fp, const_enc);
                fprintf(g_consts_idx_fp, "}\n");
            }
            pthread_mutex_unlock(&g_tap_mu);
            free(buf);
        }
    }

    fprintf(fp, "{\"ptr\":\"%p\",\"tname\":", tensor);
    fprint_json_str(fp, tname ? tname : "");
    fprintf(fp, ",\"fmt\":\"0x%08x\",\"tile_bits\":\"0x%x\",\"rank\":%u,\"dtype\":%u,\"dtype_name\":",
            fmt, tbits, od.rank, od.dtype);
    fprint_json_str(fp, dtype_name(od.dtype));
    fprintf(fp, ",\"dims\":[");
    for (uint32_t d = 0; d < od.rank && d < 8; d++) {
        fprintf(fp, "%s%llu", d ? "," : "", (unsigned long long)od.dims[d]);
    }
    fprintf(fp, "],\"zero_offset\":%d,\"scale\":%.9g,\"storage_bytes\":%zu,\"raw_data\":\"%p\"",
            od.zero_offset, (double)od.scale, sbytes, rdata);
    if (const_len > 0) {
        fprintf(fp, ",\"const_off\":%llu,\"const_len\":%llu,\"const_enc\":",
                (unsigned long long)const_off, (unsigned long long)const_len);
        fprint_json_str(fp, const_enc);
    }
    fputc('}', fp);
}

static int safe_extra_from_rl(const void *rl, size_t idx, uint64_t *out_id64, uint32_t *out_w1, uint32_t *out_w2) {
    if (out_id64) *out_id64 = 0;
    if (out_w1) *out_w1 = 0;
    if (out_w2) *out_w2 = 0;
    if (!rl) return 0;
    const uint8_t *info_beg = *(const uint8_t *const *)((const uint8_t *)rl + 0x30);
    const uint8_t *info_end = *(const uint8_t *const *)((const uint8_t *)rl + 0x38);
    if (!info_beg || !info_end || info_end <= info_beg) return 0;
    size_t count = (size_t)(info_end - info_beg) / 24;
    if (idx >= count) return 0;
    const uint8_t *ent = info_beg + idx * 24;
    if (out_id64) *out_id64 = *(const uint64_t *)(ent + 0);
    if (out_w1) *out_w1 = *(const uint32_t *)(ent + 8);
    if (out_w2) *out_w2 = *(const uint32_t *)(ent + 12);
    return 1;
}

static void dump_runlist(void *graph) {
    const uint8_t *rl_base = *(const uint8_t *const *)((const uint8_t *)graph + 0xc8);
    if (!rl_base || !g_tap_dir[0]) return;

    /* GraphData::RunLists holds up to 4 0x48-byte segments (init/main/bkgrnd/etc.) */
    size_t total_ops = 0;
    for (int seg = 0; seg < 4; seg++) {
        const uint8_t *rl = rl_base + seg * 0x48;
        void **op_beg = *(void ***)((const uint8_t *)rl + 0x18);
        void **op_end = *(void ***)((const uint8_t *)rl + 0x20);
        if (op_beg && op_end && op_end > op_beg) {
            total_ops += (size_t)(op_end - op_beg);
        }
    }
    if (total_ops == 0) return;

    free(g_op_entries);
    g_op_entries = (OpEntry *)malloc(total_ops * sizeof(OpEntry));
    g_n_ops = 0;

    char p[600];
    snprintf(p, sizeof(p), "%s/runlist.jsonl", g_tap_dir);
    FILE *fp = fopen(p, "w");
    if (!fp) return;
    setvbuf(fp, NULL, _IOFBF, 1 << 20);

    int global_idx = 0;
    for (int seg = 0; seg < 4; seg++) {
        const uint8_t *rl = rl_base + seg * 0x48;
        void **op_beg = *(void ***)((const uint8_t *)rl + 0x18);
        void **op_end = *(void ***)((const uint8_t *)rl + 0x20);
        if (!op_beg || !op_end || op_end <= op_beg) continue;
        size_t n = (size_t)(op_end - op_beg);
        for (size_t i = 0; i < n; i++) {
            const void *op = op_beg[i];
            g_op_entries[g_n_ops].op_ptr = (void *)op;
            g_op_entries[g_n_ops].idx = global_idx;
            g_n_ops++;

            uint64_t op_id64 = 0;
            uint32_t w1 = 0, w2 = 0;
            safe_extra_from_rl(rl, i, &op_id64, &w1, &w2);
            const char *tname = fn_op_true_name ? fn_op_true_name(op) : "";
            const char *tfunc = fn_op_true_func ? fn_op_true_func(op) : "";
            int is_const = fn_op_is_const ? (int)fn_op_is_const(op) : 0;

            void **vt = *(void ***)op;
            IoCounts io = ((IoCounts (*)(const void *))vt[10])(op);
            uint64_t flags = ((uint64_t (*)(const void *))vt[11])(op);
            const void *(*get_io)(const void *, size_t, int) = (const void *(*)(const void *, size_t, int))vt[15];

            fprintf(fp, "{\"idx\":%d,\"seg\":%d,\"seg_idx\":%zu,\"op_id\":\"0x%llx\",\"w1\":%u,\"w2\":%u,\"op_ptr\":\"%p\",\"true_name\":",
                    global_idx, seg, i, (unsigned long long)op_id64, w1, w2, op);
            fprint_json_str(fp, tname);
            fprintf(fp, ",\"true_func\":");
            fprint_json_str(fp, tfunc);
            fprintf(fp, ",\"is_const\":%d,\"flags\":\"0x%llx\",\"n_in\":%llu,\"n_out\":%llu,\"inputs\":[",
                    is_const, (unsigned long long)flags,
                    (unsigned long long)io.n_in, (unsigned long long)io.n_out);

            for (uint64_t j = 0; j < io.n_in; j++) {
                if (j) fputc(',', fp);
                const void *t = get_io(op, (size_t)j, 1);
                write_tensor_json(fp, t, g_dump_consts);
            }
            fprintf(fp, "],\"outputs\":[");
            for (uint64_t j = 0; j < io.n_out; j++) {
                if (j) fputc(',', fp);
                const void *t = get_io(op, (size_t)j, 0);
                write_tensor_json(fp, t, g_dump_consts && is_const);
            }
            fprintf(fp, "]}\n");
            global_idx++;
        }
    }
    qsort(g_op_entries, g_n_ops, sizeof(OpEntry), cmp_op_entry);

    pthread_mutex_lock(&g_exec_map_mu);
    free(g_exec_lookup);
    g_exec_lookup = NULL;
    g_exec_lookup_n = 0;
    if (g_exec_map_n > 0) {
        g_exec_lookup = (ExecMapEntry *)malloc(g_exec_map_n * sizeof(ExecMapEntry));
        if (g_exec_lookup) {
            for (size_t k = 0; k < g_exec_map_n; k++) {
                int oidx = find_op_idx(g_exec_map[k].op_ptr);
                if (oidx >= 0) {
                    g_exec_lookup[g_exec_lookup_n].exec_obj = g_exec_map[k].exec_obj;
                    g_exec_lookup[g_exec_lookup_n].op_ptr = g_exec_map[k].op_ptr;
                    g_exec_lookup[g_exec_lookup_n].idx = oidx;
                    g_exec_lookup_n++;
                }
            }
            if (g_exec_lookup_n > 0) {
                qsort(g_exec_lookup, g_exec_lookup_n, sizeof(ExecMapEntry), cmp_exec_entry);
            }
        }
    }
    pthread_mutex_unlock(&g_exec_map_mu);

    fclose(fp);
    if (g_consts_idx_fp) fflush(g_consts_idx_fp);
}

static void tap_after_exec(void *graph, void *obj, int exec_rc, uint32_t list_idx, int is_bkgrnd) {
    (void)graph;
    if (g_mode_deser_only || !g_taps_idx_fp || !g_taps_bin_fp) return;
    int idx = -1;
    const void *op = find_op_for_exec(obj, &idx);
    if (!op || idx < 0) return;

    int is_const = fn_op_is_const ? (int)fn_op_is_const(op) : 0;
    if (g_skip_const_taps && is_const) return;

    const char *tname = fn_op_true_name ? fn_op_true_name(op) : "";
    if (!tname || tname[0] == '@' || tname[0] == '*' || strcmp(tname, "N4hnnx14ChunkPreloadOpE") == 0) {
        return;
    }

    void **vt = *(void ***)op;
    IoCounts io = ((IoCounts (*)(const void *))vt[10])(op);
    if (io.n_out == 0) return;
    const void *(*get_io)(const void *, size_t, int) = (const void *(*)(const void *, size_t, int))vt[15];

    for (uint64_t o = 0; o < io.n_out; o++) {
        const void *t = get_io(op, (size_t)o, 0);
        if (!t) continue;
        void **tvt = *(void ***)t;
        const char *ttype = ((const char *(*)(const void *))tvt[1])(t);
        if (!ttype || strcmp(ttype, "FakeTensor") == 0) continue;

        OutputDef od;
        memset(&od, 0, sizeof(od));
        if (fn_tensor_gen_output_def) fn_tensor_gen_output_def(&od, t);
        if (od.dtype == 0xfe) continue;

        uint32_t fmt = ((uint32_t (*)(const void *))tvt[14])(t);
        uint32_t tbits = ((uint32_t (*)(const void *))tvt[30])(t);

        uint8_t *buf = NULL;
        const char *enc = "none";
        size_t nbytes = extract_tensor_bytes(t, &od, ttype, fmt, tbits, &buf, &enc);
        if (!buf || nbytes == 0) continue;

        pthread_mutex_lock(&g_tap_mu);
        if (g_max_tap_bytes == 0 || g_taps_bin_off + nbytes <= g_max_tap_bytes) {
            uint64_t seq = g_tap_seq++;
            uint64_t off = g_taps_bin_off;
            fwrite(buf, 1, nbytes, g_taps_bin_fp);
            g_taps_bin_off += nbytes;

            fprintf(g_taps_idx_fp,
                    "{\"seq\":%llu,\"idx\":%d,\"list_idx\":%u,\"bkgrnd\":%d,\"exec_rc\":%d,\"out_idx\":%llu,"
                    "\"op_ptr\":\"%p\",\"exec_obj\":\"%p\",\"true_name\":",
                    (unsigned long long)seq, idx, list_idx, is_bkgrnd, exec_rc,
                    (unsigned long long)o, op, obj);
            fprint_json_str(g_taps_idx_fp, tname);
            fprintf(g_taps_idx_fp, ",\"tensor\":\"%p\",\"ttype\":", t);
            fprint_json_str(g_taps_idx_fp, ttype);
            fprintf(g_taps_idx_fp, ",\"enc\":");
            fprint_json_str(g_taps_idx_fp, enc);
            fprintf(g_taps_idx_fp, ",\"dtype\":%u,\"dtype_name\":", od.dtype);
            fprint_json_str(g_taps_idx_fp, dtype_name(od.dtype));
            fprintf(g_taps_idx_fp, ",\"dims\":[");
            for (uint32_t d = 0; d < od.rank && d < 8; d++) {
                fprintf(g_taps_idx_fp, "%s%llu", d ? "," : "", (unsigned long long)od.dims[d]);
            }
            fprintf(g_taps_idx_fp, "],\"zero_offset\":%d,\"scale\":%.9g,\"bin_off\":%llu,\"bin_len\":%zu}\n",
                    od.zero_offset, (double)od.scale, (unsigned long long)off, nbytes);
        }
        pthread_mutex_unlock(&g_tap_mu);
        free(buf);
    }
}

/* --- Graph Execution PLT Hooks --- */

int _ZN5Graph10do_executeER12HexagonNNEnv(void *graph, void *env) {
    init_config();
    if (!real_do_execute) return -1;

    if (g_exec_count == 0) {
        g_cap_graph = graph;
        dump_runlist(graph);
        if (g_skip_exec) {
            g_exec_count++;
            if (g_extents_fp) fflush(g_extents_fp);
            return 0;
        }
        g_capturing = 1;
        int rc = real_do_execute(graph, env);
        g_capturing = 0;
        g_exec_count++;
        if (g_taps_bin_fp) fflush(g_taps_bin_fp);
        if (g_taps_idx_fp) fflush(g_taps_idx_fp);
        return rc;
    }

    g_exec_count++;
    if (g_skip_repeat_exec || g_skip_exec) return 0;
    return real_do_execute(graph, env);
}

int continue_execution_fast(void *graph, void *exec_info, ExecType *list, uint32_t num_ops,
                            uint32_t *idx_ptr, int *abort_flag, uint32_t *wait_cycles,
                            uint32_t thread_id, uint64_t oss) {
    if (!g_capturing || graph != g_cap_graph) {
        return real_continue_fast ? real_continue_fast(graph, exec_info, list, num_ops, idx_ptr, abort_flag, wait_cycles, thread_id, oss) : -1;
    }
    (void)exec_info;
    (void)wait_cycles;
    (void)thread_id;
    (void)oss;

    uint32_t idx = *idx_ptr;
    int rc = 0;
    while (idx < num_ops) {
        ExecType e = list[idx];
        uint32_t spins = 0;
        while (e.dep_count && *(volatile int *)e.dep_count != 0) {
            if (*(volatile int *)e.dep_count < 0) {
                *idx_ptr = idx;
                return 0x24;
            }
            if (abort_flag && *(volatile int *)abort_flag != 0) {
                *idx_ptr = idx;
                return 0;
            }
            if (++spins < 16) sched_yield();
            else usleep(200);
        }
        rc = e.fn(e.obj, graph, 1);
        tap_after_exec(graph, e.obj, rc, idx, 0);
        if (e.notify_ptr) {
            __atomic_store_n((volatile int *)e.notify_ptr, 0, __ATOMIC_RELEASE);
        }
        idx++;
        if (rc != 0) break;
        if (abort_flag && *(volatile int *)abort_flag != 0) break;
    }
    *idx_ptr = idx;
    return rc;
}

int continue_execution_bkgrnd_thread(void *graph, void *exec_info, ExecType *list, uint32_t num_ops,
                                     uint32_t *idx_ptr, int *abort_flag, uint32_t thread_id, uint64_t oss) {
    if (!g_capturing || graph != g_cap_graph) {
        return real_continue_bkgrnd ? real_continue_bkgrnd(graph, exec_info, list, num_ops, idx_ptr, abort_flag, thread_id, oss) : -1;
    }
    (void)exec_info;
    (void)thread_id;
    (void)oss;

    while (1) {
        uint32_t idx = __atomic_load_n(idx_ptr, __ATOMIC_ACQUIRE);
        if (idx >= num_ops) return 0;
        ExecType e = list[idx];
        uint32_t spins = 0;
        while (e.dep_count && *(volatile int *)e.dep_count != 0) {
            if (*(volatile int *)e.dep_count < 0) return 0x24;
            if (abort_flag && *(volatile int *)abort_flag != 0) return 0;
            if (++spins < 16) sched_yield();
            else usleep(200);
        }
        if (!__sync_bool_compare_and_swap(idx_ptr, idx, idx + 1)) continue;
        int rc = e.fn(e.obj, graph, 1);
        tap_after_exec(graph, e.obj, rc, idx, 1);
        if (e.notify_ptr) {
            if (__atomic_sub_fetch((volatile int *)e.notify_ptr, 1, __ATOMIC_SEQ_CST) < 0) {
                return 0x24;
            }
        }
        if (rc != 0) return rc;
        if (abort_flag && *(volatile int *)abort_flag != 0) return 0;
    }
}
