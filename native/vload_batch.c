// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Chromatic and contributors

// vload_batch: QAIRT HTP retrieve-context runner supporting single-run and --batch modes.
//
// Single-run CLI:
//   vload_batch <libdir> <ctx.bin> <graphname> <socmodel> <in_name:dims:type:sc:off:path>... -- <out_name:dims:type:sc:off:path>...
//
// Batch CLI (1 context load for N probes; manifest lines: in0,in1,...|out0,out1,...):
//   vload_batch <libdir> <ctx.bin> <graphname> <socmodel> --batch <manifest.txt> <in_names_csv> <out_names_csv>

#include <assert.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <malloc.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#if !defined(__cplusplus) && !defined(static_assert)
#define static_assert _Static_assert
#endif

#include "HTP/QnnHtpContext.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpGraph.h"
#include "HTP/QnnHtpPerfInfrastructure.h"
#include "QnnInterface.h"
#include "System/QnnSystemContext.h"
#include "System/QnnSystemInterface.h"

#define Qnn_Log_Level_t QnnLog_Level_t
#define CK(x) do { Qnn_ErrorHandle_t _e = (x); if (_e) { P("FAIL@%s err=%llu\n", #x, (unsigned long long)_e); return 9; } } while(0)

static int out_fd = 1;
#define P(...) dprintf(out_fd, __VA_ARGS__)

static void logger(const char *fmt, QnnLog_Level_t level, uint64_t ts, va_list args) {
    (void)ts;
    if (level <= QNN_LOG_LEVEL_WARN) {
        dprintf(out_fd, "QNN[%u] ", level);
        vdprintf(out_fd, fmt, args);
        dprintf(out_fd, "\n");
    }
}

typedef struct {
    char name[64];
    char path[512];
    Qnn_Tensor_t t;
    void *buf;
    size_t nbytes;
} TSPEC;

static int parse_spec_name_path(const char *s, TSPEC *t) {
    char tmp[1024];
    snprintf(tmp, sizeof(tmp), "%s", s);
    char *save = NULL;
    char *nm = strtok_r(tmp, ":", &save); if (!nm) return -1;
    snprintf(t->name, sizeof(t->name), "%s", nm);
    if (!strtok_r(NULL, ":", &save)) return -1;  // dims
    if (!strtok_r(NULL, ":", &save)) return -1;  // type
    if (!strtok_r(NULL, ":", &save)) return -1;  // scale
    if (!strtok_r(NULL, ":", &save)) return -1;  // offset
    char *pth = strtok_r(NULL, ":", &save);
    if (pth) snprintf(t->path, sizeof(t->path), "%s", pth);
    else t->path[0] = '\0';
    return 0;
}

static size_t nbytes_of(const Qnn_Tensor_t *t) {
    uint32_t rank = (t->version == QNN_TENSOR_VERSION_1) ? t->v1.rank : t->v2.rank;
    const uint32_t *dims = (t->version == QNN_TENSOR_VERSION_1) ? t->v1.dimensions : t->v2.dimensions;
    Qnn_DataType_t dt = (t->version == QNN_TENSOR_VERSION_1) ? t->v1.dataType : t->v2.dataType;
    size_t n = 1;
    for (uint32_t i = 0; i < rank; i++) n *= dims[i];
    size_t esz = 1;
    if (dt == QNN_DATATYPE_FLOAT_32 || dt == QNN_DATATYPE_UINT_32 || dt == QNN_DATATYPE_INT_32 || dt == QNN_DATATYPE_SFIXED_POINT_32) esz = 4;
    else if (dt == QNN_DATATYPE_FLOAT_16 || dt == QNN_DATATYPE_UFIXED_POINT_16 || dt == QNN_DATATYPE_SFIXED_POINT_16 || dt == QNN_DATATYPE_UINT_16 || dt == QNN_DATATYPE_INT_16) esz = 2;
    return n * esz;
}

static const Qnn_Tensor_t *find_tensor(const Qnn_Tensor_t *arr, uint32_t n, const char *name) {
    if (!arr) return NULL;
    for (uint32_t i = 0; i < n; i++) {
        const char *tn = (arr[i].version == QNN_TENSOR_VERSION_1) ? arr[i].v1.name : arr[i].v2.name;
        if (tn && strcmp(tn, name) == 0) return &arr[i];
    }
    return NULL;
}

static void dump_tensor_md(const char *io, const Qnn_Tensor_t *t) {
    if (!t) { P("MD %s null\n", io); return; }
    if (t->version == QNN_TENSOR_VERSION_1) {
        P("MD %s [V1] name=%s type=%d dtype=0x%x rank=%u dims=[", io, t->v1.name ? t->v1.name : "?", (int)t->v1.type, (unsigned)t->v1.dataType, t->v1.rank);
        for (uint32_t d = 0; d < t->v1.rank; d++) P("%u%s", t->v1.dimensions[d], d + 1 < t->v1.rank ? "," : "");
        P("]");
        if (t->v1.quantizeParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET)
            P(" scale=%.10g offset=%d", t->v1.quantizeParams.scaleOffsetEncoding.scale, t->v1.quantizeParams.scaleOffsetEncoding.offset);
        P("\n");
        return;
    }
    P("MD %s [V2] name=%s type=%d dtype=0x%x rank=%u dims=[", io, t->v2.name ? t->v2.name : "?", (int)t->v2.type, (unsigned)t->v2.dataType, t->v2.rank);
    for (uint32_t d = 0; d < t->v2.rank; d++) P("%u%s", t->v2.dimensions[d], d + 1 < t->v2.rank ? "," : "");
    P("]");
    if (t->v2.quantizeParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET)
        P(" scale=%.10g offset=%d", t->v2.quantizeParams.scaleOffsetEncoding.scale, t->v2.quantizeParams.scaleOffsetEncoding.offset);
    P("\n");
}

static int init_io_tensor(TSPEC *sp, const Qnn_Tensor_t *mt, Qnn_TensorType_t ttype) {
    memset(&sp->t, 0, sizeof(sp->t));
    if (mt->version == QNN_TENSOR_VERSION_1) {
        sp->t.version = QNN_TENSOR_VERSION_1;
        sp->t.v1 = mt->v1;
        sp->t.v1.type = ttype;
        sp->nbytes = nbytes_of(&sp->t);
        sp->buf = memalign(64, sp->nbytes ? sp->nbytes : 64);
        memset(sp->buf, 0, sp->nbytes);
        sp->t.v1.memType = QNN_TENSORMEMTYPE_RAW;
        sp->t.v1.clientBuf.data = sp->buf;
        sp->t.v1.clientBuf.dataSize = (uint32_t)sp->nbytes;
    } else {
        sp->t.version = QNN_TENSOR_VERSION_2;
        sp->t.v2 = mt->v2;
        sp->t.v2.type = ttype;
        sp->nbytes = nbytes_of(&sp->t);
        sp->buf = memalign(64, sp->nbytes ? sp->nbytes : 64);
        memset(sp->buf, 0, sp->nbytes);
        sp->t.v2.memType = QNN_TENSORMEMTYPE_RAW;
        sp->t.v2.clientBuf.data = sp->buf;
        sp->t.v2.clientBuf.dataSize = (uint32_t)sp->nbytes;
    }
    return 0;
}

static int run_one_item(const QNN_INTERFACE_VER_TYPE *h, Qnn_GraphHandle_t graph,
                        TSPEC *ins, int nin, TSPEC *outs, int nout, int item_idx) {
    static Qnn_Tensor_t itens[64], otens[64];
    for (int i = 0; i < nin; i++) {
        memset(ins[i].buf, 0, ins[i].nbytes);
        int fd = open(ins[i].path, O_RDONLY);
        if (fd < 0) { P("ERR_OPEN_IN %s\n", ins[i].path); return 5; }
        size_t off = 0;
        while (off < ins[i].nbytes) {
            ssize_t r = read(fd, (char *)ins[i].buf + off, ins[i].nbytes - off);
            if (r <= 0) break;
            off += (size_t)r;
        }
        close(fd);
        itens[i] = ins[i].t;
    }
    for (int i = 0; i < nout; i++) {
        memset(outs[i].buf, 0, outs[i].nbytes);
        otens[i] = outs[i].t;
    }
    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    Qnn_ErrorHandle_t ee = h->graphExecute(graph, itens, (uint32_t)nin, otens, (uint32_t)nout, NULL, NULL);
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    if (ee) { P("FAIL@graphExecute item=%d err=%llu\n", item_idx, (unsigned long long)ee); return 9; }
    long long ns = (ts1.tv_sec - ts0.tv_sec) * 1000000000LL + (ts1.tv_nsec - ts0.tv_nsec);
    for (int i = 0; i < nout; i++) {
        int fd = open(outs[i].path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) { P("ERR_OPEN_OUT %s\n", outs[i].path); return 6; }
        size_t off = 0;
        while (off < outs[i].nbytes) {
            ssize_t w = write(fd, (const char *)outs[i].buf + off, outs[i].nbytes - off);
            if (w <= 0) break;
            off += (size_t)w;
        }
        close(fd);
    }
    P("ITEM_OK idx=%d ns=%lld\n", item_idx, ns);
    return 0;
}

int main(int argc, char **argv) {
    const char *logpath = getenv("QNNREV_LOG");
    if (!logpath || !logpath[0]) logpath = getenv("VLOAD_LOG");
    out_fd = (logpath && logpath[0]) ? open(logpath, O_WRONLY | O_CREAT | O_TRUNC, 0644) : 1;
    if (out_fd < 0) out_fd = 1;
    if (argc < 7) {
        P("usage: %s libdir ctx.bin graphname socmodel [--batch file in_csv out_csv | in_specs -- out_specs]\n", argv[0]);
        return 2;
    }
    const char *libdir = argv[1];
    const char *ctxpath = argv[2];
    const char *gname = argv[3];
    int socmodel = atoi(argv[4]);

    bool batch_mode = (strcmp(argv[5], "--batch") == 0);
    const char *batch_file = NULL;
    static TSPEC ins[64], outs[64];
    int nin = 0, nout = 0;

    if (batch_mode) {
        if (argc != 9) { P("BAD_BATCH_ARGS\n"); return 2; }
        batch_file = argv[6];
        char in_csv[1024], out_csv[1024];
        snprintf(in_csv, sizeof(in_csv), "%s", argv[7]);
        snprintf(out_csv, sizeof(out_csv), "%s", argv[8]);
        char *s = NULL;
        for (char *p = strtok_r(in_csv, ",", &s); p && nin < 64; p = strtok_r(NULL, ",", &s))
            snprintf(ins[nin++].name, sizeof(ins[0].name), "%s", p);
        s = NULL;
        for (char *p = strtok_r(out_csv, ",", &s); p && nout < 64; p = strtok_r(NULL, ",", &s))
            snprintf(outs[nout++].name, sizeof(outs[0].name), "%s", p);
    } else {
        int sep = -1;
        for (int i = 5; i < argc; i++) if (strcmp(argv[i], "--") == 0) { sep = i; break; }
        if (sep < 0 || sep == 5 || sep == argc - 1 || sep - 5 > 64 || argc - sep - 1 > 64) { P("BAD_ARGS\n"); return 2; }
        nin = sep - 5;
        nout = argc - sep - 1;
        for (int i = 0; i < nin; i++) if (parse_spec_name_path(argv[5 + i], &ins[i])) return 2;
        for (int i = 0; i < nout; i++) if (parse_spec_name_path(argv[sep + 1 + i], &outs[i])) return 2;
    }

    char pathbuf[512];
    snprintf(pathbuf, sizeof(pathbuf), "%s/libQnnHtp.so", libdir);
    void *lib = dlopen(pathbuf, RTLD_NOW | RTLD_LOCAL);
    if (!lib) { P("DLOPEN_ERROR=%s\n", dlerror()); return 1; }
    Qnn_ErrorHandle_t (*getprov)(const QnnInterface_t ***, uint32_t *) =
        (Qnn_ErrorHandle_t (*)(const QnnInterface_t ***, uint32_t *))dlsym(lib, "QnnInterface_getProviders");
    const QnnInterface_t **hp = NULL; uint32_t hn = 0;
    CK(getprov(&hp, &hn));
    const QNN_INTERFACE_VER_TYPE *h = &hp[0]->QNN_INTERFACE_VER_NAME;

    Qnn_LogHandle_t log = NULL;
    if (h->logCreate) h->logCreate(logger, QNN_LOG_LEVEL_WARN, &log);
    Qnn_BackendHandle_t backend = NULL;
    CK(h->backendCreate(log, NULL, &backend));

    Qnn_DeviceHandle_t device = NULL;
    if (socmodel > 0) {
        static QnnHtpDevice_CustomConfig_t hcfg[2];
        memset(hcfg, 0, sizeof(hcfg));
        hcfg[0].option = QNN_HTP_DEVICE_CONFIG_OPTION_SOC;
        hcfg[0].socModel = (uint32_t)socmodel;
        hcfg[1].option = QNN_HTP_DEVICE_CONFIG_OPTION_ARCH;
        hcfg[1].arch.deviceId = 0;
        hcfg[1].arch.arch = QNN_HTP_DEVICE_ARCH_V73;
        static QnnDevice_Config_t dcfg[2];
        memset(dcfg, 0, sizeof(dcfg));
        dcfg[0].option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
        dcfg[0].customConfig = &hcfg[0];
        dcfg[1].option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
        dcfg[1].customConfig = &hcfg[1];
        const QnnDevice_Config_t *dcfgp[3] = { &dcfg[0], &dcfg[1], NULL };
        Qnn_ErrorHandle_t de = h->deviceCreate(log, dcfgp, &device);
        if (de) {
            P("DEVCFG_FALLBACK err=%llu\n", (unsigned long long)de);
            const QnnDevice_Config_t *darchp[2] = { &dcfg[1], NULL };
            de = h->deviceCreate(log, darchp, &device);
            if (de) CK(h->deviceCreate(log, NULL, &device));
        }
    } else {
        CK(h->deviceCreate(log, NULL, &device));
    }

    QnnHtpDevice_PerfInfrastructure_t perfInfra = QNN_HTP_DEVICE_PERF_INFRASTRUCTURE_INIT;
    uint32_t powerConfigId = 0;
    bool hasPowerConfigId = false;
    if (h->deviceGetInfrastructure) {
        QnnDevice_Infrastructure_t infra = NULL;
        Qnn_ErrorHandle_t ie = h->deviceGetInfrastructure(&infra);
        if (!ie && infra) {
            const QnnHtpDevice_Infrastructure_t *htpInfra = (const QnnHtpDevice_Infrastructure_t *)infra;
            if (htpInfra->infraType == QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) {
                perfInfra = htpInfra->perfInfra;
                if (perfInfra.createPowerConfigId && perfInfra.setPowerConfig) {
                    if (!perfInfra.createPowerConfigId(0, 0, &powerConfigId)) {
                        hasPowerConfigId = true;
                        QnnHtpPerfInfrastructure_PowerConfig_t pcfg[3];
                        memset(pcfg, 0, sizeof(pcfg));
                        pcfg[0].option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
                        pcfg[0].dcvsV3Config.contextId = powerConfigId;
                        pcfg[0].dcvsV3Config.setDcvsEnable = 1;
                        pcfg[0].dcvsV3Config.dcvsEnable = 0;
                        pcfg[0].dcvsV3Config.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
                        pcfg[0].dcvsV3Config.setSleepLatency = 1;
                        pcfg[0].dcvsV3Config.sleepLatency = 40;
                        pcfg[0].dcvsV3Config.setSleepDisable = 1;
                        pcfg[0].dcvsV3Config.sleepDisable = 1;
                        pcfg[0].dcvsV3Config.setBusParams = 1;
                        pcfg[0].dcvsV3Config.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
                        pcfg[0].dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
                        pcfg[0].dcvsV3Config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
                        pcfg[0].dcvsV3Config.setCoreParams = 1;
                        pcfg[0].dcvsV3Config.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
                        pcfg[0].dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
                        pcfg[0].dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
                        pcfg[1].option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
                        pcfg[1].rpcControlLatencyConfig = 100;
                        pcfg[2].option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
                        pcfg[2].rpcPollingTimeConfig = 9999;
                        const QnnHtpPerfInfrastructure_PowerConfig_t *cfg_ptrs[4] = {
                            &pcfg[0], &pcfg[1], &pcfg[2], NULL
                        };
                        Qnn_ErrorHandle_t pe = perfInfra.setPowerConfig(powerConfigId, cfg_ptrs);
                        P("PERF_BURST id=%u err=%llu\n", powerConfigId, (unsigned long long)pe);
                    }
                }
            }
        }
    }

    int fd = open(ctxpath, O_RDONLY);
    if (fd < 0) { P("NO_CTX_FILE %s\n", ctxpath); return 3; }
    off_t fsz = lseek(fd, 0, SEEK_END); lseek(fd, 0, SEEK_SET);
    void *bin = malloc((size_t)fsz);
    size_t got = 0;
    while (got < (size_t)fsz) {
        ssize_t r = read(fd, (char *)bin + got, (size_t)fsz - got);
        if (r <= 0) break;
        got += (size_t)r;
    }
    close(fd);

    Qnn_ContextHandle_t context = NULL;
    static QnnHtpContext_CustomConfig_t hctx_cfg;
    memset(&hctx_cfg, 0, sizeof(hctx_cfg));
    hctx_cfg.option = QNN_HTP_CONTEXT_CONFIG_OPTION_PARALLEL_WEIGHT_LOADING;
    hctx_cfg.parallelWeightLoading.mode = QNN_HTP_PARALLEL_WEIGHT_LOADING_MODE_ON;
    hctx_cfg.parallelWeightLoading.thresholdInMbOverride = 0;
    static QnnContext_Config_t ctx_cfg;
    memset(&ctx_cfg, 0, sizeof(ctx_cfg));
    ctx_cfg.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
    ctx_cfg.customConfig = &hctx_cfg;
    const QnnContext_Config_t *ctx_cfgs[2] = { &ctx_cfg, NULL };
    Qnn_ErrorHandle_t ce = h->contextCreateFromBinary(
        backend, device, ctx_cfgs, bin, (Qnn_ContextBinarySize_t)fsz, &context, NULL);
    if (ce) {
        P("CTXCFG_FALLBACK err=%llu\n", (unsigned long long)ce);
        CK(h->contextCreateFromBinary(backend, device, NULL, bin, (Qnn_ContextBinarySize_t)fsz, &context, NULL));
    }
    Qnn_GraphHandle_t graph = NULL;
    CK(h->graphRetrieve(context, gname, &graph));

    snprintf(pathbuf, sizeof(pathbuf), "%s/libQnnSystem.so", libdir);
    void *slib = dlopen(pathbuf, RTLD_NOW | RTLD_LOCAL);
    if (!slib) { P("SYS_DLOPEN_ERROR=%s\n", dlerror()); return 1; }
    Qnn_ErrorHandle_t (*sysgetprov)(const QnnSystemInterface_t ***, uint32_t *) =
        (Qnn_ErrorHandle_t(*)(const QnnSystemInterface_t ***, uint32_t *))dlsym(slib, "QnnSystemInterface_getProviders");
    const QnnSystemInterface_t **sp = NULL; uint32_t sn = 0;
    CK(sysgetprov(&sp, &sn));
    const QNN_SYSTEM_INTERFACE_VER_TYPE *sh = &sp[0]->QNN_SYSTEM_INTERFACE_VER_NAME;
    QnnSystemContext_Handle_t sysctx = NULL;
    CK(sh->systemContextCreate(&sysctx));
    const QnnSystemContext_BinaryInfo_t *binfo = NULL;
    Qnn_ContextBinarySize_t binfoSize = 0;
    CK(sh->systemContextGetBinaryInfo(sysctx, bin, (uint64_t)fsz, &binfo, &binfoSize));
    uint32_t nbinfo_graphs = 0; const QnnSystemContext_GraphInfo_t *binfoGraphs = NULL;
    if (binfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
        nbinfo_graphs = binfo->contextBinaryInfoV1.numGraphs; binfoGraphs = binfo->contextBinaryInfoV1.graphs;
    } else if (binfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
        nbinfo_graphs = binfo->contextBinaryInfoV2.numGraphs; binfoGraphs = binfo->contextBinaryInfoV2.graphs;
    } else if (binfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
        nbinfo_graphs = binfo->contextBinaryInfoV3.numGraphs; binfoGraphs = binfo->contextBinaryInfoV3.graphs;
    }

    const Qnn_Tensor_t *gin_ptr = NULL, *gout_ptr = NULL;
    uint32_t ngin_ptr = 0, ngout_ptr = 0;
    for (uint32_t g = 0; g < nbinfo_graphs; g++) {
        const QnnSystemContext_GraphInfo_t *gg = &binfoGraphs[g];
        const char *gn = NULL;
        const Qnn_Tensor_t *gin = NULL, *gout = NULL;
        uint32_t ngin = 0, ngout = 0;
        if (gg->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
            gn = gg->graphInfoV3.graphName; gin = gg->graphInfoV3.graphInputs; ngin = gg->graphInfoV3.numGraphInputs;
            gout = gg->graphInfoV3.graphOutputs; ngout = gg->graphInfoV3.numGraphOutputs;
        } else if (gg->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {
            gn = gg->graphInfoV2.graphName; gin = gg->graphInfoV2.graphInputs; ngin = gg->graphInfoV2.numGraphInputs;
            gout = gg->graphInfoV2.graphOutputs; ngout = gg->graphInfoV2.numGraphOutputs;
        } else if (gg->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
            gn = gg->graphInfoV1.graphName; gin = gg->graphInfoV1.graphInputs; ngin = gg->graphInfoV1.numGraphInputs;
            gout = gg->graphInfoV1.graphOutputs; ngout = gg->graphInfoV1.numGraphOutputs;
        }
        if (gn && strcmp(gn, gname) == 0) {
            gin_ptr = gin; ngin_ptr = ngin; gout_ptr = gout; ngout_ptr = ngout;
            break;
        }
    }
    if (!gin_ptr) { P("GRAPH_NOT_FOUND %s\n", gname); return 4; }

    for (int i = 0; i < nin; i++) {
        const Qnn_Tensor_t *mt = find_tensor(gin_ptr, ngin_ptr, ins[i].name);
        if (!mt) { P("NO_META_IN %s\n", ins[i].name); return 4; }
        dump_tensor_md("IN", mt);
        init_io_tensor(&ins[i], mt, QNN_TENSOR_TYPE_APP_WRITE);
    }
    for (int i = 0; i < nout; i++) {
        const Qnn_Tensor_t *mt = find_tensor(gout_ptr, ngout_ptr, outs[i].name);
        if (!mt) { P("NO_META_OUT %s\n", outs[i].name); return 4; }
        dump_tensor_md("OUT", mt);
        init_io_tensor(&outs[i], mt, QNN_TENSOR_TYPE_APP_READ);
    }

    int rc = 0;
    if (!batch_mode) {
        rc = run_one_item(h, graph, ins, nin, outs, nout, 0);
    } else {
        FILE *bf = fopen(batch_file, "r");
        if (!bf) { P("ERR_BATCH_FILE %s\n", batch_file); return 7; }
        char line[4096];
        int idx = 0;
        while (fgets(line, sizeof(line), bf)) {
            size_t L = strlen(line);
            while (L > 0 && (line[L-1] == '\n' || line[L-1] == '\r')) line[--L] = '\0';
            if (L == 0 || line[0] == '#') continue;
            char *bar = strchr(line, '|');
            if (!bar) { P("BAD_BATCH_LINE %d\n", idx); rc = 7; break; }
            *bar = '\0';
            char *in_str = line;
            char *out_str = bar + 1;
            char *s = NULL;
            int ii = 0;
            for (char *p = strtok_r(in_str, ",", &s); p && ii < nin; p = strtok_r(NULL, ",", &s))
                snprintf(ins[ii++].path, sizeof(ins[0].path), "%s", p);
            s = NULL;
            int oi = 0;
            for (char *p = strtok_r(out_str, ",", &s); p && oi < nout; p = strtok_r(NULL, ",", &s))
                snprintf(outs[oi++].path, sizeof(outs[0].path), "%s", p);
            if (ii != nin || oi != nout) { P("BAD_IO_COUNT line=%d\n", idx); rc = 7; break; }
            rc = run_one_item(h, graph, ins, nin, outs, nout, idx);
            if (rc) break;
            idx++;
        }
        fclose(bf);
    }

    if (sh->systemContextFree) sh->systemContextFree(sysctx);
    if (h->contextFree) h->contextFree(context, NULL);
    if (hasPowerConfigId && perfInfra.destroyPowerConfigId) perfInfra.destroyPowerConfigId(powerConfigId);
    if (h->deviceFree) h->deviceFree(device);
    if (h->backendFree) h->backendFree(backend);
    if (h->logFree && log) h->logFree(log);
    for (int i = 0; i < nin; i++) free(ins[i].buf);
    for (int i = 0; i < nout; i++) free(outs[i].buf);
    free(bin);
    P("VLOAD_BATCH_DONE rc=%d\n", rc);
    return rc;
}
