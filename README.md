# MobileKV


核心库接口简洁稳定

上层 runtime 容易接

兼容你现在的 CPU-only mobile KV cache

后续扩展到 INT8 / ring / GPU

便于适配 llama.cpp / MLC / ExecuTorch / 自研 runtime

我建议把接口分成三层：

Core API：真正的底层 KV cache 能力

Runtime-Friendly API：方便推理框架接入

Compatibility API：给不同 runtime 做适配

一、接口设计原则

这套接口要满足 5 个原则：

1. 核心对象只有一个

就是 KVCache

2. 写路径和读路径分开

写：append / seal prefix / compact / reset

读：get token ptr / get read view / metadata

3. 不把内部布局泄漏太多

现在是 [layer][token][hidden]，以后可能变 ring 或 quant。
所以上层不要直接依赖内部 offset 公式。

4. 输入输出都支持“每层连续 hidden”

因为这是 CPU decode 最自然的格式。

5. 从一开始就预留后向兼容字段

比如：

layout type

dtype

backend type

flags

二、推荐目录结构

我建议头文件这样分：

include/mobilekv/
  kv_types.h
  kv_cache.h
  kv_compat.h

如果你现在先做一个头，也可以先合并，逻辑上还是按这三层设计。

三、第一层：Core API

这是最底层，也是你项目最重要的一层。

3.1 基础类型
enum KVStatus {
  KV_OK = 0,
  KV_ERR_NULL,
  KV_ERR_BAD_ARG,
  KV_ERR_BAD_STATE,
  KV_ERR_NO_SPACE,
  KV_ERR_UNSUPPORTED,
};

enum KVDataType {
  KV_DTYPE_FP16 = 0,
  KV_DTYPE_INT8 = 1,
};

enum KVLayoutType {
  KV_LAYOUT_LAYER_TOKEN_HIDDEN = 0,   // current CPU layout
  KV_LAYOUT_LAYER_BLOCK_HIDDEN = 1,   // future
  KV_LAYOUT_RING = 2,                 // future
};

enum KVBackendType {
  KV_BACKEND_CPU = 0,
  KV_BACKEND_GPU = 1,   // reserved
};
3.2 配置结构
struct KVShape {
  int layers;
  int heads;
  int head_dim;
  int hidden;
  int max_seq;
};

struct KVPolicy {
  int prefix_tokens;
  int recent_keep;
};

struct KVQuantConfig {
  int enabled;
  int group_size;
  int scale_bytes;
};

struct KVConfig {
  KVShape shape;
  KVPolicy policy;
  KVDataType dtype;
  KVLayoutType layout;
  KVBackendType backend;
  KVQuantConfig quant;
};

这样以后初始化只传一个配置对象。

3.3 运行时状态
struct KVState {
  int cursor;
  int valid_tokens;
  int base_token;
  int prefix_frozen;
  int prefix_tokens;
};
3.4 主对象
struct KVCache;

对外可以前置声明；如果要做纯 C API，就暴露句柄。

3.5 核心生命周期接口

这是最基本的一组。

size_t KVRequiredBytes(const KVConfig* config);

KVStatus KVInitPreallocated(
    KVCache* kv,
    const KVConfig* config,
    void* arena_base,
    size_t arena_bytes);

void KVReset(KVCache* kv);

KVStatus KVSealPrefix(KVCache* kv, int prefix_tokens);

这里比前一版更好，因为只传 KVConfig。

3.6 核心写接口
当前最重要的 append
KVStatus KVAppend(
    KVCache* kv,
    const void* new_k_layers,
    const void* new_v_layers);

但这个接口还不够，因为不同上游输出的 stride 可能不同。
所以建议再提供一个更通用版本：

struct KVAppendView {
  const void* k;
  const void* v;
  size_t layer_stride_bytes;   // bytes between layer i and layer i+1
};

KVStatus KVAppendViewWrite(
    KVCache* kv,
    const KVAppendView* view);
为什么推荐保留两个接口

KVAppend()：给简单场景

KVAppendViewWrite()：给真正 runtime 接入

这样兼容性最好。

3.7 compact 接口
KVStatus KVCompact(KVCache* kv);

通常 runtime 不一定主动调，内部 append 也可以自动触发。
但保留它可以方便调试和 benchmark。

3.8 元数据读取接口
int KVValidTokens(const KVCache* kv);
int KVCursor(const KVCache* kv);
int KVBaseToken(const KVCache* kv);
int KVPrefixTokens(const KVCache* kv);
int KVRecentTokens(const KVCache* kv);
KVDataType KVDType(const KVCache* kv);
KVLayoutType KVLayout(const KVCache* kv);
3.9 读接口：不要只暴露 token ptr，最好暴露 ReadView

这是为了以后兼容 ring / block layout。

先定义 view
struct KVSegment {
  int start_token;
  int token_count;
};

struct KVReadView {
  int segment_count;       // current compact layout => always 1
  KVSegment segments[2];   // reserve for future ring layout
};
接口
KVStatus KVGetReadView(
    const KVCache* kv,
    KVReadView* out_view);

对当前 compact 版本，返回：

segment_count = 1
segments[0] = {0, valid_tokens}

未来 ring 版就能自然扩展成两段。

这比只给 valid_tokens 更有前瞻性。

3.10 读接口：按 token 取指针

保留，给 CPU attention kernel 用。

const void* KVKToken(const KVCache* kv, int layer, int token);
const void* KVVToken(const KVCache* kv, int layer, int token);

const uint16_t* KVKTokenFP16(const KVCache* kv, int layer, int token);
const uint16_t* KVVTokenFP16(const KVCache* kv, int layer, int token);

const int8_t* KVKTokenINT8(const KVCache* kv, int layer, int token);   // future
const int8_t* KVVTokenINT8(const KVCache* kv, int layer, int token);   // future
const uint16_t* KVKScales(const KVCache* kv, int layer, int token);    // future
const uint16_t* KVVScales(const KVCache* kv, int layer, int token);    // future
四、第二层：Runtime-Friendly API

这一层是给“推理 runtime”直接接入的，不要求知道太多底层细节。

4.1 每层 token 写入接口

有些 runtime 每层单独计算 K/V，不一定一次把所有层都给你。
所以要支持 per-layer append target。

先申请写入槽位
KVStatus KVReserveTokenSlot(KVCache* kv, int* out_token);

返回本次 token 的物理 slot。

对某一层写 K/V
KVStatus KVWriteLayerToken(
    KVCache* kv,
    int layer,
    int token,
    const void* k_data,
    const void* v_data,
    size_t bytes);
完成 token
KVStatus KVCommitToken(KVCache* kv, int token);

这组接口的意义非常大。因为很多实际 runtime 流程是：

for each token:
  reserve slot
  for each layer:
    compute K/V
    write layer KV
  commit

这比一次性传 [layers][hidden] 更通用。

4.2 当前 token 快速写接口

如果 runtime 已经是“所有层 K/V 都准备好了”，继续用简单接口：

KVStatus KVAppend(KVCache* kv, const void* new_k_layers, const void* new_v_layers);

所以两种模式都支持：

整体 append

分层 reserve/write/commit

4.3 attention 读取接口

为了让上层 kernel 不耦合太深，建议提供：

struct KVLayerReadSpan {
  const void* k_base;
  const void* v_base;
  int token_count;
  size_t token_stride_bytes;
};

KVStatus KVGetLayerReadSpan(
    const KVCache* kv,
    int layer,
    KVLayerReadSpan* out_span);

对于当前 layout：

k_base = K[layer][0]

v_base = V[layer][0]

token_count = valid_tokens

token_stride_bytes = hidden * elem_size

这样 CPU attention kernel 可以非常简单地扫：

for t in [0, token_count):
  k_ptr = k_base + t * token_stride_bytes
  v_ptr = v_base + t * token_stride_bytes

这比每次调用 KVKTokenFP16() 更利于性能。

五、第三层：Compatibility API

这是给不同框架做适配的接口层。
重点不是“功能更多”，而是“让对接更轻”。

5.1 llama.cpp 风格兼容接口

llama.cpp 风格常见需求是：

获取 seq 长度

获取某层 KV 连续内存

做 context shift

做 prefix 保留

所以可以设计一组适配接口：

struct KVCompatLlamaSpan {
  const void* k;
  const void* v;
  int n_tokens;
  int hidden;
  size_t stride_bytes;
};

KVStatus KVCompatGetLlamaSpan(
    const KVCache* kv,
    int layer,
    KVCompatLlamaSpan* out_span);

这样上层只要把它映射到自己的 K/V 读取逻辑就行。

5.2 通用 C 风格接口

如果你想以后做跨语言绑定，建议再提供一层纯 C API。

例如：

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mobilekv_handle mobilekv_handle_t;

size_t mobilekv_required_bytes(const KVConfig* config);

mobilekv_handle_t* mobilekv_init_preallocated(
    void* handle_mem,
    size_t handle_mem_bytes,
    const KVConfig* config,
    void* arena_base,
    size_t arena_bytes);

int mobilekv_append(
    mobilekv_handle_t* kv,
    const void* new_k_layers,
    const void* new_v_layers);

int mobilekv_valid_tokens(const mobilekv_handle_t* kv);

const void* mobilekv_k_token(
    const mobilekv_handle_t* kv,
    int layer,
    int token);

#ifdef __cplusplus
}
#endif

但如果你当前只做 C++，这个先不用实现，只要名字预留好。

六、我最推荐的接口组合

如果只选一套最平衡的，我建议保留下面这些。

6.1 最小稳定 Core API
size_t KVRequiredBytes(const KVConfig* config);

KVStatus KVInitPreallocated(
    KVCache* kv,
    const KVConfig* config,
    void* arena_base,
    size_t arena_bytes);

void KVReset(KVCache* kv);

KVStatus KVSealPrefix(KVCache* kv, int prefix_tokens);

KVStatus KVAppend(
    KVCache* kv,
    const void* new_k_layers,
    const void* new_v_layers);

KVStatus KVCompact(KVCache* kv);

int KVValidTokens(const KVCache* kv);
int KVBaseToken(const KVCache* kv);

KVStatus KVGetReadView(const KVCache* kv, KVReadView* out_view);

KVStatus KVGetLayerReadSpan(
    const KVCache* kv,
    int layer,
    KVLayerReadSpan* out_span);
6.2 最推荐的 Runtime 扩展接口
KVStatus KVReserveTokenSlot(KVCache* kv, int* out_token);

KVStatus KVWriteLayerToken(
    KVCache* kv,
    int layer,
    int token,
    const void* k_data,
    const void* v_data,
    size_t bytes);

KVStatus KVCommitToken(KVCache* kv, int token);

这三项对接 runtime 时非常值。

七、建议的状态机

为了让接口不容易被误用，建议文档里明确状态机。

INIT
  -> APPEND
  -> SEAL_PREFIX (optional)
  -> APPEND
  -> COMPACT (internal or explicit)
  -> APPEND
  -> RESET

如果支持 reserve/write/commit，则：

INIT
  -> RESERVE_SLOT
  -> WRITE_LAYER_TOKEN x L
  -> COMMIT_TOKEN
  -> RESERVE_SLOT
  -> ...
八、一个很实用的兼容建议：不要把 prefix 固定死在 init

虽然 KVPolicy 里有 prefix_tokens，但真正运行时更合理的是：

init 时只配 recent_keep

prefix 通过 KVSealPrefix() 在 prefill 完系统提示后确定

所以建议策略改成：

struct KVPolicy {
  int max_prefix_tokens;   // budget
  int recent_keep;
};

运行时：

KVSealPrefix(kv, actual_prefix_tokens);

这样更贴近真实推理流程。

九、推荐的最终头文件形态

如果你要我给结论，我推荐最终 kv_cache.h 按这个结构组织：

// types
enum ...
struct KVShape ...
struct KVPolicy ...
struct KVConfig ...
struct KVState ...
struct KVReadView ...
struct KVLayerReadSpan ...
struct KVAppendView ...

// lifecycle
size_t KVRequiredBytes(...)
KVStatus KVInitPreallocated(...)
void KVReset(...)
KVStatus KVSealPrefix(...)

// write
KVStatus KVAppend(...)
KVStatus KVAppendViewWrite(...)
KVStatus KVReserveTokenSlot(...)
KVStatus KVWriteLayerToken(...)
KVStatus KVCommitToken(...)
KVStatus KVCompact(...)

// read
int KVValidTokens(...)
int KVBaseToken(...)
KVStatus KVGetReadView(...)
KVStatus KVGetLayerReadSpan(...)
const void* KVKToken(...)
const void* KVVToken(...)

// debug / introspection
KVDataType KVDType(...)
KVLayoutType KVLayout(...)
int KVPrefixTokens(...)
int KVRecentTokens(...)

这套接口足够干净，也足够扩展。

十、我的最终建议

如果你现在只做 CPU-only MVP，不要一口气全实现。
最合理的是：

先实现这些

KVRequiredBytes

KVInitPreallocated

KVReset

KVSealPrefix

KVAppend

KVCompact

KVGetReadView

KVGetLayerReadSpan

KVValidTokens

KVBaseToken

预留但先不实现这些

KVAppendViewWrite

KVReserveTokenSlot

KVWriteLayerToken

KVCommitToken

INT8 相关读取接口

GPU backend 相关接口

这样兼容性已经够强，而且不会把项目做复杂。