#include "backend/metal/metal_backend.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <set>

#include "backend/metal/metal_kernels.h"

namespace dooms {

namespace {

constexpr size_t kReducePipeline = 8;  // index of splitk_reduce in the pipeline list
constexpr int kTileM = 64, kTileN = 64, kTileK = 16;
// Split-reduction policy (measured with exp_metal_conv): below this many output tiles the GPU is underfilled, so a long reduction is cut into slices.
constexpr int64_t kSplitTargetTiles = 64, kSplitMinK = 512, kSplitMinSteps = 8, kSplitMax = 8;
constexpr int64_t kMaxElements = int64_t(1) << 31;  // kernels index with 32-bit thread ids

struct ConvParams { int32_t ic, ih, iw, oc, oh, ow, sh, sw, K, Kp, flags, ksplit, kslice; };
struct ReduceParams { uint32_t n; int32_t ksplit, oc, ohw, flags; };
struct UnaryParams { uint32_t n; int32_t kind; float scale; };
struct BinaryParams { uint32_t n; int32_t kind, d0, d1, d2, d3, b0, b1, b2, b3; };
struct CopyParams { uint32_t total; int32_t count, inner, src_len, dst_len, src_off, dst_off; };
struct PermuteParams { uint32_t n; int32_t o0, o1, o2, o3, s0, s1, s2, s3; };
struct PoolParams { uint32_t total; int32_t ih, iw, oh, ow, kh, kw, sh, sw, ph, pw; };
struct UpParams { uint32_t total; int32_t ih, iw, oh, ow, sh, sw; };

std::string describe(NSError * e) { return e ? std::string([[e localizedDescription] UTF8String]) : std::string("unknown error"); }

int64_t round_up(int64_t v, int64_t m) { return (v + m - 1) / m * m; }

bool supported_op(OpKind op) {
    switch (op) {
        case OpKind::Conv2d: case OpKind::Add: case OpKind::Mul: case OpKind::Relu: case OpKind::Silu: case OpKind::Scale: case OpKind::Reshape:
        case OpKind::Permute: case OpKind::Slice: case OpKind::Concat: case OpKind::MaxPool2d: case OpKind::UpsampleNearest2d:
            return true;
        default:
            return false;
    }
}

// Float bits <-> half conversion on the host (weights are packed to half).
__fp16 to_half(float f) { return static_cast<__fp16>(f); }

enum class StepKind { Conv, Unary, Binary, Copy, Permute, Pool, Upsample };

// One kernel launch of the compiled program. Values are named by canonical value id; buffers are bound after the memory plan is made.
struct Plan {
    StepKind kind = StepKind::Unary;
    std::vector<ValueId> reads;            // canonical values bound as buffers 0.. in order (a conv additionally gets its weight/ktab/bias/residual below)
    ValueId out = kInvalidValue;
    int pipeline = 0;
    std::vector<uint8_t> params;
    // convolution only
    ValueId weight = kInvalidValue, bias = kInvalidValue, residual = kInvalidValue;
    int ktab = -1;                          // index into Impl::tables
    int ksplit = 1;                         // > 1: the convolution runs as ksplit partial sums into the scratch buffer, then a reduce kernel (this plan's params are the reduce's)
    std::vector<uint8_t> conv_params;       // the convolution kernel's params when ksplit > 1
    int64_t scratch_bytes = 0;
    MTLSize groups{1, 1, 1}, threads{1, 1, 1};
    uint32_t elements = 0;                  // 1-D kernels: thread count
    int node = -1;                          // IR node index (the last one of a fused group) for messages
};

template <class T>
std::vector<uint8_t> bytes_of(const T & v) {
    std::vector<uint8_t> b(sizeof(T));
    std::memcpy(b.data(), &v, sizeof(T));
    return b;
}

}  // namespace

class MetalExecutable;

class MetalBackend final : public Backend {
public:
    MetalBackend(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLLibrary> library, MetalBackendOptions o) : device_(device), queue_(queue), library_(library), options_(o) {}
    const char * name() const override { return "metal"; }
    Status supports(const Graph & graph) const override;
    Result<std::unique_ptr<Executable>> compile(const Graph & graph) override;

private:
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLLibrary> library_;
    MetalBackendOptions options_;
};

class MetalExecutable final : public Executable {
public:
    Graph graph_;
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    std::vector<id<MTLComputePipelineState>> pipelines;
    std::vector<id<MTLBuffer>> buffers;               // every device buffer of the program
    std::vector<id<MTLBuffer>> tables;                // convolution gather tables
    id<MTLBuffer> scratch = nil;                      // partial sums of split-reduction convolutions (one at a time, shared)
    std::vector<Plan> plans;
    std::vector<int> slot_of;                         // canonical value id -> index into buffers (-1: not materialized)
    std::vector<ValueId> canon;                       // value id -> canonical value id (reshape outputs alias their input)
    std::map<std::string, ValueId> weight_ids;
    std::set<std::string> weights_set;
    std::set<ValueId> conv_weights;                   // weights stored packed as half

    const Graph & graph() const override { return graph_; }

    Status set_weight(std::string_view name, const TensorView & data) override;
    Status run(const std::vector<NamedTensor> & inputs, const std::vector<NamedTensor> & outputs) override;
    Result<ProfileReport> profile(const std::vector<NamedTensor> & inputs, int iterations) override;

private:
    void encode(id<MTLComputeCommandEncoder> enc, const Plan & p);
};

Status MetalBackend::supports(const Graph & graph) const {
    for (const Node & n : graph.nodes()) {
        if (!supported_op(n.op)) return unsupported(std::string("the Metal backend does not implement '") + op_name(n.op) + "'");
    }
    for (const Value & v : graph.values()) {
        if (v.kind != ValueKind::Weight && v.type.dtype != DType::F32) return unsupported("the Metal backend computes in float32; got a " + std::string(dtype_name(v.type.dtype)) + " value");
        if (v.kind == ValueKind::Weight && v.type.dtype != DType::F32 && v.type.dtype != DType::F16) return unsupported("weights must be f32 or f16");
        if (v.type.shape.numel() >= kMaxElements) return unsupported("a tensor has 2^31 or more elements");
    }
    for (const Node & n : graph.nodes()) {
        if (n.op == OpKind::Conv2d && graph.values()[n.inputs[1]].kind != ValueKind::Weight) return unsupported("a convolution's kernel must be a graph weight");
    }
    return Status();
}

Result<std::unique_ptr<Backend>> create_metal_backend(const MetalBackendOptions & options) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return unsupported("no Metal device");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) return internal_error("could not create a Metal command queue");
        NSError * error = nil;
        MTLCompileOptions * co = [MTLCompileOptions new];
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:metal_kernel_source()] options:co error:&error];
        if (!library) return internal_error("could not compile the Metal kernels: " + describe(error));
        return std::unique_ptr<Backend>(new MetalBackend(device, queue, library, options));
    }
}

Result<std::unique_ptr<Executable>> MetalBackend::compile(const Graph & graph) {
    @autoreleasepool {
        DOOMS_RETURN_IF_ERROR(supports(graph));
        auto exe = std::make_unique<MetalExecutable>();
        exe->graph_ = graph;
        exe->device = device_;
        exe->queue = queue_;
        const auto & values = graph.values();
        const auto & nodes = graph.nodes();
        const size_t nv = values.size();

        // pipelines
        // pipeline 0: the convolution with the column staging, 1: with the pixel-run staging (function constant), then the other kernels
        static const char * kNames[] = {"conv_igemm", "conv_igemm", "ew_unary", "ew_binary", "copy_axis", "permute4", "maxpool", "upsample_nearest", "splitk_reduce"};
        int which = 0;
        for (const char * n : kNames) {
            NSError * error = nil;
            id<MTLFunction> fn;
            if (std::string(n) == "conv_igemm") {
                MTLFunctionConstantValues * fc = [MTLFunctionConstantValues new];
                bool run_mapping = which == 1;
                [fc setConstantValue:&run_mapping type:MTLDataTypeBool atIndex:0];
                fn = [library_ newFunctionWithName:@"conv_igemm" constantValues:fc error:&error];
            } else {
                fn = [library_ newFunctionWithName:[NSString stringWithUTF8String:n]];
            }
            which++;
            if (!fn) return internal_error(std::string("missing Metal kernel ") + n + ": " + describe(error));
            id<MTLComputePipelineState> pso = [device_ newComputePipelineStateWithFunction:fn error:&error];
            if (!pso) return internal_error(std::string("could not build the pipeline for ") + n + ": " + describe(error));
            exe->pipelines.push_back(pso);
        }
        enum { kConv = 0, kConvRun, kUnary, kBinary, kCopy, kPermute, kPool, kUp, kReduce };

        // reshape outputs alias their input
        exe->canon.resize(nv);
        for (size_t i = 0; i < nv; i++) exe->canon[i] = static_cast<ValueId>(i);
        std::vector<int> producer(nv, -1), uses(nv, 0);
        for (size_t i = 0; i < nodes.size(); i++) {
            producer[static_cast<size_t>(nodes[i].output)] = static_cast<int>(i);
            for (ValueId in : nodes[i].inputs) uses[static_cast<size_t>(in)]++;
            if (nodes[i].op == OpKind::Reshape) exe->canon[static_cast<size_t>(nodes[i].output)] = exe->canon[static_cast<size_t>(nodes[i].inputs[0])];
        }
        for (const GraphOutput & o : graph.outputs()) uses[static_cast<size_t>(o.value)]++;
        auto cv = [&](ValueId v) { return exe->canon[static_cast<size_t>(v)]; };
        // uses are per value; aliasing means the fusion tests below must look at the canonical value's total uses
        std::vector<int> canon_uses(nv, 0);
        for (size_t i = 0; i < nv; i++) canon_uses[static_cast<size_t>(exe->canon[i])] += uses[i];

        // ---- plans ----
        std::vector<bool> consumed(nodes.size(), false);  // nodes folded into an earlier plan
        auto consumer_of = [&](ValueId v) -> int {          // the single node reading value v, or -1
            if (uses[static_cast<size_t>(v)] != 1) return -1;
            for (size_t i = 0; i < nodes.size(); i++)
                for (ValueId in : nodes[i].inputs) if (in == v) return static_cast<int>(i);
            return -1;
        };
        for (size_t ni = 0; ni < nodes.size(); ni++) {
            if (consumed[ni]) continue;
            const Node & n = nodes[ni];
            const Value & out_v = values[static_cast<size_t>(n.output)];
            const Shape & os = out_v.type.shape;
            Plan p;
            p.node = static_cast<int>(ni);
            auto finish_1d = [&](int pipeline, uint32_t count) { p.pipeline = pipeline; p.elements = count; };
            switch (n.op) {
                case OpKind::Reshape:
                    continue;  // an alias: no work
                case OpKind::Conv2d: {
                    auto cp = conv2d_params(n.attrs);
                    if (!cp.is_ok()) return cp.status();
                    const Shape & xs = values[static_cast<size_t>(n.inputs[0])].type.shape;
                    const Shape & ws = values[static_cast<size_t>(n.inputs[1])].type.shape;
                    ConvParams c{};
                    c.ic = static_cast<int32_t>(xs.dim(1)); c.ih = static_cast<int32_t>(xs.dim(2)); c.iw = static_cast<int32_t>(xs.dim(3));
                    c.oc = static_cast<int32_t>(ws.dim(0));
                    const int64_t kh = ws.dim(2), kw = ws.dim(3);
                    c.oh = static_cast<int32_t>(os.dim(2)); c.ow = static_cast<int32_t>(os.dim(3));
                    c.sh = static_cast<int32_t>(cp.value().stride_h); c.sw = static_cast<int32_t>(cp.value().stride_w);
                    const int64_t K = static_cast<int64_t>(c.ic) * kh * kw;
                    if (K >= kMaxElements) return unsupported("a convolution's reduction is too long");
                    c.K = static_cast<int32_t>(K);
                    c.Kp = static_cast<int32_t>(round_up(K, kTileK));
                    c.flags = 0;
                    // gather table
                    std::vector<int32_t> tab(static_cast<size_t>(K) * 4, 0);
                    for (int64_t k = 0; k < K; k++) {
                        const int64_t ic = k / (kh * kw), r = k % (kh * kw), ky = r / kw, kx = r % kw;
                        tab[static_cast<size_t>(k) * 4 + 0] = static_cast<int32_t>(ic * c.ih * c.iw);
                        tab[static_cast<size_t>(k) * 4 + 1] = static_cast<int32_t>(ky * cp.value().dilation_h - cp.value().pad_h);
                        tab[static_cast<size_t>(k) * 4 + 2] = static_cast<int32_t>(kx * cp.value().dilation_w - cp.value().pad_w);
                    }
                    id<MTLBuffer> tb = [device_ newBufferWithBytes:tab.data() length:tab.size() * sizeof(int32_t) options:MTLResourceStorageModeShared];
                    if (!tb) return resource_exhausted("could not allocate a convolution table");
                    exe->tables.push_back(tb);
                    p.ktab = static_cast<int>(exe->tables.size()) - 1;
                    p.kind = StepKind::Conv;
                    // measured on the detector's layers (exp_metal_conv): the pixel-run staging wins for stride-1 layers with many input channels
                    p.pipeline = (c.sh == 1 && c.sw == 1 && c.ic >= 96) ? kConvRun : kConv;
                    p.reads = {n.inputs[0]};
                    p.weight = n.inputs[1];
                    exe->conv_weights.insert(n.inputs[1]);
                    ValueId final_out = n.output;
                    // epilogue fusion: Conv -> Add(per-channel bias) -> Silu|Relu
                    if (options_.fuse_epilogue) {
                        int add_i = consumer_of(final_out);
                        if (add_i >= 0 && nodes[static_cast<size_t>(add_i)].op == OpKind::Add && nodes[static_cast<size_t>(add_i)].inputs[0] == final_out) {
                            const Node & an = nodes[static_cast<size_t>(add_i)];
                            const ValueId b = an.inputs[1];
                            const Value & bv = values[static_cast<size_t>(b)];
                            const ValueId bc = cv(b);
                            const bool bias_ok = values[static_cast<size_t>(bc)].kind == ValueKind::Weight && bv.type.shape.numel() == c.oc && bv.type.shape.rank() >= 1 &&
                                                 bv.type.shape.dim(0) == c.oc && (bv.type.shape.rank() == 1 || (bv.type.shape.dim(bv.type.shape.rank() - 1) == 1 && bv.type.shape.dim(bv.type.shape.rank() - 2) == 1)) &&
                                                 !exe->conv_weights.count(bc);
                            // a [OC,1,1] / [1,OC,1,1] bias broadcast over NCHW: dim(0) == OC with rank 3, or rank 4 with leading 1 handled below
                            const bool shape_ok = bv.type.shape.rank() == 3 && bv.type.shape.dim(0) == c.oc && bv.type.shape.dim(1) == 1 && bv.type.shape.dim(2) == 1;
                            if (bias_ok && shape_ok) {
                                p.bias = bc;
                                c.flags |= 1;
                                consumed[static_cast<size_t>(add_i)] = true;
                                final_out = an.output;
                                int act_i = consumer_of(final_out);
                                if (act_i >= 0 && (nodes[static_cast<size_t>(act_i)].op == OpKind::Silu || nodes[static_cast<size_t>(act_i)].op == OpKind::Relu)) {
                                    c.flags |= (nodes[static_cast<size_t>(act_i)].op == OpKind::Silu ? 1 : 2) << 1;
                                    consumed[static_cast<size_t>(act_i)] = true;
                                    final_out = nodes[static_cast<size_t>(act_i)].output;
                                }
                            }
                        }
                    }
                    p.out = final_out;
                    const int64_t tiles = ((static_cast<int64_t>(c.oh) * c.ow + kTileN - 1) / kTileN) * ((c.oc + kTileM - 1) / kTileM) * xs.dim(0);
                    // Few output tiles and a long reduction leave most of the GPU idle: split the reduction over several threadgroups and add the slices afterwards.
                    c.ksplit = 1;
                    c.kslice = c.Kp;
                    if (tiles < kSplitTargetTiles && c.K >= kSplitMinK) {
                        const int64_t steps = c.Kp / kTileK;
                        int64_t want = std::min<int64_t>({(kSplitTargetTiles + tiles - 1) / tiles, steps / kSplitMinSteps, kSplitMax});
                        if (want > 1) {
                            const int64_t per = (steps + want - 1) / want;       // tile steps per slice
                            c.ksplit = static_cast<int32_t>((steps + per - 1) / per);
                            c.kslice = static_cast<int32_t>(per * kTileK);
                        }
                    }
                    p.groups = MTLSizeMake(static_cast<NSUInteger>((static_cast<int64_t>(c.oh) * c.ow + kTileN - 1) / kTileN), static_cast<NSUInteger>((c.oc + kTileM - 1) / kTileM), static_cast<NSUInteger>(xs.dim(0) * c.ksplit));
                    p.threads = MTLSizeMake(128, 1, 1);
                    if (c.ksplit > 1) {
                        p.ksplit = c.ksplit;
                        p.conv_params = bytes_of(c);
                        p.scratch_bytes = static_cast<int64_t>(c.ksplit) * xs.dim(0) * c.oc * c.oh * c.ow * 4;
                        ReduceParams rp{static_cast<uint32_t>(os.numel()), c.ksplit, c.oc, c.oh * c.ow, c.flags};
                        (void) os;
                        p.params = bytes_of(rp);
                        p.elements = rp.n;
                    } else {
                        p.params = bytes_of(c);
                    }
                    break;
                }
                case OpKind::Add: case OpKind::Mul: {
                    const Shape & as = values[static_cast<size_t>(n.inputs[0])].type.shape;
                    const Shape & bs = values[static_cast<size_t>(n.inputs[1])].type.shape;
                    if (as.rank() > 4) return unsupported("broadcast elementwise ops are limited to rank 4");
                    BinaryParams b{};
                    b.n = static_cast<uint32_t>(os.numel());
                    b.kind = n.op == OpKind::Add ? 0 : 1;
                    int32_t d[4] = {1, 1, 1, 1}, st[4] = {0, 0, 0, 0};
                    // b is right-aligned against a; its stride is 0 where its dim is 1
                    std::vector<int64_t> bstride(static_cast<size_t>(bs.rank()), 1);
                    for (int i = bs.rank() - 2; i >= 0; i--) bstride[static_cast<size_t>(i)] = bstride[static_cast<size_t>(i + 1)] * bs.dim(i + 1);
                    for (int i = 0; i < as.rank(); i++) {
                        const int slot = 4 - as.rank() + i;
                        d[slot] = static_cast<int32_t>(as.dim(i));
                        const int bi = i - (as.rank() - bs.rank());
                        st[slot] = (bi >= 0 && bs.dim(bi) != 1) ? static_cast<int32_t>(bstride[static_cast<size_t>(bi)]) : 0;
                    }
                    b.d0 = d[0]; b.d1 = d[1]; b.d2 = d[2]; b.d3 = d[3]; b.b0 = st[0]; b.b1 = st[1]; b.b2 = st[2]; b.b3 = st[3];
                    p.kind = StepKind::Binary;
                    p.reads = {n.inputs[0], n.inputs[1]};
                    p.out = n.output;
                    p.params = bytes_of(b);
                    finish_1d(kBinary, b.n);
                    break;
                }
                case OpKind::Relu: case OpKind::Silu: case OpKind::Scale: {
                    UnaryParams u{static_cast<uint32_t>(os.numel()), n.op == OpKind::Silu ? 0 : (n.op == OpKind::Relu ? 1 : 2), n.attrs.scalar};
                    p.kind = StepKind::Unary;
                    p.reads = {n.inputs[0]};
                    p.out = n.output;
                    p.params = bytes_of(u);
                    finish_1d(kUnary, u.n);
                    break;
                }
                case OpKind::Permute: {
                    const Shape & xs = values[static_cast<size_t>(n.inputs[0])].type.shape;
                    if (xs.rank() > 4) return unsupported("permute is limited to rank 4");
                    int64_t xstride[4] = {0, 0, 0, 0}, ostride_in[4] = {0, 0, 0, 0};
                    int64_t acc = 1;
                    for (int i = xs.rank() - 1; i >= 0; i--) { xstride[i] = acc; acc *= xs.dim(i); }
                    PermuteParams q{};
                    q.n = static_cast<uint32_t>(os.numel());
                    int32_t od[4] = {1, 1, 1, 1};
                    int32_t is[4] = {0, 0, 0, 0};
                    const int r = xs.rank();
                    for (int i = 0; i < r; i++) {
                        od[4 - r + i] = static_cast<int32_t>(os.dim(i));
                        is[4 - r + i] = static_cast<int32_t>(xstride[n.attrs.ints[static_cast<size_t>(i)]]);
                    }
                    (void) ostride_in;
                    q.o0 = od[0]; q.o1 = od[1]; q.o2 = od[2]; q.o3 = od[3]; q.s0 = is[0]; q.s1 = is[1]; q.s2 = is[2]; q.s3 = is[3];
                    p.kind = StepKind::Permute;
                    p.reads = {n.inputs[0]};
                    p.out = n.output;
                    p.params = bytes_of(q);
                    finish_1d(kPermute, q.n);
                    break;
                }
                case OpKind::Slice: {
                    const Shape & xs = values[static_cast<size_t>(n.inputs[0])].type.shape;
                    const int axis = static_cast<int>(n.attrs.axis);
                    int64_t outer = 1, inner = 1;
                    for (int i = 0; i < axis; i++) outer *= xs.dim(i);
                    for (int i = axis + 1; i < xs.rank(); i++) inner *= xs.dim(i);
                    CopyParams cpar{static_cast<uint32_t>(os.numel()), static_cast<int32_t>(os.dim(axis)), static_cast<int32_t>(inner), static_cast<int32_t>(xs.dim(axis)),
                                    static_cast<int32_t>(os.dim(axis)), static_cast<int32_t>(n.attrs.ints[0]), 0};
                    (void) outer;
                    p.kind = StepKind::Copy;
                    p.reads = {n.inputs[0]};
                    p.out = n.output;
                    p.params = bytes_of(cpar);
                    finish_1d(kCopy, cpar.total);
                    break;
                }
                case OpKind::Concat: {
                    // one copy per input, appended to a list of plans that all write the same output (the memory plan treats them as one producer)
                    const int axis = static_cast<int>(n.attrs.axis);
                    int64_t inner = 1;
                    for (int i = axis + 1; i < os.rank(); i++) inner *= os.dim(i);
                    int64_t offset = 0;
                    for (size_t k = 0; k < n.inputs.size(); k++) {
                        const Shape & xs = values[static_cast<size_t>(n.inputs[k])].type.shape;
                        CopyParams cpar{static_cast<uint32_t>(xs.numel()), static_cast<int32_t>(xs.dim(axis)), static_cast<int32_t>(inner), static_cast<int32_t>(xs.dim(axis)),
                                        static_cast<int32_t>(os.dim(axis)), 0, static_cast<int32_t>(offset)};
                        offset += xs.dim(axis);
                        Plan q;
                        q.node = static_cast<int>(ni);
                        q.kind = StepKind::Copy;
                        q.reads = {n.inputs[k]};
                        q.out = n.output;
                        q.params = bytes_of(cpar);
                        q.pipeline = kCopy;
                        q.elements = cpar.total;
                        exe->plans.push_back(q);
                    }
                    continue;
                }
                case OpKind::MaxPool2d: {
                    auto mp = max_pool2d_params(n.attrs);
                    if (!mp.is_ok()) return mp.status();
                    const Shape & xs = values[static_cast<size_t>(n.inputs[0])].type.shape;
                    PoolParams q{static_cast<uint32_t>(os.numel()), static_cast<int32_t>(xs.dim(2)), static_cast<int32_t>(xs.dim(3)), static_cast<int32_t>(os.dim(2)), static_cast<int32_t>(os.dim(3)),
                                 static_cast<int32_t>(mp.value().kernel_h), static_cast<int32_t>(mp.value().kernel_w), static_cast<int32_t>(mp.value().stride_h), static_cast<int32_t>(mp.value().stride_w),
                                 static_cast<int32_t>(mp.value().pad_h), static_cast<int32_t>(mp.value().pad_w)};
                    p.kind = StepKind::Pool;
                    p.reads = {n.inputs[0]};
                    p.out = n.output;
                    p.params = bytes_of(q);
                    finish_1d(kPool, q.total);
                    break;
                }
                case OpKind::UpsampleNearest2d: {
                    auto up = upsample_nearest2d_params(n.attrs);
                    if (!up.is_ok()) return up.status();
                    const Shape & xs = values[static_cast<size_t>(n.inputs[0])].type.shape;
                    UpParams q{static_cast<uint32_t>(os.numel()), static_cast<int32_t>(xs.dim(2)), static_cast<int32_t>(xs.dim(3)), static_cast<int32_t>(os.dim(2)), static_cast<int32_t>(os.dim(3)),
                               static_cast<int32_t>(up.value().scale_h), static_cast<int32_t>(up.value().scale_w)};
                    p.kind = StepKind::Upsample;
                    p.reads = {n.inputs[0]};
                    p.out = n.output;
                    p.params = bytes_of(q);
                    finish_1d(kUp, q.total);
                    break;
                }
                default:
                    return internal_error("unreachable: unsupported op reached the planner");
            }
            exe->plans.push_back(p);
        }

        // ---- memory plan ----
        // Slots: inputs and weights get their own buffers; every other canonical value gets a pooled buffer from its first write to its last read.
        exe->slot_of.assign(nv, -1);
        std::vector<int> last_read(nv, -1);
        for (size_t pi = 0; pi < exe->plans.size(); pi++) {
            for (ValueId r : exe->plans[pi].reads) last_read[static_cast<size_t>(cv(r))] = static_cast<int>(pi);
            if (exe->plans[pi].weight != kInvalidValue) last_read[static_cast<size_t>(cv(exe->plans[pi].weight))] = static_cast<int>(pi);
            if (exe->plans[pi].bias != kInvalidValue) last_read[static_cast<size_t>(cv(exe->plans[pi].bias))] = static_cast<int>(pi);
        }
        std::set<ValueId> is_output;
        for (const GraphOutput & o : graph.outputs()) is_output.insert(cv(o.value));
        auto make_buffer = [&](int64_t bytes) -> id<MTLBuffer> {
            return [device_ newBufferWithLength:static_cast<NSUInteger>(std::max<int64_t>(bytes, 16)) options:MTLResourceStorageModeShared];
        };
        // dedicated buffers
        for (size_t i = 0; i < nv; i++) {
            const Value & v = values[i];
            if (v.kind == ValueKind::Intermediate) continue;
            int64_t bytes = v.type.shape.numel() * 4;
            if (v.kind == ValueKind::Weight && exe->conv_weights.count(static_cast<ValueId>(i))) {
                const Shape & ws = v.type.shape;
                bytes = round_up(ws.dim(0), kTileM) * round_up(ws.dim(1) * ws.dim(2) * ws.dim(3), kTileK) * 2;
            }
            id<MTLBuffer> b = make_buffer(bytes);
            if (!b) return resource_exhausted("could not allocate a Metal buffer for '" + v.name + "'");
            if (v.kind == ValueKind::Weight && exe->conv_weights.count(static_cast<ValueId>(i))) std::memset([b contents], 0, [b length]);
            exe->buffers.push_back(b);
            exe->slot_of[i] = static_cast<int>(exe->buffers.size()) - 1;
            if (v.kind == ValueKind::Weight) exe->weight_ids[v.name] = static_cast<ValueId>(i);
        }
        struct Pooled { int slot; int64_t bytes; bool free; };
        std::vector<Pooled> pool;
        for (size_t pi = 0; pi < exe->plans.size(); pi++) {
            Plan & p = exe->plans[pi];
            const ValueId o = cv(p.out);
            if (exe->slot_of[static_cast<size_t>(o)] < 0) {  // first writer allocates (a concat's later parts reuse the slot)
                const int64_t bytes = values[static_cast<size_t>(o)].type.shape.numel() * 4;
                int best = -1;
                for (size_t k = 0; k < pool.size(); k++)
                    if (pool[k].free && pool[k].bytes >= bytes && (best < 0 || pool[k].bytes < pool[static_cast<size_t>(best)].bytes)) best = static_cast<int>(k);
                if (best < 0) {
                    id<MTLBuffer> b = make_buffer(bytes);
                    if (!b) return resource_exhausted("could not allocate a Metal buffer of " + std::to_string(bytes) + " bytes");
                    exe->buffers.push_back(b);
                    pool.push_back({static_cast<int>(exe->buffers.size()) - 1, bytes, false});
                    best = static_cast<int>(pool.size()) - 1;
                }
                pool[static_cast<size_t>(best)].free = false;
                exe->slot_of[static_cast<size_t>(o)] = pool[static_cast<size_t>(best)].slot;
            }
            // release inputs whose last read is this plan (never the output's own slot, never graph outputs)
            std::vector<ValueId> done = p.reads;
            for (ValueId r : done) {
                const ValueId c = cv(r);
                if (last_read[static_cast<size_t>(c)] == static_cast<int>(pi) && !is_output.count(c) && values[static_cast<size_t>(c)].kind == ValueKind::Intermediate && c != o) {
                    for (auto & pe : pool) if (pe.slot == exe->slot_of[static_cast<size_t>(c)]) pe.free = true;
                }
            }
        }
        for (const GraphOutput & g : graph.outputs()) {
            if (exe->slot_of[static_cast<size_t>(cv(g.value))] < 0) return internal_error("graph output '" + g.name + "' was never produced");
        }
        int64_t scratch_bytes = 0;
        for (const Plan & p : exe->plans) scratch_bytes = std::max(scratch_bytes, p.scratch_bytes);
        if (scratch_bytes > 0) {
            exe->scratch = make_buffer(scratch_bytes);
            if (!exe->scratch) return resource_exhausted("could not allocate the split-reduction scratch buffer");
        }
        return std::unique_ptr<Executable>(std::move(exe));
    }
}

void MetalExecutable::encode(id<MTLComputeCommandEncoder> enc, const Plan & p) {
    auto buf_of = [&](ValueId v) { return buffers[static_cast<size_t>(slot_of[static_cast<size_t>(canon[static_cast<size_t>(v)])])]; };
    id<MTLComputePipelineState> pso = pipelines[static_cast<size_t>(p.pipeline)];
    [enc setComputePipelineState:pso];
    if (p.kind == StepKind::Conv) {
        [enc setBuffer:buf_of(p.reads[0]) offset:0 atIndex:0];
        [enc setBuffer:buf_of(p.weight) offset:0 atIndex:1];
        [enc setBuffer:tables[static_cast<size_t>(p.ktab)] offset:0 atIndex:2];
        id<MTLBuffer> dummy = buf_of(p.reads[0]);
        [enc setBuffer:(p.bias != kInvalidValue ? buf_of(p.bias) : dummy) offset:0 atIndex:3];
        [enc setBuffer:dummy offset:0 atIndex:4];  // residual (unused until a residual epilogue exists)
        if (p.ksplit > 1) {
            [enc setBuffer:scratch offset:0 atIndex:5];
            [enc setBytes:p.conv_params.data() length:p.conv_params.size() atIndex:6];
            [enc dispatchThreadgroups:p.groups threadsPerThreadgroup:p.threads];
            [enc setComputePipelineState:pipelines[kReducePipeline]];
            [enc setBuffer:scratch offset:0 atIndex:0];
            [enc setBuffer:(p.bias != kInvalidValue ? buf_of(p.bias) : dummy) offset:0 atIndex:1];
            [enc setBuffer:dummy offset:0 atIndex:2];
            [enc setBuffer:buf_of(p.out) offset:0 atIndex:3];
            [enc setBytes:p.params.data() length:p.params.size() atIndex:4];
            const NSUInteger tg = std::min<NSUInteger>(256, pipelines[kReducePipeline].maxTotalThreadsPerThreadgroup);
            [enc dispatchThreads:MTLSizeMake(p.elements, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            return;
        }
        [enc setBuffer:buf_of(p.out) offset:0 atIndex:5];
        [enc setBytes:p.params.data() length:p.params.size() atIndex:6];
        [enc dispatchThreadgroups:p.groups threadsPerThreadgroup:p.threads];
    } else {
        NSUInteger idx = 0;
        for (ValueId r : p.reads) [enc setBuffer:buf_of(r) offset:0 atIndex:idx++];
        [enc setBuffer:buf_of(p.out) offset:0 atIndex:idx++];
        [enc setBytes:p.params.data() length:p.params.size() atIndex:idx];
        const NSUInteger tg = std::min<NSUInteger>(256, pso.maxTotalThreadsPerThreadgroup);
        [enc dispatchThreads:MTLSizeMake(p.elements, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    }
}

Result<ProfileReport> MetalExecutable::profile(const std::vector<NamedTensor> & inputs, int iterations) {
    @autoreleasepool {
        if (iterations < 1 || iterations > 100000) return invalid_argument("iterations must be in [1, 100000]");
        std::vector<ValueId> in_ids;
        DOOMS_RETURN_IF_ERROR(check_input_bindings(graph_, inputs, &in_ids));
        for (const auto & kv : weight_ids) if (!weights_set.count(kv.first)) return invalid_argument("weight '" + kv.first + "' was not set");
        for (size_t i = 0; i < inputs.size(); i++) {
            id<MTLBuffer> b = buffers[static_cast<size_t>(slot_of[static_cast<size_t>(in_ids[i])])];
            std::memcpy([b contents], inputs[i].tensor.data(), static_cast<size_t>(inputs[i].tensor.shape().numel()) * 4);
        }
        const auto & nodes = graph_.nodes();
        std::vector<std::vector<double>> samples(nodes.size());
        std::vector<int> ops(nodes.size(), 0);
        std::vector<double> whole;
        for (int it = 0; it < iterations; it++) {
            {   // the normal, unobserved run
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                for (const Plan & p : plans) encode(enc, p);
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                whole.push_back((cb.GPUEndTime - cb.GPUStartTime) * 1e3);
            }
            std::vector<double> per_node(nodes.size(), 0.0);
            for (const Plan & p : plans) {  // one command buffer per kernel launch: its GPU time, attributed to the IR node
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                encode(enc, p);
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                if (cb.status != MTLCommandBufferStatusCompleted) return internal_error("the Metal command buffer failed: " + describe(cb.error));
                per_node[static_cast<size_t>(p.node)] += (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
                if (it == 0) ops[static_cast<size_t>(p.node)]++;
            }
            for (size_t n = 0; n < nodes.size(); n++) samples[n].push_back(per_node[n]);
        }
        auto med = [](std::vector<double> v) { if (v.empty()) return 0.0; std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
        ProfileReport rep;
        for (size_t n = 0; n < nodes.size(); n++) rep.nodes.push_back({static_cast<int>(n), nodes[n].op, med(samples[n]), ops[n]});
        rep.whole_graph_ms = med(whole);
        rep.observation_overhead_ms = 0;  // GPU timestamps of separate command buffers: no per-operation synchronization cost is included
        rep.iterations = iterations;
        return rep;
    }
}

Status MetalExecutable::set_weight(std::string_view name, const TensorView & data) {
    auto wid = check_weight(graph_, name, data);
    if (!wid.is_ok()) return wid.status();
    const ValueId w = wid.value();
    id<MTLBuffer> buf = buffers[static_cast<size_t>(slot_of[static_cast<size_t>(w)])];
    const Shape & s = graph_.values()[static_cast<size_t>(w)].type.shape;
    const int64_t n = s.numel();
    auto value_at = [&](int64_t i) -> float {
        if (data.dtype() == DType::F32) return static_cast<const float *>(data.data())[i];
        return static_cast<float>(static_cast<const __fp16 *>(data.data())[i]);
    };
    for (int64_t i = 0; i < n; i++) if (!std::isfinite(value_at(i))) return invalid_argument("weight '" + std::string(name) + "' has non-finite values");
    if (conv_weights.count(w)) {
        const int64_t oc = s.dim(0), k = s.dim(1) * s.dim(2) * s.dim(3), kp = round_up(k, kTileK);
        __fp16 * dst = static_cast<__fp16 *>([buf contents]);
        std::memset(dst, 0, [buf length]);
        for (int64_t r = 0; r < oc; r++)
            for (int64_t c = 0; c < k; c++) dst[r * kp + c] = to_half(value_at(r * k + c));
    } else {
        float * dst = static_cast<float *>([buf contents]);
        for (int64_t i = 0; i < n; i++) dst[i] = value_at(i);
    }
    weights_set.insert(std::string(name));
    return Status();
}

Status MetalExecutable::run(const std::vector<NamedTensor> & inputs, const std::vector<NamedTensor> & outputs) {
    @autoreleasepool {
        std::vector<ValueId> in_ids, out_ids;
        DOOMS_RETURN_IF_ERROR(check_bindings(graph_, inputs, outputs, &in_ids, &out_ids));
        for (const auto & kv : weight_ids) if (!weights_set.count(kv.first)) return invalid_argument("weight '" + kv.first + "' was not set");
        for (size_t i = 0; i < inputs.size(); i++) {
            id<MTLBuffer> b = buffers[static_cast<size_t>(slot_of[static_cast<size_t>(in_ids[i])])];
            std::memcpy([b contents], inputs[i].tensor.data(), static_cast<size_t>(inputs[i].tensor.shape().numel()) * 4);
        }
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        for (const Plan & p : plans) encode(enc, p);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) return internal_error("the Metal command buffer failed: " + describe(cb.error));
        for (size_t i = 0; i < outputs.size(); i++) {
            id<MTLBuffer> b = buffers[static_cast<size_t>(slot_of[static_cast<size_t>(canon[static_cast<size_t>(out_ids[i])])])];
            std::memcpy(outputs[i].tensor.data(), [b contents], static_cast<size_t>(outputs[i].tensor.shape().numel()) * 4);
        }
        return Status();
    }
}

}  // namespace dooms
