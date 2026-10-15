#include <limits>
#include <string>
#include <vector>

#include "check.h"
#include "ir/cost.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "ir/shape_infer.h"

using namespace dooms;

static Shape S(std::initializer_list<int64_t> d) { return Shape::create(d).value(); }
static TypeInfo T(std::initializer_list<int64_t> d, DType t = DType::F32) { return TypeInfo{S(d), t}; }

static Code infer_code(OpKind op, const std::vector<TypeInfo> & in, const Attrs & a = Attrs()) {
    auto r = infer_output(op, in, a);
    return r.is_ok() ? Code::Ok : r.status().code();
}

// ---- ops ----

TEST(op_names_and_arity_roundtrip) {
    const OpKind all[] = {OpKind::Add, OpKind::Mul, OpKind::MatMul, OpKind::Gelu,
                          OpKind::Softmax, OpKind::LayerNorm, OpKind::Reshape, OpKind::Permute,
                          OpKind::Slice, OpKind::Concat, OpKind::Conv2d, OpKind::Scale, OpKind::SelfAttention, OpKind::Relu, OpKind::ConvTranspose2d};
    const int arity[] = {2, 2, 2, 1, 1, 3, 1, 1, 1, kVariadic, 2, 1, 1, 1, 2};
    for (size_t i = 0; i < sizeof(arity) / sizeof(arity[0]); i++) {
        auto back = op_from_name(op_name(all[i]));
        EXPECT(back.is_ok());
        EXPECT(back.value() == all[i]);
        EXPECT_EQ(op_arity(all[i]), arity[i]);
    }
    EXPECT(op_from_name("conv9d").status().code() == Code::InvalidArgument);
}

// ---- shape inference: elementwise ----

TEST(infer_add_same_shape_and_bias_broadcast) {
    auto same = infer_output(OpKind::Add, {T({2, 3}), T({2, 3})}, Attrs());
    EXPECT(same.is_ok());
    EXPECT(same.value() == T({2, 3}));
    auto bias = infer_output(OpKind::Add, {T({8, 192, 1280}), T({1280})}, Attrs());
    EXPECT(bias.is_ok());
    EXPECT(bias.value() == T({8, 192, 1280}));
    auto ones = infer_output(OpKind::Mul, {T({4, 5}), T({4, 1})}, Attrs());
    EXPECT(ones.is_ok());
}

TEST(infer_add_rejects_bad_operands) {
    EXPECT(infer_code(OpKind::Add, {T({2, 3}), T({3, 2})}) == Code::InvalidArgument);
    EXPECT(infer_code(OpKind::Add, {T({3}), T({2, 3})}) == Code::InvalidArgument);  // b has higher rank
    EXPECT(infer_code(OpKind::Add, {T({2, 3}), T({2, 3}, DType::F16)}) == Code::InvalidArgument);
    EXPECT(infer_code(OpKind::Add, {T({2, 3})}) == Code::InvalidArgument);          // wrong arity
}

// ---- shape inference: matmul ----

TEST(infer_matmul_plain_and_transposed) {
    Attrs plain;
    auto r = infer_output(OpKind::MatMul, {T({8, 256}), T({256, 1024})}, plain);
    EXPECT(r.is_ok());
    EXPECT(r.value() == T({8, 1024}));

    Attrs tr;
    tr.transpose_b = true;
    auto r2 = infer_output(OpKind::MatMul, {T({8, 256}), T({1024, 256})}, tr);
    EXPECT(r2.is_ok());
    EXPECT(r2.value() == T({8, 1024}));
}

TEST(infer_matmul_weight_broadcast_over_batch) {
    Attrs tr;
    tr.transpose_b = true;
    auto r = infer_output(OpKind::MatMul, {T({2, 192, 1280}), T({3840, 1280})}, tr);
    EXPECT(r.is_ok());
    EXPECT(r.value() == T({2, 192, 3840}));
}

TEST(infer_matmul_attention_scores) {
    Attrs tr;
    tr.transpose_b = true;
    auto r = infer_output(OpKind::MatMul, {T({2, 16, 192, 80}), T({2, 16, 192, 80})}, tr);
    EXPECT(r.is_ok());
    EXPECT(r.value() == T({2, 16, 192, 192}));
}

TEST(infer_matmul_accepts_low_precision_weights_under_f32_activations) {
    Attrs tr;
    tr.transpose_b = true;
    for (DType w : {DType::F16, DType::BF16}) {
        auto r = infer_output(OpKind::MatMul, {T({2, 192, 1280}), T({3840, 1280}, w)}, tr);
        EXPECT(r.is_ok());
        if (r.is_ok()) EXPECT(r.value() == T({2, 192, 3840}));   // the result has the activations' dtype
    }
    Graph g;
    auto x = g.add_input("x", S({8, 256}), DType::F32).value();
    auto w = g.add_weight("w", S({1024, 256}), DType::F16).value();
    g.add_matmul(x, w, true).value();
    NodeCost c = node_cost(g, g.nodes().back());
    EXPECT_EQ(c.bytes_read, 4.0 * 8 * 256 + 2.0 * 1024 * 256);   // weights count 2 bytes per element
    EXPECT_EQ(c.flops, 2.0 * 8 * 1024 * 256);
}

TEST(infer_matmul_rejects_bad_operands) {
    Attrs plain, tr;
    tr.transpose_b = true;
    EXPECT(infer_code(OpKind::MatMul, {T({8, 256}), T({255, 4})}, plain) == Code::InvalidArgument);  // K mismatch
    EXPECT(infer_code(OpKind::MatMul, {T({8, 256}), T({256, 4})}, tr) == Code::InvalidArgument);     // wrong layout
    EXPECT(infer_code(OpKind::MatMul, {T({256}), T({256, 4})}, plain) == Code::InvalidArgument);     // rank 1
    EXPECT(infer_code(OpKind::MatMul, {T({2, 3, 4}), T({5, 4, 6})}, plain) == Code::InvalidArgument);  // batch mismatch
    EXPECT(infer_code(OpKind::MatMul, {T({2, 3, 4}), T({4, 4, 6})}, plain) == Code::InvalidArgument);  // rank mismatch
    EXPECT(infer_code(OpKind::MatMul, {T({2, 3}, DType::I32), T({3, 2}, DType::I32)}, plain) == Code::InvalidArgument);
    EXPECT(infer_code(OpKind::MatMul, {T({2, 3}, DType::F16), T({3, 2}, DType::F32)}, plain) == Code::InvalidArgument);   // f16 activations x f32 weights
    EXPECT(infer_code(OpKind::MatMul, {T({2, 3}, DType::F16), T({3, 2}, DType::BF16)}, plain) == Code::InvalidArgument);  // two different low precisions
    EXPECT(infer_code(OpKind::MatMul, {T({2, 3}), T({3, 2}, DType::I32)}, plain) == Code::InvalidArgument);               // integer weights
    EXPECT(infer_code(OpKind::MatMul, {T({2, 3}, DType::BF16), T({3, 2}, DType::F16)}, plain) == Code::InvalidArgument);
}

// ---- shape inference: unary and normalization ----

TEST(infer_gelu_and_softmax) {
    EXPECT(infer_output(OpKind::Gelu, {T({4, 5}, DType::F16)}, Attrs()).value() == T({4, 5}, DType::F16));
    EXPECT(infer_code(OpKind::Gelu, {T({4}, DType::I32)}) == Code::InvalidArgument);

    Attrs a;
    a.axis = 1;
    EXPECT(infer_output(OpKind::Softmax, {T({4, 5})}, a).is_ok());
    a.axis = 2;
    EXPECT(infer_code(OpKind::Softmax, {T({4, 5})}, a) == Code::OutOfRange);
    a.axis = -1;
    EXPECT(infer_code(OpKind::Softmax, {T({4, 5})}, a) == Code::OutOfRange);
}

TEST(infer_layer_norm) {
    Attrs a;
    a.eps = 1e-6f;
    EXPECT(infer_output(OpKind::LayerNorm, {T({2, 192, 1280}), T({1280}), T({1280})}, a).is_ok());
    EXPECT(infer_code(OpKind::LayerNorm, {T({2, 192, 1280}), T({1279}), T({1280})}, a) == Code::InvalidArgument);
    EXPECT(infer_code(OpKind::LayerNorm, {T({2, 192, 1280}), T({1280}), T({1, 1280})}, a) == Code::InvalidArgument);
    EXPECT(infer_code(OpKind::LayerNorm, {T({2, 1280}), T({1280}, DType::F16), T({1280})}, a) == Code::InvalidArgument);
    Attrs bad;  // eps == 0
    EXPECT(infer_code(OpKind::LayerNorm, {T({2, 4}), T({4}), T({4})}, bad) == Code::InvalidArgument);
    bad.eps = -1e-6f;
    EXPECT(infer_code(OpKind::LayerNorm, {T({2, 4}), T({4}), T({4})}, bad) == Code::InvalidArgument);
}

// ---- shape inference: layout ops ----

TEST(infer_reshape) {
    Attrs a;
    a.ints = {2, 192, 16, 80};
    EXPECT(infer_output(OpKind::Reshape, {T({2, 192, 1280})}, a).value() == T({2, 192, 16, 80}));
    a.ints = {2, 192, 16, 81};
    EXPECT(infer_code(OpKind::Reshape, {T({2, 192, 1280})}, a) == Code::InvalidArgument);  // numel mismatch
    a.ints = {2, 0};
    EXPECT(infer_code(OpKind::Reshape, {T({2, 192})}, a) == Code::InvalidArgument);        // zero dim
    a.ints = {1, 1, 1, 1, 1};
    EXPECT(infer_code(OpKind::Reshape, {T({1})}, a) == Code::InvalidArgument);             // rank > max
}

TEST(infer_permute) {
    Attrs a;
    a.ints = {0, 2, 1, 3};
    EXPECT(infer_output(OpKind::Permute, {T({2, 192, 16, 80})}, a).value() == T({2, 16, 192, 80}));
    a.ints = {0, 2, 2, 3};
    EXPECT(infer_code(OpKind::Permute, {T({2, 192, 16, 80})}, a) == Code::InvalidArgument);  // repeated
    a.ints = {0, 1, 2, 4};
    EXPECT(infer_code(OpKind::Permute, {T({2, 192, 16, 80})}, a) == Code::OutOfRange);
    a.ints = {1, 0};
    EXPECT(infer_code(OpKind::Permute, {T({2, 3, 4})}, a) == Code::InvalidArgument);         // wrong length
}

// ---- shape inference: slice / concat / conv2d ----

static Attrs slice_attrs(int64_t axis, int64_t start, int64_t end) {
    Attrs a;
    a.axis = axis;
    a.ints = {start, end};
    return a;
}

TEST(infer_slice) {
    auto r = infer_output(OpKind::Slice, {T({2, 192, 3840})}, slice_attrs(2, 1280, 2560));
    EXPECT(r.is_ok());
    EXPECT(r.value() == T({2, 192, 1280}));
    EXPECT(infer_output(OpKind::Slice, {T({4, 5}, DType::F16)}, slice_attrs(0, 3, 4)).value() == T({1, 5}, DType::F16));
    EXPECT(infer_output(OpKind::Slice, {T({4, 5})}, slice_attrs(1, 0, 5)).value() == T({4, 5}));  // full range
}

TEST(infer_slice_rejects_bad_ranges) {
    EXPECT(infer_code(OpKind::Slice, {T({4, 5})}, slice_attrs(2, 0, 1)) == Code::OutOfRange);   // axis
    EXPECT(infer_code(OpKind::Slice, {T({4, 5})}, slice_attrs(-1, 0, 1)) == Code::OutOfRange);
    EXPECT(infer_code(OpKind::Slice, {T({4, 5})}, slice_attrs(0, -1, 2)) == Code::OutOfRange);  // negative start
    EXPECT(infer_code(OpKind::Slice, {T({4, 5})}, slice_attrs(0, 2, 2)) == Code::OutOfRange);   // empty
    EXPECT(infer_code(OpKind::Slice, {T({4, 5})}, slice_attrs(0, 3, 2)) == Code::OutOfRange);   // reversed
    EXPECT(infer_code(OpKind::Slice, {T({4, 5})}, slice_attrs(0, 0, 5)) == Code::OutOfRange);   // past the end
    Attrs wrong_count = slice_attrs(0, 0, 1);
    wrong_count.ints.push_back(7);
    EXPECT(infer_code(OpKind::Slice, {T({4, 5})}, wrong_count) == Code::InvalidArgument);
}

TEST(infer_concat) {
    Attrs a;
    a.axis = 1;
    auto r = infer_output(OpKind::Concat, {T({2, 3, 4}), T({2, 5, 4}), T({2, 1, 4})}, a);
    EXPECT(r.is_ok());
    EXPECT(r.value() == T({2, 9, 4}));
    a.axis = 0;
    EXPECT(infer_output(OpKind::Concat, {T({1, 8}), T({2, 8})}, a).value() == T({3, 8}));
}

TEST(infer_concat_rejects_bad_inputs) {
    Attrs a;
    a.axis = 1;
    EXPECT(infer_code(OpKind::Concat, {T({2, 3})}, a) == Code::InvalidArgument);                              // one input
    EXPECT(infer_code(OpKind::Concat, {T({2, 3}), T({3, 3})}, a) == Code::InvalidArgument);                    // other dim differs
    EXPECT(infer_code(OpKind::Concat, {T({2, 3}), T({2, 3, 1})}, a) == Code::InvalidArgument);                 // rank differs
    EXPECT(infer_code(OpKind::Concat, {T({2, 3}), T({2, 3}, DType::F16)}, a) == Code::InvalidArgument);        // dtype differs
    a.axis = 2;
    EXPECT(infer_code(OpKind::Concat, {T({2, 3}), T({2, 3})}, a) == Code::OutOfRange);                         // axis
    EXPECT(infer_code(OpKind::Concat, {T({}), T({})}, a) == Code::InvalidArgument);                            // scalars
}

TEST(infer_concat_detects_size_overflow) {
    const int64_t big = std::numeric_limits<int64_t>::max() / 2 + 1;
    Attrs a;
    a.axis = 0;
    EXPECT(infer_code(OpKind::Concat, {T({big}), T({big})}, a) == Code::ResourceExhausted);
}

TEST(conv2d_params_roundtrip_and_validation) {
    Conv2dParams p;
    p.stride_h = 16; p.stride_w = 8; p.pad_h = 4; p.pad_w = 2; p.dilation_h = 1; p.dilation_w = 3;
    auto back = conv2d_params(conv2d_attrs(p));
    EXPECT(back.is_ok());
    EXPECT_EQ(back.value().stride_h, int64_t(16));
    EXPECT_EQ(back.value().stride_w, int64_t(8));
    EXPECT_EQ(back.value().pad_h, int64_t(4));
    EXPECT_EQ(back.value().pad_w, int64_t(2));
    EXPECT_EQ(back.value().dilation_h, int64_t(1));
    EXPECT_EQ(back.value().dilation_w, int64_t(3));

    Attrs a = conv2d_attrs(Conv2dParams());
    a.ints[0] = 0;  // stride 0
    EXPECT(conv2d_params(a).status().code() == Code::InvalidArgument);
    a = conv2d_attrs(Conv2dParams());
    a.ints[2] = -1;  // negative padding
    EXPECT(conv2d_params(a).status().code() == Code::InvalidArgument);
    a = conv2d_attrs(Conv2dParams());
    a.ints[5] = 0;  // dilation 0
    EXPECT(conv2d_params(a).status().code() == Code::InvalidArgument);
    a.ints.pop_back();
    EXPECT(conv2d_params(a).status().code() == Code::InvalidArgument);  // wrong count
}

TEST(infer_conv2d_patch_embed_like_vit_h) {
    // ViT-H patch embed (values read from the real checkpoint by tools/convert): 16x16 kernel, stride 16,
    // padding 2, on a 256x192 image -> a 16x12 grid of patches.
    Conv2dParams p;
    p.stride_h = p.stride_w = 16;
    p.pad_h = p.pad_w = 2;
    auto r = infer_output(OpKind::Conv2d, {T({1, 3, 256, 192}), T({1280, 3, 16, 16})}, conv2d_attrs(p));
    EXPECT(r.is_ok());
    EXPECT(r.value() == T({1, 1280, 16, 12}));
}

TEST(infer_conv2d_stride_pad_dilation_arithmetic) {
    Conv2dParams p;  // 3x3 on 7x9, stride 1, no pad -> 5x7
    EXPECT(infer_output(OpKind::Conv2d, {T({2, 4, 7, 9}), T({6, 4, 3, 3})}, conv2d_attrs(p)).value() == T({2, 6, 5, 7}));
    p.pad_h = p.pad_w = 1;  // "same" padding -> 7x9
    EXPECT(infer_output(OpKind::Conv2d, {T({2, 4, 7, 9}), T({6, 4, 3, 3})}, conv2d_attrs(p)).value() == T({2, 6, 7, 9}));
    p = Conv2dParams();
    p.stride_h = 2; p.stride_w = 3;  // floor division
    EXPECT(infer_output(OpKind::Conv2d, {T({1, 1, 8, 10}), T({1, 1, 3, 3})}, conv2d_attrs(p)).value() == T({1, 1, 3, 3}));
    p = Conv2dParams();
    p.dilation_h = p.dilation_w = 2;  // effective kernel 5 -> 8-5+1 = 4
    EXPECT(infer_output(OpKind::Conv2d, {T({1, 1, 8, 8}), T({1, 1, 3, 3})}, conv2d_attrs(p)).value() == T({1, 1, 4, 4}));
}

TEST(infer_conv2d_rejects_bad_operands) {
    const Attrs ok = conv2d_attrs(Conv2dParams());
    EXPECT(infer_code(OpKind::Conv2d, {T({1, 3, 8, 8}), T({4, 2, 3, 3})}, ok) == Code::InvalidArgument);   // channels
    EXPECT(infer_code(OpKind::Conv2d, {T({3, 8, 8}), T({4, 3, 3, 3})}, ok) == Code::InvalidArgument);      // input rank
    EXPECT(infer_code(OpKind::Conv2d, {T({1, 3, 8, 8}), T({4, 3, 3})}, ok) == Code::InvalidArgument);      // weight rank
    EXPECT(infer_code(OpKind::Conv2d, {T({1, 3, 8, 8}, DType::F16), T({4, 3, 3, 3})}, ok) == Code::InvalidArgument);
    EXPECT(infer_code(OpKind::Conv2d, {T({1, 3, 2, 2}), T({4, 3, 3, 3})}, ok) == Code::InvalidArgument);   // kernel > input
    EXPECT(infer_code(OpKind::Conv2d, {T({1, 3, 8, 8}), T({4, 3, 3, 3})}, Attrs()) == Code::InvalidArgument);  // no params
    Conv2dParams big;
    big.pad_h = std::numeric_limits<int64_t>::max() / 2 + 1;
    EXPECT(infer_code(OpKind::Conv2d, {T({1, 3, 8, 8}), T({4, 3, 3, 3})}, conv2d_attrs(big)) == Code::ResourceExhausted);
}

TEST(infer_scale) {
    Attrs a;
    a.scalar = 0.125f;
    EXPECT(infer_output(OpKind::Scale, {T({2, 3})}, a).value() == T({2, 3}));
    EXPECT(infer_output(OpKind::Scale, {T({4}, DType::F16)}, a).value() == T({4}, DType::F16));
    a.scalar = -3.f;  // negative and zero factors are legitimate
    EXPECT(infer_output(OpKind::Scale, {T({2})}, a).is_ok());
    a.scalar = 0.f;
    EXPECT(infer_output(OpKind::Scale, {T({2})}, a).is_ok());
    EXPECT(infer_code(OpKind::Scale, {T({2}, DType::I32)}, a) == Code::InvalidArgument);
    a.scalar = std::numeric_limits<float>::infinity();
    EXPECT(infer_code(OpKind::Scale, {T({2})}, a) == Code::InvalidArgument);
    a.scalar = std::numeric_limits<float>::quiet_NaN();
    EXPECT(infer_code(OpKind::Scale, {T({2})}, a) == Code::InvalidArgument);
}

TEST(graph_scale_helper_and_dump) {
    Graph g;
    auto x = g.add_input("x", S({2, 2}), DType::F32).value();
    auto y = g.add_scale(x, 0.5f);
    EXPECT(y.is_ok());
    EXPECT(g.mark_output("y", y.value()).is_ok());
    EXPECT(g.dump().find("scale(%0) scalar=0.5") != std::string::npos);
    EXPECT(!g.add_scale(x, std::numeric_limits<float>::infinity()).is_ok());
    EXPECT_EQ(g.nodes().size(), size_t(1));  // the failed add left the graph unchanged
}

// ---- self attention ----

static Attrs attn_attrs(int64_t heads, float scale = 0.125f) {
    SelfAttentionParams p;
    p.heads = heads;
    p.scale = scale;
    return self_attention_attrs(p);
}

TEST(self_attention_params_roundtrip_and_validation) {
    auto back = self_attention_params(attn_attrs(16, 0.1118f));
    EXPECT(back.is_ok());
    EXPECT_EQ(back.value().heads, int64_t(16));
    EXPECT_EQ(back.value().scale, 0.1118f);
    EXPECT(self_attention_params(attn_attrs(0)).status().code() == Code::InvalidArgument);
    EXPECT(self_attention_params(attn_attrs(-2)).status().code() == Code::InvalidArgument);
    EXPECT(self_attention_params(attn_attrs(4, std::numeric_limits<float>::infinity())).status().code() == Code::InvalidArgument);
    EXPECT(self_attention_params(attn_attrs(4, std::numeric_limits<float>::quiet_NaN())).status().code() == Code::InvalidArgument);
    Attrs two = attn_attrs(4);
    two.ints.push_back(7);
    EXPECT(self_attention_params(two).status().code() == Code::InvalidArgument);
    EXPECT(self_attention_params(Attrs()).status().code() == Code::InvalidArgument);  // no heads
}

TEST(infer_self_attention) {
    // ViT-H: packed qkv for 210 tokens, E = 1280 (3840 packed), 16 heads
    auto r = infer_output(OpKind::SelfAttention, {T({1, 210, 3840})}, attn_attrs(16));
    EXPECT(r.is_ok());
    EXPECT(r.value() == T({1, 210, 1280}));
    EXPECT(infer_output(OpKind::SelfAttention, {T({2, 5, 24}, DType::F16)}, attn_attrs(4)).value() == T({2, 5, 8}, DType::F16));
    EXPECT(infer_output(OpKind::SelfAttention, {T({1, 1, 3})}, attn_attrs(1)).is_ok());  // smallest case
}

TEST(infer_self_attention_rejections) {
    EXPECT(infer_code(OpKind::SelfAttention, {T({1, 210, 3841})}, attn_attrs(16)) == Code::InvalidArgument);   // not a multiple of 3
    EXPECT(infer_code(OpKind::SelfAttention, {T({1, 210, 3840})}, attn_attrs(7)) == Code::InvalidArgument);    // 1280 % 7 != 0
    EXPECT(infer_code(OpKind::SelfAttention, {T({210, 3840})}, attn_attrs(16)) == Code::InvalidArgument);      // rank 2
    EXPECT(infer_code(OpKind::SelfAttention, {T({1, 1, 210, 3840})}, attn_attrs(16)) == Code::InvalidArgument);// rank 4
    EXPECT(infer_code(OpKind::SelfAttention, {T({1, 4, 12}, DType::I32)}, attn_attrs(2)) == Code::InvalidArgument);
    EXPECT(infer_code(OpKind::SelfAttention, {T({1, 4, 12})}, attn_attrs(0)) == Code::InvalidArgument);        // bad heads
    EXPECT(infer_code(OpKind::SelfAttention, {T({1, 4, 12})}, Attrs()) == Code::InvalidArgument);              // no params
    EXPECT(infer_code(OpKind::SelfAttention, {T({1, 4, 12}), T({1, 4, 12})}, attn_attrs(2)) == Code::InvalidArgument);  // arity
    Attrs stray = attn_attrs(2);
    stray.axis = 1;                                                                                              // attribute it does not use
    EXPECT(infer_code(OpKind::SelfAttention, {T({1, 4, 12})}, stray) == Code::InvalidArgument);
}

TEST(graph_self_attention_helper_dump_and_cost) {
    Graph g;
    auto qkv = g.add_input("qkv", S({1, 210, 3840}), DType::F32).value();
    SelfAttentionParams p;
    p.heads = 16;
    p.scale = 0.125f;
    auto y = g.add_self_attention(qkv, p);
    EXPECT(y.is_ok());
    EXPECT(g.values()[y.value()].type == T({1, 210, 1280}));
    EXPECT(g.mark_output("y", y.value()).is_ok());
    EXPECT(g.validate().is_ok());
    EXPECT(g.dump().find("self_attention(%0) scale=0.125 [16]") != std::string::npos);
    EXPECT_EQ(node_cost(g, g.nodes().back()).flops, 4.0 * 1 * 210 * 210 * 1280);   // q.k^T plus p.v
    EXPECT_EQ(node_cost(g, g.nodes().back()).bytes_read, 4.0 * 210 * 3840);
    EXPECT(!g.add_self_attention(qkv, SelfAttentionParams{7, 1.f}).is_ok());       // 1280 % 7 != 0
    EXPECT_EQ(g.nodes().size(), size_t(1));
}

// ---- relu ----

TEST(infer_relu) {
    EXPECT(infer_output(OpKind::Relu, {T({2, 3})}, Attrs()).value() == T({2, 3}));
    EXPECT(infer_output(OpKind::Relu, {T({4}, DType::F16)}, Attrs()).value() == T({4}, DType::F16));
    EXPECT(infer_code(OpKind::Relu, {T({4}, DType::I32)}) == Code::InvalidArgument);
    EXPECT(infer_code(OpKind::Relu, {T({4}), T({4})}) == Code::InvalidArgument);   // arity
    Attrs stray;
    stray.scalar = 1.f;
    EXPECT(infer_code(OpKind::Relu, {T({4})}, stray) == Code::InvalidArgument);
}

// ---- conv_transpose2d ----

static Attrs convt_attrs(int64_t sh, int64_t sw, int64_t ph, int64_t pw) {
    ConvTranspose2dParams p;
    p.stride_h = sh; p.stride_w = sw; p.pad_h = ph; p.pad_w = pw;
    return conv_transpose2d_attrs(p);
}

TEST(conv_transpose2d_params_roundtrip_and_validation) {
    auto back = conv_transpose2d_params(convt_attrs(2, 3, 1, 4));
    EXPECT(back.is_ok());
    EXPECT_EQ(back.value().stride_h, int64_t(2));
    EXPECT_EQ(back.value().stride_w, int64_t(3));
    EXPECT_EQ(back.value().pad_h, int64_t(1));
    EXPECT_EQ(back.value().pad_w, int64_t(4));
    EXPECT(conv_transpose2d_params(convt_attrs(0, 1, 0, 0)).status().code() == Code::InvalidArgument);
    EXPECT(conv_transpose2d_params(convt_attrs(1, -1, 0, 0)).status().code() == Code::InvalidArgument);
    EXPECT(conv_transpose2d_params(convt_attrs(1, 1, -1, 0)).status().code() == Code::InvalidArgument);
    Attrs three = convt_attrs(1, 1, 0, 0);
    three.ints.pop_back();
    EXPECT(conv_transpose2d_params(three).status().code() == Code::InvalidArgument);
}

TEST(infer_conv_transpose2d_sizes) {
    // RefineNet: kernel 4, stride 2, padding 1 exactly doubles the size: (in-1)*2 - 2 + 4 = 2*in
    EXPECT(infer_output(OpKind::ConvTranspose2d, {T({1, 640, 16, 12}), T({640, 320, 4, 4})}, convt_attrs(2, 2, 1, 1)).value() == T({1, 320, 32, 24}));
    EXPECT(infer_output(OpKind::ConvTranspose2d, {T({1, 320, 32, 24}), T({320, 160, 4, 4})}, convt_attrs(2, 2, 1, 1)).value() == T({1, 160, 64, 48}));
    // other geometries: out = (in - 1) * stride - 2 * pad + kernel, per axis
    EXPECT(infer_output(OpKind::ConvTranspose2d, {T({2, 3, 5, 7}), T({3, 4, 3, 2})}, convt_attrs(1, 1, 0, 0)).value() == T({2, 4, 7, 8}));
    EXPECT(infer_output(OpKind::ConvTranspose2d, {T({1, 2, 4, 4}), T({2, 1, 2, 2})}, convt_attrs(2, 2, 0, 0)).value() == T({1, 1, 8, 8}));
    EXPECT(infer_output(OpKind::ConvTranspose2d, {T({1, 2, 4, 6}), T({2, 1, 3, 5})}, convt_attrs(3, 2, 1, 2)).value() == T({1, 1, 10, 11}));
}

TEST(infer_conv_transpose2d_rejections) {
    const Attrs ok = convt_attrs(2, 2, 1, 1);
    EXPECT(infer_code(OpKind::ConvTranspose2d, {T({1, 640, 16, 12}), T({320, 640, 4, 4})}, ok) == Code::InvalidArgument);   // weight is [OC,IC,..]: channel mismatch
    EXPECT(infer_code(OpKind::ConvTranspose2d, {T({640, 16, 12}), T({640, 320, 4, 4})}, ok) == Code::InvalidArgument);      // input rank
    EXPECT(infer_code(OpKind::ConvTranspose2d, {T({1, 640, 16, 12}), T({640, 320, 4})}, ok) == Code::InvalidArgument);      // weight rank
    EXPECT(infer_code(OpKind::ConvTranspose2d, {T({1, 4, 4, 4}, DType::F16), T({4, 4, 2, 2})}, ok) == Code::InvalidArgument);
    EXPECT(infer_code(OpKind::ConvTranspose2d, {T({1, 4, 4, 4}), T({4, 4, 2, 2})}, Attrs()) == Code::InvalidArgument);        // no params
    EXPECT(infer_code(OpKind::ConvTranspose2d, {T({1, 1, 1, 1}), T({1, 1, 2, 2})}, convt_attrs(1, 1, 5, 0)) == Code::InvalidArgument);  // padding removes everything
    EXPECT(infer_code(OpKind::ConvTranspose2d, {T({1, 1, 1, 1}), T({1, 1, 2, 2})}, convt_attrs(1, 1, 1, 0)) == Code::InvalidArgument);  // exactly zero left
    const int64_t big = std::numeric_limits<int64_t>::max() / 2 + 1;
    EXPECT(infer_code(OpKind::ConvTranspose2d, {T({1, 1, 3, 3}), T({1, 1, 2, 2})}, convt_attrs(big, 1, 0, 0)) == Code::ResourceExhausted);
}

TEST(graph_conv_transpose2d_helper_dump_and_cost) {
    Graph g;
    auto x = g.add_input("x", S({1, 640, 16, 12}), DType::F32).value();
    auto w = g.add_weight("w", S({640, 320, 4, 4}), DType::F32).value();
    ConvTranspose2dParams p;
    p.stride_h = p.stride_w = 2;
    p.pad_h = p.pad_w = 1;
    auto y = g.add_conv_transpose2d(x, w, p);
    EXPECT(y.is_ok());
    EXPECT(g.values()[y.value()].type == T({1, 320, 32, 24}));
    auto r = g.add_relu(y.value());
    EXPECT(r.is_ok());
    EXPECT(g.mark_output("y", r.value()).is_ok());
    EXPECT(g.validate().is_ok());
    EXPECT(g.dump().find("conv_transpose2d(%0, %1) [2, 2, 1, 1]") != std::string::npos);
    EXPECT(g.dump().find("relu(%2)") != std::string::npos);
    // each of the 16*12 input pixels meets the whole [320, 4, 4] slice of its 640 channels
    EXPECT_EQ(node_cost(g, g.nodes()[0]).flops, 2.0 * (640.0 * 16 * 12) * (320 * 4 * 4));
    EXPECT(!g.add_conv_transpose2d(x, x, p).is_ok());   // weight must be [IC, OC, KH, KW]
    EXPECT_EQ(g.nodes().size(), size_t(2));
}

// ---- silu / max_pool2d / upsample_nearest2d (detector ops) ----

static Attrs pool_attrs(int64_t kh, int64_t kw, int64_t sh, int64_t sw, int64_t ph, int64_t pw) {
    MaxPool2dParams p;
    p.kernel_h = kh; p.kernel_w = kw; p.stride_h = sh; p.stride_w = sw; p.pad_h = ph; p.pad_w = pw;
    return max_pool2d_attrs(p);
}

static Attrs up_attrs(int64_t sh, int64_t sw) {
    UpsampleParams p;
    p.scale_h = sh; p.scale_w = sw;
    return upsample_nearest2d_attrs(p);
}

TEST(infer_silu) {
    EXPECT(infer_output(OpKind::Silu, {T({2, 3})}, Attrs()).value() == T({2, 3}));
    EXPECT(infer_output(OpKind::Silu, {T({4}, DType::F16)}, Attrs()).value() == T({4}, DType::F16));
    EXPECT(infer_code(OpKind::Silu, {T({4}, DType::I32)}) == Code::InvalidArgument);
    EXPECT(infer_code(OpKind::Silu, {T({4}), T({4})}) == Code::InvalidArgument);
}

TEST(max_pool2d_params_and_sizes) {
    auto back = max_pool2d_params(pool_attrs(5, 3, 2, 1, 2, 1));
    EXPECT(back.is_ok());
    EXPECT(back.value().kernel_h == 5 && back.value().kernel_w == 3 && back.value().stride_h == 2 && back.value().stride_w == 1 && back.value().pad_h == 2 && back.value().pad_w == 1);
    EXPECT(max_pool2d_params(pool_attrs(0, 3, 1, 1, 0, 0)).status().code() == Code::InvalidArgument);
    EXPECT(max_pool2d_params(pool_attrs(3, 3, 0, 1, 0, 0)).status().code() == Code::InvalidArgument);
    EXPECT(max_pool2d_params(pool_attrs(3, 3, 1, 1, 2, 0)).status().code() == Code::InvalidArgument);  // pad > kernel / 2
    EXPECT(max_pool2d_params(pool_attrs(3, 3, 1, 1, -1, 0)).status().code() == Code::InvalidArgument);
    // SPPF pooling keeps the size: k5 s1 p2
    EXPECT(infer_output(OpKind::MaxPool2d, {T({1, 8, 16, 12})}, pool_attrs(5, 5, 1, 1, 2, 2)).value() == T({1, 8, 16, 12}));
    EXPECT(infer_output(OpKind::MaxPool2d, {T({2, 3, 7, 9})}, pool_attrs(2, 3, 2, 3, 0, 0)).value() == T({2, 3, 3, 3}));
    EXPECT(infer_code(OpKind::MaxPool2d, {T({1, 1, 2, 2})}, pool_attrs(5, 5, 1, 1, 0, 0)) == Code::InvalidArgument);  // kernel larger than the input
    EXPECT(infer_code(OpKind::MaxPool2d, {T({8, 8})}, pool_attrs(2, 2, 2, 2, 0, 0)) == Code::InvalidArgument);        // rank
    EXPECT(infer_code(OpKind::MaxPool2d, {T({1, 1, 8, 8}, DType::I32)}, pool_attrs(2, 2, 2, 2, 0, 0)) == Code::InvalidArgument);
    EXPECT(infer_code(OpKind::MaxPool2d, {T({1, 1, 8, 8})}, Attrs()) == Code::InvalidArgument);
}

TEST(upsample_nearest2d_params_and_sizes) {
    EXPECT(upsample_nearest2d_params(up_attrs(2, 3)).value().scale_w == 3);
    EXPECT(upsample_nearest2d_params(up_attrs(0, 1)).status().code() == Code::InvalidArgument);
    EXPECT(infer_output(OpKind::UpsampleNearest2d, {T({1, 4, 8, 6})}, up_attrs(2, 2)).value() == T({1, 4, 16, 12}));
    EXPECT(infer_output(OpKind::UpsampleNearest2d, {T({1, 4, 8, 6})}, up_attrs(3, 1)).value() == T({1, 4, 24, 6}));
    EXPECT(infer_code(OpKind::UpsampleNearest2d, {T({4, 8, 6})}, up_attrs(2, 2)) == Code::InvalidArgument);
    EXPECT(infer_code(OpKind::UpsampleNearest2d, {T({1, 4, 8, 6})}, Attrs()) == Code::InvalidArgument);
    const int64_t big = std::numeric_limits<int64_t>::max() / 2 + 1;
    EXPECT(infer_code(OpKind::UpsampleNearest2d, {T({1, 1, 4, 4})}, up_attrs(big, 1)) == Code::ResourceExhausted);
}

TEST(graph_helpers_for_the_detector_ops) {
    Graph g;
    auto x = g.add_input("x", S({1, 8, 16, 12}), DType::F32).value();
    MaxPool2dParams mp;
    mp.kernel_h = mp.kernel_w = 5;
    mp.pad_h = mp.pad_w = 2;
    UpsampleParams up;
    up.scale_h = up.scale_w = 2;
    auto a = g.add_silu(x);
    auto b = g.add_max_pool2d(a.value(), mp);
    auto c = g.add_upsample_nearest2d(b.value(), up);
    EXPECT(a.is_ok() && b.is_ok() && c.is_ok());
    EXPECT(g.values()[c.value()].type == T({1, 8, 32, 24}));
    EXPECT(g.mark_output("y", c.value()).is_ok());
    EXPECT(g.validate().is_ok());
    EXPECT(g.dump().find("silu(%0)") != std::string::npos);
    EXPECT(g.dump().find("max_pool2d(%1) [5, 5, 1, 1, 2, 2]") != std::string::npos);
    EXPECT(g.dump().find("upsample_nearest2d(%2) [2, 2]") != std::string::npos);
}

TEST(infer_rejects_attributes_the_op_does_not_use) {
    Attrs a;
    a.axis = 1;
    EXPECT(infer_code(OpKind::Gelu, {T({4, 5})}, a) == Code::InvalidArgument);
    Attrs b;
    b.transpose_b = true;
    EXPECT(infer_code(OpKind::Add, {T({4}), T({4})}, b) == Code::InvalidArgument);
    Attrs c;
    c.ints = {1};
    EXPECT(infer_code(OpKind::Softmax, {T({4})}, c) == Code::InvalidArgument);
    Attrs d;
    d.scalar = 2.f;
    EXPECT(infer_code(OpKind::Gelu, {T({4})}, d) == Code::InvalidArgument);
}

// ---- graph building ----

// A two-layer MLP: y = gelu(x W1^T) W2^T.
static Result<Graph> build_mlp() {
    Graph g;
    auto x = g.add_input("x", S({8, 256}), DType::F32);
    auto w1 = g.add_weight("w1", S({1024, 256}), DType::F32);
    auto w2 = g.add_weight("w2", S({64, 1024}), DType::F32);
    if (!x.is_ok() || !w1.is_ok() || !w2.is_ok()) return internal_error("setup");
    auto h = g.add_matmul(x.value(), w1.value(), true);
    if (!h.is_ok()) return h.status();
    auto a = g.add_gelu(h.value());
    if (!a.is_ok()) return a.status();
    auto y = g.add_matmul(a.value(), w2.value(), true);
    if (!y.is_ok()) return y.status();
    Status st = g.mark_output("y", y.value());
    if (!st.is_ok()) return st;
    return g;
}

TEST(graph_mlp_builds_validates_and_dumps) {
    auto g = build_mlp();
    EXPECT(g.is_ok());
    if (!g.is_ok()) return;
    EXPECT(g.value().validate().is_ok());
    EXPECT_EQ(g.value().nodes().size(), size_t(3));
    EXPECT_EQ(g.value().outputs().size(), size_t(1));
    const Value & y = g.value().values()[g.value().outputs()[0].value];
    EXPECT(y.type == T({8, 64}));
    const std::string expected =
        "%0 = input \"x\" : f32[8, 256]\n"
        "%1 = weight \"w1\" : f32[1024, 256]\n"
        "%2 = weight \"w2\" : f32[64, 1024]\n"
        "%3 = matmul(%0, %1) transpose_b : f32[8, 1024]\n"
        "%4 = gelu(%3) : f32[8, 1024]\n"
        "%5 = matmul(%4, %2) transpose_b : f32[8, 64]\n"
        "output \"y\" = %5\n";
    EXPECT_EQ(g.value().dump(), expected);
}

TEST(graph_attention_scores_path) {
    Graph g;
    auto x = g.add_input("x", S({2, 192, 16, 80}), DType::F32).value();
    auto q = g.add_permute(x, {0, 2, 1, 3});
    EXPECT(q.is_ok());
    auto s = g.add_matmul(q.value(), q.value(), true);
    EXPECT(s.is_ok());
    auto p = g.add_softmax(s.value(), 3);
    EXPECT(p.is_ok());
    EXPECT(g.values()[p.value()].type == T({2, 16, 192, 192}));
    EXPECT(g.mark_output("p", p.value()).is_ok());
    EXPECT(g.validate().is_ok());
}

TEST(graph_vit_front_end_and_qkv_split) {
    // image -> patch embed (conv2d) -> tokens; fused qkv projection -> three slices.
    Graph g;
    auto img = g.add_input("img", S({1, 3, 64, 48}), DType::F32).value();
    auto pw = g.add_weight("patch_w", S({32, 3, 16, 16}), DType::F32).value();
    Conv2dParams p;
    p.stride_h = p.stride_w = 16;
    p.pad_h = p.pad_w = 2;
    auto feat = g.add_conv2d(img, pw, p);
    EXPECT(feat.is_ok());
    EXPECT(g.values()[feat.value()].type == T({1, 32, 4, 3}));
    auto flat = g.add_reshape(feat.value(), {1, 32, 12}).value();
    auto tok = g.add_permute(flat, {0, 2, 1}).value();            // [1, 12, 32]
    auto qkv_w = g.add_weight("qkv_w", S({96, 32}), DType::F32).value();
    auto qkv = g.add_matmul(tok, qkv_w, true).value();            // [1, 12, 96]
    auto q = g.add_slice(qkv, 2, 0, 32);
    auto k = g.add_slice(qkv, 2, 32, 64);
    auto v = g.add_slice(qkv, 2, 64, 96);
    EXPECT(q.is_ok() && k.is_ok() && v.is_ok());
    EXPECT(g.values()[q.value()].type == T({1, 12, 32}));
    auto joined = g.add_concat({q.value(), k.value(), v.value()}, 2);
    EXPECT(joined.is_ok());
    EXPECT(g.values()[joined.value()].type == T({1, 12, 96}));
    EXPECT(g.mark_output("out", joined.value()).is_ok());
    EXPECT(g.validate().is_ok());
    EXPECT(g.dump().find("slice(%") != std::string::npos);
    EXPECT(g.dump().find("axis=2") != std::string::npos);
    EXPECT(g.dump().find("conv2d(") != std::string::npos);
}

TEST(graph_new_op_failures_leave_graph_unchanged) {
    Graph g;
    auto x = g.add_input("x", S({4, 5}), DType::F32).value();
    const size_t values_before = g.values().size();
    EXPECT(!g.add_slice(x, 0, 2, 9).is_ok());
    EXPECT(!g.add_concat({x}, 0).is_ok());
    EXPECT(!g.add_concat({x, 99}, 0).is_ok());
    EXPECT(!g.add_conv2d(x, x, Conv2dParams()).is_ok());
    EXPECT_EQ(g.values().size(), values_before);
    EXPECT_EQ(g.nodes().size(), size_t(0));
}

// ---- analytical cost ----

static const Node & last_node(const Graph & g) { return g.nodes().back(); }

TEST(cost_of_matmul_counts_two_flops_per_multiply_add) {
    Graph g;
    auto a = g.add_input("a", S({8, 256}), DType::F32).value();
    auto w = g.add_weight("w", S({1024, 256}), DType::F32).value();
    g.add_matmul(a, w, true).value();
    NodeCost c = node_cost(g, last_node(g));
    EXPECT_EQ(c.flops, 2.0 * 8 * 1024 * 256);
    EXPECT_EQ(c.bytes_read, 4.0 * (8 * 256 + 1024 * 256));
    EXPECT_EQ(c.bytes_written, 4.0 * 8 * 1024);
}

TEST(cost_of_batched_matmul_includes_batch_dims) {
    Graph g;
    auto q = g.add_input("q", S({1, 16, 210, 80}), DType::F32).value();
    g.add_matmul(q, q, true).value();  // [1,16,210,210], K = 80
    EXPECT_EQ(node_cost(g, last_node(g)).flops, 2.0 * (16.0 * 210 * 210) * 80);
}

TEST(cost_of_conv2d) {
    Graph g;
    auto x = g.add_input("x", S({1, 3, 256, 192}), DType::F32).value();
    auto w = g.add_weight("w", S({1280, 3, 16, 16}), DType::F32).value();
    Conv2dParams p;
    p.stride_h = p.stride_w = 16;
    p.pad_h = p.pad_w = 2;
    g.add_conv2d(x, w, p).value();  // output [1,1280,16,12]
    EXPECT_EQ(node_cost(g, last_node(g)).flops, 2.0 * (1280.0 * 16 * 12) * (3 * 16 * 16));
}

TEST(cost_of_memory_bound_ops_has_no_flops) {
    Graph g;
    auto x = g.add_input("x", S({4, 8}), DType::F16).value();
    g.add_gelu(x).value();
    NodeCost c = node_cost(g, last_node(g));
    EXPECT_EQ(c.flops, 0.0);
    EXPECT_EQ(c.bytes_read, 2.0 * 32);
    EXPECT_EQ(c.bytes_written, 2.0 * 32);
}

TEST(cost_of_views_and_copies) {
    Graph g;
    auto x = g.add_input("x", S({2, 6, 4}), DType::F32).value();
    g.add_reshape(x, {2, 3, 2, 4}).value();
    EXPECT_EQ(node_cost(g, last_node(g)).bytes(), 0.0);              // reshape is a view
    g.add_permute(x, {0, 2, 1}).value();
    EXPECT_EQ(node_cost(g, last_node(g)).bytes(), 2.0 * 4 * 48);      // one read + one write of the output
    g.add_slice(x, 1, 0, 3).value();
    EXPECT_EQ(node_cost(g, last_node(g)).bytes_read, 4.0 * 24);       // only the sliced region is read
}

TEST(graph_find_named) {
    auto g = build_mlp().value();
    auto w = g.find_named("w1");
    EXPECT(w.is_ok());
    EXPECT_EQ(w.value(), 1);
    EXPECT(g.find_named("missing").status().code() == Code::NotFound);
    EXPECT(g.find_named("").status().code() == Code::NotFound);
}

TEST(graph_builder_rejects_bad_input) {
    Graph g;
    EXPECT(g.add_input("", S({1}), DType::F32).status().code() == Code::InvalidArgument);
    EXPECT(g.add_input("x", S({2}), DType::F32).is_ok());
    EXPECT(g.add_weight("x", S({2}), DType::F32).status().code() == Code::InvalidArgument);  // duplicate name
    EXPECT(g.add_gelu(99).status().code() == Code::InvalidArgument);                          // unknown id
    EXPECT(g.add_gelu(-1).status().code() == Code::InvalidArgument);
    EXPECT(g.mark_output("o", 99).code() == Code::InvalidArgument);
    EXPECT(g.mark_output("", 0).code() == Code::InvalidArgument);
    EXPECT(g.mark_output("o", 0).is_ok());
    EXPECT(g.mark_output("o", 0).code() == Code::InvalidArgument);                            // duplicate output
    EXPECT_EQ(g.nodes().size(), size_t(0));  // failed adds leave the graph untouched
    EXPECT_EQ(g.values().size(), size_t(1));
}

TEST(graph_failed_node_leaves_graph_unchanged) {
    Graph g;
    auto a = g.add_input("a", S({2, 3}), DType::F32).value();
    auto b = g.add_input("b", S({4, 5}), DType::F32).value();
    const size_t values_before = g.values().size();
    EXPECT(!g.add_matmul(a, b).is_ok());
    EXPECT_EQ(g.values().size(), values_before);
    EXPECT_EQ(g.nodes().size(), size_t(0));
}

TEST(graph_requires_an_output) {
    Graph g;
    g.add_input("x", S({1}), DType::F32);
    EXPECT(g.validate().code() == Code::InvalidArgument);
}

// ---- from_parts: every way a hand-built graph can be wrong ----

namespace {
struct Parts {
    std::vector<Value> values;
    std::vector<Node> nodes;
    std::vector<GraphOutput> outputs;
};

Value named(ValueKind k, const char * name, TypeInfo t) {
    Value v;
    v.kind = k;
    v.name = name;
    v.type = t;
    return v;
}
Value inter(TypeInfo t) {
    Value v;
    v.kind = ValueKind::Intermediate;
    v.type = t;
    return v;
}
Node gelu_node(ValueId in, ValueId out) {
    Node n;
    n.op = OpKind::Gelu;
    n.inputs = {in};
    n.output = out;
    return n;
}
// x -> gelu -> y
Parts good_parts() {
    Parts p;
    p.values = {named(ValueKind::Input, "x", T({4})), inter(T({4}))};
    p.nodes = {gelu_node(0, 1)};
    p.outputs = {{"y", 1}};
    return p;
}
Code build_code(Parts p) {
    auto r = Graph::from_parts(std::move(p.values), std::move(p.nodes), std::move(p.outputs));
    return r.is_ok() ? Code::Ok : r.status().code();
}
}  // namespace

TEST(from_parts_accepts_valid_graph) {
    EXPECT(build_code(good_parts()) == Code::Ok);
}

TEST(from_parts_rejects_wrong_stored_type) {
    Parts p = good_parts();
    p.values[1].type = T({5});
    EXPECT(build_code(p) == Code::InvalidArgument);
    p = good_parts();
    p.values[1].type = T({4}, DType::F16);
    EXPECT(build_code(p) == Code::InvalidArgument);
}

TEST(from_parts_rejects_forward_reference_and_cycles) {
    Parts p = good_parts();
    p.nodes[0].inputs = {1};  // node reads its own output
    EXPECT(build_code(p) == Code::InvalidArgument);
}

TEST(from_parts_rejects_bad_ids) {
    Parts p = good_parts();
    p.nodes[0].inputs = {7};
    EXPECT(build_code(p) == Code::InvalidArgument);
    p = good_parts();
    p.nodes[0].output = 9;
    EXPECT(build_code(p) == Code::InvalidArgument);
    p = good_parts();
    p.outputs[0].value = 9;
    EXPECT(build_code(p) == Code::InvalidArgument);
    p = good_parts();
    p.nodes[0].inputs = {-1};
    EXPECT(build_code(p) == Code::InvalidArgument);
}

TEST(from_parts_rejects_naming_errors) {
    Parts p = good_parts();
    p.values[1].name = "oops";  // intermediate must be unnamed
    EXPECT(build_code(p) == Code::InvalidArgument);
    p = good_parts();
    p.values[0].name = "";
    EXPECT(build_code(p) == Code::InvalidArgument);
    p = good_parts();
    p.values.push_back(named(ValueKind::Weight, "x", T({4})));  // duplicate name
    EXPECT(build_code(p) == Code::InvalidArgument);
    p = good_parts();
    p.outputs.push_back({"y", 1});  // duplicate output name
    EXPECT(build_code(p) == Code::InvalidArgument);
    p = good_parts();
    p.outputs[0].name = "";
    EXPECT(build_code(p) == Code::InvalidArgument);
}

TEST(from_parts_rejects_producer_errors) {
    Parts p = good_parts();
    p.values.push_back(inter(T({4})));  // intermediate with no producer
    EXPECT(build_code(p) == Code::InvalidArgument);
    p = good_parts();
    p.values.push_back(inter(T({4})));
    p.nodes.push_back(gelu_node(0, 1));  // second node writes value 1 again
    EXPECT(build_code(p) == Code::InvalidArgument);
    p = good_parts();
    p.nodes[0].output = 0;  // node writes an input
    EXPECT(build_code(p) == Code::InvalidArgument);
}

TEST(from_parts_rejects_no_outputs_and_bad_arity) {
    Parts p = good_parts();
    p.outputs.clear();
    EXPECT(build_code(p) == Code::InvalidArgument);
    p = good_parts();
    p.nodes[0].inputs = {0, 0};
    EXPECT(build_code(p) == Code::InvalidArgument);
}

TEST(from_parts_error_message_names_the_node) {
    Parts p = good_parts();
    p.values[1].type = T({5});
    auto r = Graph::from_parts(p.values, p.nodes, p.outputs);
    EXPECT(!r.is_ok());
    EXPECT(r.status().message().find("node 0 (gelu)") != std::string::npos);
}

int main() { return dooms_test::run_all(); }
