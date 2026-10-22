#include "ir/cost.h"

namespace dooms {

namespace {

double bytes_of(const TypeInfo & t) {
    return static_cast<double>(t.shape.numel()) / static_cast<double>(dtype_block_elems(t.dtype)) * static_cast<double>(dtype_block_bytes(t.dtype));
}

}  // namespace

NodeCost node_cost(const Graph & g, const Node & n) {
    NodeCost c;
    const TypeInfo & out = g.values()[n.output].type;
    const double out_elems = static_cast<double>(out.shape.numel());
    switch (n.op) {
        case OpKind::MatMul: {
            const TypeInfo & a = g.values()[n.inputs[0]].type;
            c.flops = 2.0 * out_elems * static_cast<double>(a.shape.dim(a.shape.rank() - 1));  // K is a's last dim
            break;
        }
        case OpKind::ConvTranspose2d: {
            // every input pixel is multiplied with the whole [OC, KH, KW] kernel slice of its channel
            const TypeInfo & x = g.values()[n.inputs[0]].type;
            const TypeInfo & w = g.values()[n.inputs[1]].type;   // [IC, OC, KH, KW]
            c.flops = 2.0 * static_cast<double>(x.shape.numel()) * static_cast<double>(w.shape.dim(1) * w.shape.dim(2) * w.shape.dim(3));
            break;
        }
        case OpKind::SelfAttention: {
            // q.k^T and p.v: 2 * N * N * E multiply-adds each, per batch (E = heads * head_dim).
            const double b = static_cast<double>(out.shape.dim(0)), n = static_cast<double>(out.shape.dim(1)), e = static_cast<double>(out.shape.dim(2));
            c.flops = 4.0 * b * n * n * e;
            break;
        }
        case OpKind::Conv2d: {
            const TypeInfo & w = g.values()[n.inputs[1]].type;  // [OC, IC, KH, KW]
            c.flops = 2.0 * out_elems * static_cast<double>(w.shape.dim(1) * w.shape.dim(2) * w.shape.dim(3));
            break;
        }
        default:
            break;
    }
    switch (n.op) {
        case OpKind::Reshape:
            return c;  // a view: no flops, no traffic
        case OpKind::Permute:
        case OpKind::Slice:
        case OpKind::Concat:
            c.bytes_read = bytes_of(out);
            c.bytes_written = bytes_of(out);
            return c;
        default:
            for (ValueId id : n.inputs) c.bytes_read += bytes_of(g.values()[id].type);
            c.bytes_written = bytes_of(out);
            return c;
    }
}

}  // namespace dooms
