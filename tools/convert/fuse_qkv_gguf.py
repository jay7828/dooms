"""Writes a copy of a llama-architecture GGUF whose attention q/k/v projections are one fused `blk.N.attn_qkv.weight`.

llama.cpp's llama architecture loads a fused qkv tensor when the file has one (llama-model.cpp: create_tensor_qkv), so the text decoder
then issues one matrix multiplication instead of three per layer: fewer GPU dispatches per generated token, same arithmetic.
Usage: fuse_qkv_gguf.py IN.gguf OUT.gguf
"""
import sys

import gguf
import numpy as np


def fuse_qkv(src, dst):
    """Writes dst with fused attention projections; returns the number of layers fused."""
    r = gguf.GGUFReader(src)
    arch = bytes(r.fields["general.architecture"].parts[-1]).decode()
    w = gguf.GGUFWriter(dst, arch)
    for name, f in r.fields.items():
        if name.startswith("GGUF.") or name == "general.architecture":
            continue
        t = f.types[0]
        if t == gguf.GGUFValueType.ARRAY:
            vals = f.contents()
            w.add_array(name, vals)
        else:
            w.add_key_value(name, f.contents(), t)
    by_name = {t.name: t for t in r.tensors}
    fused = 0
    for t in r.tensors:
        if ".attn_q." in t.name or ".attn_k." in t.name or ".attn_v." in t.name:
            if not t.name.endswith(".weight"):
                raise SystemExit(f"unsupported: {t.name} (q/k/v biases are not fused here)")
            if ".attn_q." not in t.name:
                continue
            blk = t.name.split(".attn_q.")[0]
            parts = [by_name[f"{blk}.attn_{x}.weight"] for x in "qkv"]
            if len({p.tensor_type for p in parts}) != 1:
                raise SystemExit(f"{blk}: q/k/v have different types")
            data = np.concatenate([np.asarray(p.data) for p in parts], axis=0)
            w.add_tensor(f"{blk}.attn_qkv.weight", data, raw_dtype=parts[0].tensor_type)
            fused += 1
        else:
            w.add_tensor(t.name, np.asarray(t.data), raw_dtype=t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return fused


def main():
    src, dst = sys.argv[1], sys.argv[2]
    print(f"fused q/k/v of {fuse_qkv(src, dst)} layers: {src} -> {dst}")


if __name__ == "__main__":
    main()
