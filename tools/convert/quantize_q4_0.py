#!/usr/bin/env python3
"""Quantizes an f16/f32 GGUF to pure Q4_0 (every matrix, the embeddings too; 1-D tensors such as norms and biases stay as they are) with gguf-py, the same
result as llama.cpp's `llama-quantize --pure IN OUT Q4_0` (ggml's reference Q4_0: absmax / -8 scale per 32 values, round to nearest).

    python3 tools/convert/quantize_q4_0.py --in qwen2.5-0.5b-instruct-fp16.gguf --out qwen2.5-0.5b-instruct-q4_0-pure.gguf

Used by scripts/reproduce-4bit.sh for the 4-bit comparison at MLX-LM's model size. Exit codes: 0 ok, 1 failed, 2 usage.
"""
import argparse
import os
import sys

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        import gguf  # noqa: PLC0415
        from gguf import GGMLQuantizationType as Q  # noqa: PLC0415
    except ImportError:
        print("error: needs the gguf package (pip install gguf numpy)", file=sys.stderr)
        return 2
    if not os.path.isfile(a.src):
        print(f"error: {a.src} is not a file", file=sys.stderr)
        return 2
    r = gguf.GGUFReader(a.src)
    arch = str(r.fields["general.architecture"].contents())
    w = gguf.GGUFWriter(a.out + ".tmp", arch)
    for name, f in r.fields.items():
        if name.startswith("GGUF.") or name in ("general.architecture", "general.file_type"):
            continue
        vtype = f.types[0]
        if vtype == gguf.GGUFValueType.ARRAY:
            w.add_key_value(name, f.contents(), vtype, sub_type=f.types[-1])
        else:
            w.add_key_value(name, f.contents(), vtype)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_Q4_0)
    for t in r.tensors:
        shape = [int(x) for x in reversed(t.shape)]   # row-major
        if t.tensor_type not in (Q.F16, Q.F32):
            print(f"error: tensor {t.name} is {t.tensor_type.name}; expected an f16 or f32 file", file=sys.stderr)
            return 1
        data = np.asarray(t.data).reshape(shape)
        if len(shape) >= 2 and shape[-1] % 32 == 0:
            q = gguf.quants.quantize(data.astype(np.float32), Q.Q4_0)
            w.add_tensor(t.name, q, raw_shape=q.shape, raw_dtype=Q.Q4_0)
        else:
            w.add_tensor(t.name, data, raw_dtype=t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    os.replace(a.out + ".tmp", a.out)
    print(f"{a.out}: {os.path.getsize(a.out) / 1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
