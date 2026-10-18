#!/usr/bin/env python3
"""Reference activations of Kokoro-82M (kokoro 0.9.4, PyTorch fp32 on the CPU) for the engine's tests: <model>/dooms/kokoro-reference.gguf.

    .venv-kokoro/bin/python tools/parity/kokoro_reference.py --model models/kokoro/Kokoro-82M

Per case `caseK.`: the phonemes (misaki), ids, the style vector (voice row len(phonemes) - 1), the random draws the model really made (recorded from torch.rand /
torch.randn_like inside its sine generator), every stage's tensors as the reference modules compute them, and the audio of KModel itself with those draws. Each
stage's tensors come from the reference's own inputs (hooks on KModel), so the engine can be checked one stage at a time. Exit codes: 0 ok, 1 failed, 2 usage.
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "convert"))

import kokoro_model as km  # noqa: E402

CASES = [  # (text, voice): a short phrase, a long sentence with punctuation and numbers, other voices
    ("Hello world.", "af_heart"),
    ("The quick brown fox jumps over the lazy dog. Speech synthesis on Apple Silicon should be fast.", "af_heart"),
    ("In 1969, two astronauts walked on the Moon, while a third orbited above them, waiting.", "am_michael"),
    ("Please call Stella; ask her to bring these things with her from the store.", "bf_emma"),
]


def record_draws(model, ids, ref_s):
    """KModel.forward_with_tokens with the sine generator's random draws recorded; also the stage tensors via hooks."""
    rec, cap = {}, {}
    g = model.decoder.generator
    hooks = [model.decoder.register_forward_hook(lambda m, i, o: cap.update(dec_in=[t.detach().clone() for t in i])),
             g.m_source.register_forward_hook(lambda m, i, o: cap.update(source=o[0].detach().clone())),
             g.conv_post.register_forward_hook(lambda m, i, o: cap.update(post=o.detach().clone()))]
    orig_transform = g.stft.transform

    def transform(x):
        o = orig_transform(x)
        cap["har"] = torch.cat([o[0], o[1]], 1).detach().clone()
        return o
    g.stft.transform = transform
    orig_rand, orig_randn_like = torch.rand, torch.randn_like

    def rand(*a, **k):
        r = orig_rand(*a, **k)
        rec.setdefault("rand_ini", r.clone())
        return r

    def randn_like(x, *a, **k):
        r = orig_randn_like(x, *a, **k)
        rec.setdefault("noise", r.clone())
        return r
    torch.rand, torch.randn_like = rand, randn_like
    try:
        with torch.no_grad():
            audio, dur = model.forward_with_tokens(torch.LongTensor([ids]), ref_s, 1.0)
    finally:
        torch.rand, torch.randn_like = orig_rand, orig_randn_like
        g.stft.transform = orig_transform
        for h in hooks:
            h.remove()
    return audio.numpy(), dur.numpy(), rec, cap


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None, help="default: <model>/dooms/kokoro-reference.gguf")
    ap.add_argument("--cases", type=int, default=len(CASES))
    a = ap.parse_args(argv)
    import gguf  # noqa: PLC0415
    from kokoro import KPipeline  # noqa: PLC0415
    model, cfg = km.load(a.model)
    g2p = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=False)
    g2p_b = KPipeline(lang_code="b", repo_id="hexgrad/Kokoro-82M", model=False)
    P, Fn, V = km.Prosody(model).eval(), km.F0N(model).eval(), km.Vocoder(model).eval()
    c = km.generator_constants(model)
    lin = model.decoder.generator.m_source.l_linear
    lw, lb = lin.weight.detach().numpy()[0], float(lin.bias.detach().numpy()[0])
    out = a.out or os.path.join(a.model, "dooms", "kokoro-reference.gguf")
    w = gguf.GGUFWriter(out + ".tmp", "kokoro_reference")
    n_cases = 0
    for k, (text, voice_name) in enumerate(CASES[:a.cases]):
        pipe = g2p_b if voice_name.startswith("b") else g2p          # British voices: misaki's British English
        voice = torch.load(os.path.join(a.model, "voices", voice_name + ".pt"), weights_only=True)
        chunks = [r.phonemes for r in pipe(text, voice=voice)]       # model=False: phonemes only
        assert len(chunks) == 1, f"case {k} splits into {len(chunks)} chunks"
        phonemes = chunks[0]
        ids = [0] + [model.vocab[p] for p in phonemes if p in model.vocab] + [0]
        ref_s = voice[len(phonemes) - 1]
        audio, frames_ref, rec, cap = record_draws(model, ids, ref_s)
        asr, f0, n, s_dec = cap["dec_in"]
        with torch.no_grad():
            dur, d, t_en = P(torch.tensor([ids], dtype=torch.int32), ref_s[:, 128:])
            x = V(asr, f0, n, s_dec, cap["har"])
        frames, en, asr_np = km.align(dur[0].numpy(), d[0].numpy(), t_en[0].numpy())
        assert np.array_equal(frames, frames_ref), "the split path's durations differ from KModel's"
        assert np.allclose(asr_np, asr[0].numpy(), atol=1e-6), "the expansion differs from KModel's alignment"
        split_audio, _ = km.synthesize_split(model, np.array(ids), ref_s.numpy(), 1.0, rec["rand_ini"][0].numpy(), rec["noise"][0].numpy())
        err = float(np.linalg.norm(split_audio - audio) / np.linalg.norm(audio))
        p = f"case{k}."
        w.add_string(p + "text", text)
        w.add_string(p + "phonemes", phonemes)
        w.add_string(p + "voice", voice_name)
        w.add_array(p + "ids", ids)
        tensors = {"ref_s": ref_s.numpy()[0], "rand_ini": rec["rand_ini"].numpy()[0], "noise": rec["noise"].numpy()[0], "durations": dur.numpy()[0],
                   "frames": frames.astype(np.float32), "d": d.numpy()[0], "t_en": t_en.numpy()[0], "en": en, "asr": asr.numpy()[0], "f0": f0.numpy()[0],
                   "n": n.numpy()[0], "source": cap["source"].numpy()[0, :, 0], "har": cap["har"].numpy()[0], "x": x.numpy()[0], "audio": audio}
        for name, t in tensors.items():
            w.add_tensor(p + name, np.ascontiguousarray(t, dtype=np.float32))
        print(f"case {k}: {len(ids)} ids, {int(frames.sum())} frames ({audio.shape[0] / km.SAMPLE_RATE:.2f} s), voice {voice_name}; split path vs KModel {err:.1e}")
        n_cases += 1
    w.add_uint32("kokoro_reference.n_cases", n_cases)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    os.replace(out + ".tmp", out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
