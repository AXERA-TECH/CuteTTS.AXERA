#!/usr/bin/env python3
"""AX650 board-side per-submodel accuracy check (pyaxengine, no simulation).

Runs every axmodel on real recorded inputs, compares against the fp32 ONNX
outputs saved on the host, and prints cosine / SNR.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cutetts_ax.tts import AxEngine  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EV = os.path.join(ROOT, "export", "eval_io")
MODELS = os.environ.get("CUTETTS_MODELS", os.path.join(ROOT, "compile"))
if not os.path.isdir(MODELS):
    MODELS = os.path.join(ROOT, "models")
ROWS = []


def metrics(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    snr = float(10 * np.log10(np.sum(a * a) / (np.sum((a - b) ** 2) + 1e-12)))
    return cos, snr


def run(name, feed, refs, tags, sel=None):
    path = os.path.join(MODELS, name + ".axmodel")
    if not os.path.exists(path):
        print(f"== {name}: not built, skip")
        return None
    m = AxEngine(path)
    outs = m.run(feed)
    if sel is not None:
        refs = [sel(r) for r in refs]
        outs = [sel(o) for o in outs]
    for tag, ref, got in zip(tags, refs, outs):
        cos, snr = metrics(ref, got)
        ROWS.append((tag, cos, snr))
        print(f"  {tag:22s} cosine {cos:.6f}  SNR {snr:6.2f} dB")
    return outs


def main():
    L = lambda p: np.load(os.path.join(EV, p))

    wv = L("vae_enc_wav.npy")
    nz = np.nonzero(np.abs(wv).reshape(-1) > 1e-6)[0]
    nreal = max(int((nz[-1] + 1) // 1920), 1) if len(nz) else 1
    run("vae_enc", {"wav": wv}, [L("vae_enc_latent.npy")[:, :nreal]], ["vae_enc"],
        sel=lambda a: np.asarray(a)[:, :nreal])
    outs = run("spk_body", {"logmel": L("spk_logmel.npy")}, [L("spk_body_stats.npy")], ["spk_body stats"])
    if outs is not None:
        A = "/ai_data/AX615/hy/workspace/TTS_Train/TTS_quant/quant/CuteTTS-distill/sdk/assets"
        A = os.path.join(ROOT, "sdk", "assets")
        mean, var, w, b, lw, lb = (np.load(os.path.join(A, f)).astype(np.float64) for f in (
            "spk_bn_mean.npy", "spk_bn_var.npy", "spk_bn_w.npy", "spk_bn_b.npy",
            "spk_linear_w.npy", "spk_linear_b.npy"))
        y = (outs[0][0].astype(np.float64) - mean) / np.sqrt(var + 1e-5) * w + b
        emb = y @ lw.T + lb
        emb /= np.linalg.norm(emb)
        cos, snr = metrics(L("spk_embedding.npy")[0].astype(np.float64), emb)
        ROWS.append(("speaker embedding", cos, snr))
        print(f"  {'speaker embedding':22s} cosine {cos:.6f}  SNR {snr:6.2f} dB")

    run("locenc_step", {"patches": L("locenc_patches.npy")[:, :1]},
        [L("locenc_out.npy")[:, :1]], ["locenc_step"])

    # dit: 4 host-side steps vs fp32 references
    for k in range(4):
        path = os.path.join(MODELS, "dit_step.axmodel")
        if not os.path.exists(path):
            print("== dit_step: not built, skip")
            break
        if k == 0:
            eng = AxEngine(path)
            refs = L("dit_step_out.npy")
            x = L("dit_x0.npy"); z = L("dit_z.npy"); cond = L("dit_cond.npy")
            sp = L("dit_speaker.npy"); cfg = L("dit_cfg.npy")
        v = eng.run({"x": x, "z": z, "cond": cond, "speaker": sp, "cfg_strength": cfg,
                     "t": np.array([k * 0.25], np.float32),
                     "dt": np.array([0.25], np.float32)})[0]
        cos, snr = metrics(refs[k], v)
        print(f"  dit_step{str(k):<14s} cosine {cos:.6f}  SNR {snr:6.2f} dB")
        x = x + v * 0.25
        if k == 3:
            cos, snr = metrics(L("dit_out.npy"), x)
            ROWS.append(("dit 4-step latent", cos, snr))
            print(f"  dit 4-step latent      cosine {cos:.6f}  SNR {snr:6.2f} dB")

    plen = int(L("prompt_len.npy")[0]) if os.path.exists(os.path.join(EV, "prompt_len.npy")) else 256
    def cut(a):
        a = np.asarray(a)
        return a[:, :plen] if a.ndim >= 3 else a
    def sel(a):
        a = np.asarray(a)
        return a[:, :plen] if a.ndim == 3 else (a[:, :, :plen] if a.ndim == 4 else a)
    run("lm_prefill", {"embeds": L("lm_prefill_embeds.npy")},
        [L("lm_prefill_hidden.npy"), L("lm_prefill_stop.npy")]
        + [L(f"lm_prefill_{n}.npy") for i in range(7) for n in (f"k{i}", f"v{i}")],
        ["lm_prefill hidden", "lm_prefill stop"]
        + [f"lm_prefill {n}" for i in range(7) for n in (f"k{i}", f"v{i}")], sel=sel)

    feed = {"embed": L("lm_decode_embed.npy"), "mask_add": L("lm_decode_mask.npy"),
            "cos": L("lm_decode_cos.npy"), "sin": L("lm_decode_sin.npy")}
    for i in range(7):
        feed[f"pk{i}"] = L(f"lm_decode_pk{i}.npy")
        feed[f"pv{i}"] = L(f"lm_decode_pv{i}.npy")
    pl = os.path.join(MODELS, "lm_prefill.axmodel")
    if os.path.exists(pl):
        o = AxEngine(pl).run({"embeds": L("lm_prefill_embeds.npy")})[0]
        h = L("lm_prefill_hidden.npy")
        cos, snr = metrics(h[0, plen - 1], o[0, plen - 1])
        ROWS.append(("lm_prefill hidden(last)", cos, snr))
        print(f"  {'lm_prefill hidden(last)':22s} cosine {cos:.6f}  SNR {snr:6.2f} dB")

    run("lm_decode", feed, [L("lm_decode_hidden.npy"), L("lm_decode_stop.npy")],
        ["lm_decode hidden", "lm_decode stop"])

    for lb in (64, 128, 256):
        run(f"vae_dec_l{lb}", {"latent": L(f"vae_dec_latent{lb}.npy")},
            [L(f"vae_dec_wav{lb}.npy")], [f"vae_dec_l{lb}"])

    print("\n| model | cosine | SNR (dB) |")
    print("|---|---|---|")
    for tag, cos, snr in ROWS:
        print(f"| {tag} | {cos:.6f} | {snr:.2f} |")


if __name__ == "__main__":
    main()
