#!/usr/bin/env python3
"""Generate ONNX (fp32) reference outputs for board-side per-model checks."""

import glob
import os
import sys

import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX = os.path.join(ROOT, "export", "onnx")
EV = os.path.join(ROOT, "export", "eval_io")


def sess(name):
    return ort.InferenceSession(os.path.join(ONNX, name + ".onnx"), providers=["CPUExecutionProvider"])


def main():
    os.makedirs(EV, exist_ok=True)
    # per-step dit references (t = 0, 0.25, 0.5, 0.75)
    x = np.load(f"{EV}/dit_x0.npy"); z = np.load(f"{EV}/dit_z.npy")
    cond = np.load(f"{EV}/dit_cond.npy"); sp = np.load(f"{EV}/dit_speaker.npy")
    cfg = np.load(f"{EV}/dit_cfg.npy")
    s = sess("dit_step")
    out = []
    for k in range(4):
        out.append(s.run(None, {"x": x, "z": z, "cond": cond, "speaker": sp, "cfg_strength": cfg,
                                "t": np.array([k * 0.25], np.float32),
                                "dt": np.array([0.25], np.float32)})[0])
    np.save(f"{EV}/dit_step_out.npy", np.stack(out, 0))
    # vae dec 64 bucket
    lat = np.load(f"{EV}/vae_dec_latent256.npy")[:, :, :64]
    np.save(f"{EV}/vae_dec_latent64.npy", lat.astype(np.float32))
    np.save(f"{EV}/vae_dec_wav64.npy", sess("vae_dec_l64").run(None, {"latent": lat})[0])
    # vae dec 128
    lat = np.load(f"{EV}/vae_dec_latent256.npy")[:, :, :128]
    np.save(f"{EV}/vae_dec_latent128.npy", lat.astype(np.float32))
    np.save(f"{EV}/vae_dec_wav128.npy", sess("vae_dec_l128").run(None, {"latent": lat})[0])
    print("done")


if __name__ == "__main__":
    main()
