#!/usr/bin/env python3
"""Calibration data for the stateful streaming VAE decoder (latent + conv states)."""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/ai_data/AX615/hy/workspace/TTS_Train/TTS_quant/CuteTTS/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import export_onnx as E  # noqa: E402
import export_vae_stream as S  # noqa: E402

ROOT = E.ROOT
OUT = f"{ROOT}/calib/data/vae_dec_stream"


def main():
    cfg, model, vae, spk = E.load_all()
    sd = S.StreamDecoder(vae.decoder, S.collect_convs(vae.decoder))
    shapes = sd.state_shapes

    srcs = [np.load(f"{ROOT}/export/eval_io/vae_dec_latent256.npy")]
    d = f"{ROOT}/calib/data/vae_dec_l256/latent"
    srcs += [np.load(os.path.join(d, f)) for f in sorted(os.listdir(d))[:3]]

    for n in ["latent"] + [f"s{i}" for i in range(len(shapes))]:
        os.makedirs(f"{OUT}/{n}", exist_ok=True)

    k = 0
    for l in srcs:
        lat = torch.from_numpy(np.asarray(l, np.float32))
        states = [torch.zeros(s) for s in shapes]
        with torch.no_grad():
            for i in range(0, min(lat.shape[-1], 128), S.PATCH):
                piece = lat[:, :, i:i + S.PATCH]
                if piece.shape[-1] < S.PATCH:
                    break
                if i % 16 == 0 and k < 32:
                    tag = f"c{k:02d}"
                    k += 1
                    np.save(f"{OUT}/latent/latent_{tag}.npy", piece.numpy())
                    for j, st in enumerate(states):
                        np.save(f"{OUT}/s{j}/s{j}_{tag}.npy", st.numpy())
                states = sd(piece, *states)[1:]
    print("calibration samples:", k)


if __name__ == "__main__":
    main()
