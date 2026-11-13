#!/usr/bin/env python3
"""Export a *stateful* streaming VAE decoder (2 latents in, 3840 samples out).

Every causal conv keeps its left context in an explicit state tensor, so the board can
decode one latent patch (2 frames = 160 ms) at a time with O(1) work and ~0 latency.
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/ai_data/AX615/hy/workspace/TTS_Train/TTS_quant/CuteTTS/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import export_onnx as E  # noqa: E402
from cutetts.audio_codec.model.audio_vae import CausalConv1d, CausalTransposeConv1d  # noqa: E402

PATCH = 2
HOP = 1920


def collect_convs(module):
    out = []
    for m in module.modules():
        if isinstance(m, CausalConv1d):
            k = m.weight.shape[-1]          # may be zero-stuffed (longer than kernel_size)
            pad = (k - 1) * m.dilation[0]
            out.append((m, "conv", m.in_channels, pad))
        elif isinstance(m, CausalTransposeConv1d):
            k = m.weight.shape[-1]
            ctx = (k - 1) // m.stride[0]
            out.append((m, "tconv", m.in_channels, ctx))
    return out


class StreamDecoder(nn.Module):
    def __init__(self, dec, convs):
        super().__init__()
        self.dec = dec
        self.specs = convs                  # [(module, kind, channels, ctx)]
        self.state_shapes = []
        for m, kind, ch, ctx in convs:
            if ctx > 0:
                self.state_shapes.append((1, ch, ctx))

    def forward(self, latent, *states):
        it = iter(states)
        out = []

        def apply(mod, x):
            if isinstance(mod, CausalConv1d):
                pad = (mod.weight.shape[-1] - 1) * mod.dilation[0]
                if pad == 0:
                    return F.conv1d(x, mod.weight, mod.bias, stride=mod.stride,
                                    dilation=mod.dilation, groups=mod.groups)
                st = next(it)
                xin = torch.cat([st, x], dim=-1)
                y = F.conv1d(xin, mod.weight, mod.bias, stride=mod.stride,
                             dilation=mod.dilation, groups=mod.groups)
                out.append(xin[..., -pad:])
                return y
            if isinstance(mod, CausalTransposeConv1d):
                s = mod.stride[0]
                ctx = (mod.weight.shape[-1] - 1) // s
                trim = 2 * ((s + 1) // 2) - (s % 2)
                st = next(it)
                full = torch.cat([st, x], dim=-1)
                out.append(full[..., -ctx:])
                y = F.conv_transpose1d(full, mod.weight, mod.bias, stride=s)
                return y[..., ctx * s: y.shape[-1] - trim]
            if isinstance(mod, nn.Sequential):
                for m in mod:
                    x = apply(m, x)
                return x
            name = mod.__class__.__name__
            if name == "CausalResidualUnit":
                return x + apply(mod.block, x)
            if name == "CausalDecoderBlock":
                return apply(mod.block, x)
            return mod(x)

        wav = apply(self.dec.model, latent)
        return [wav] + out


def main():
    cfg, model, vae, spk = E.load_all()
    dec = vae.decoder
    convs = collect_convs(dec)
    sd = StreamDecoder(dec, convs)
    print("state tensors:", len(sd.state_shapes), "floats:",
          sum(int(np.prod(s)) for s in sd.state_shapes))

    ev = os.path.join(E.EXPORT, "eval_io")
    lat = torch.from_numpy(np.load(f"{ev}/vae_dec_latent64.npy"))  # [1,64,64]

    # reference: offline decode
    with torch.no_grad():
        ref = dec(lat)  # [1,1,64*1920]

    # streaming decode: 2 latents per step
    states = [torch.zeros(s) for s in sd.state_shapes]
    chunks = []
    with torch.no_grad():
        for i in range(0, lat.shape[-1], PATCH):
            piece = lat[:, :, i:i + PATCH]
            res = sd(piece, *states)
            states = res[1:]
            chunks.append(res[0])
    got = torch.cat(chunks, dim=-1)
    n = min(ref.shape[-1], got.shape[-1])
    a = ref[0, 0, :n].numpy().astype(np.float64)
    b = got[0, 0, :n].numpy().astype(np.float64)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    snr = float(10 * np.log10(np.sum(a * a) / (np.sum((a - b) ** 2) + 1e-12)))
    print(f"stream vs offline: {ref.shape[-1]} vs {got.shape[-1]} samples, cos {cos:.7f}, SNR {snr:.1f} dB")

    # export with 2 latent frames
    lat2 = torch.zeros(1, 64, PATCH)
    names_in = ["latent"] + [f"s{i}" for i in range(len(sd.state_shapes))]
    names_out = ["wav"] + [f"so{i}" for i in range(len(sd.state_shapes))]
    torch.onnx.export(sd, (lat2, *[torch.zeros(s) for s in sd.state_shapes]),
                      f"{E.ONNX}/vae_dec_stream.onnx", input_names=names_in, output_names=names_out,
                      opset_version=17, do_constant_folding=True, dynamo=False)
    E.verify(f"{E.ONNX}/vae_dec_stream.onnx",
             {"latent": lat2, **{f"s{i}": torch.zeros(s) for i, s in enumerate(sd.state_shapes)}},
             [torch.zeros(1, 1, PATCH * HOP)] + [torch.zeros(s) for s in sd.state_shapes],
             names_out)
    json.dump({"states": [list(s) for s in sd.state_shapes]},
              open(os.path.join(E.ROOT, "sdk", "assets", "vae_stream.json"), "w"), indent=1)
    print("exported vae_dec_stream.onnx")


if __name__ == "__main__":
    main()
