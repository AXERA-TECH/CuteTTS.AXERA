#!/usr/bin/env python3
"""Reference (torch) implementation of the CuteTTS-distill NPU pipeline.

Mirrors exactly what the board runtime does with the exported static graphs:
  vae_enc -> speaker_enc -> prompt embeds (locenc + proj + table)
  -> lm prefill -> [dit head 4-step -> vae decode | lm decode + locenc step] x N
Used to (a) validate the split against the original model and (b) dump
calibration tensors for Pulsar2.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.environ.get("CUTETTS_SRC", "/ai_data/AX615/hy/workspace/TTS_Train/TTS_quant/CuteTTS/src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import export_onnx as E  # noqa: E402
from models import B, LOCENC_PROMPT_N, SPK_T, TP, VAE_ENC_T, DiTHead, LMDecode, LMPrefill, LocEncProj, VAEDecoder, VAEEncoder, rope_tables  # noqa: E402

ROOT = E.ROOT
REF_WAV = "/ai_data/AX615/hy/workspace/TTS_Train/TTS_quant/CuteTTS/assets/default_reference.wav"


def build_prompt(model, proc, spk, wav24, text, voice_clone=True):
    from cutetts.inference.conditioning import (
        build_guidance_plan, build_prefix_segment, dit_speaker_for_plan, lm_speaker_for_branch,
    )
    if voice_clone:
        with torch.no_grad():
            ref_lat = proc.acoustic_feature_forward(wav24).float()
            spk_emb = spk(speaker_wave(wav24), 16000)["embedding"].float()
    else:
        ref_lat, spk_emb = None, None
    plan = build_guidance_plan("voice_clone" if voice_clone else "tts", "nocfg", 0.0)
    prefix = build_prefix_segment(proc, plan.conditional, target_text=text,
                                  reference_features=None if ref_lat is None else ref_lat[0])
    with torch.no_grad():
        embeds, _, _ = model.prepare_input_embeds(
            prefix, lm_speaker_embedding=lm_speaker_for_branch(plan.conditional, spk_emb))
        dit_spk = dit_speaker_for_plan(plan, spk_emb)
    return embeds, dit_spk, ref_lat


def speaker_wave(wav24):
    from scipy.signal import resample_poly
    w = wav24[0, 0].numpy()
    y = resample_poly(w, 16000, 24000)
    y = y[: 8 * 16000]
    if len(y) < 32000:
        y = np.tile(y, int(np.ceil(32000 / len(y))))
    return torch.from_numpy(y.astype(np.float32))[None]


class RefPipe:
    def __init__(self, cfg, model, vae, spk):
        self.cfg = cfg
        self.model = model
        self.vae = vae
        self.spk = spk
        self.pre = LMPrefill(model.qwen_backbone, model.stop_predictor)
        self.dec = LMDecode(model.qwen_backbone, model.stop_predictor)
        self.loc = LocEncProj(model)
        self.dit = DiTHead(model.head)
        self.vae_dec = VAEDecoder(vae)
        self.vae_enc = VAEEncoder(vae)
        self.scale = float(model.speech_scaling_factor)
        self.bias = float(model.speech_bias_factor)

    @torch.no_grad()
    def run(self, embeds, dit_spk, max_new=256, cfg_strength=2.0, seed=42, calib_dir=None, tag=""):
        L = int(embeds.shape[1])
        assert L <= TP
        e = torch.zeros(1, TP, 1024)
        e[:, :L] = embeds
        outs = self.pre(e)
        hidden = outs[0][:, L - 1, :]
        stop_logits = outs[1]
        kvs = []
        for i in range(7):
            kk = torch.zeros(1, 8, B, 128)
            vv = torch.zeros(1, 8, B, 128)
            kk[:, :, :L] = outs[2 + 2 * i][:, :, :L]
            vv[:, :, :L] = outs[3 + 2 * i][:, :, :L]
            kvs += [kk, vv]
        mask = torch.full((1, 1, 1, B + 1), -30.0)
        mask[..., :L] = 0.0

        torch.manual_seed(seed)
        # the upstream pipeline draws one randn(1) for the VAE posterior std before
        # the first diffusion sample; consume it so x0 matches the reference run
        torch.randn(1)
        cfg_t = torch.tensor([cfg_strength], dtype=torch.float32)
        latents = []
        cond = torch.zeros(1, 2, 64)
        pos = L
        calib_steps = {0, 1, 2, 4, 8, 16} if calib_dir else set()
        for step in range(max_new):
            stop_now = int(torch.argmax(stop_logits, dim=1)) == 1
            x0 = torch.randn(1, 2, 64)
            pred = self.dit(x0, hidden, cond, dit_spk, cfg_t)
            latents.append(pred)
            if calib_dir and step in calib_steps:
                self._dump_dit(calib_dir, tag, step, x0, hidden, cond, dit_spk, cfg_t, pred)
            if stop_now:
                break
            cond = pred
            emb = self.loc(pred[:, None, :, :])
            if pos >= B - 1:
                break
            mask[..., pos] = 0.0
            mask[..., B] = 0.0
            cos_, sin_ = rope_tables(torch.tensor([pos]), 128, 1e6)
            cos_ = cos_.view(1, 1, 1, 128)
            sin_ = sin_.view(1, 1, 1, 128)
            if calib_dir and step in calib_steps:
                self._dump_lm(calib_dir, tag, step, emb, mask, cos_, sin_, kvs)
            d = self.dec(emb, mask, cos_, sin_, *kvs)
            hidden, stop_logits = d[0][:, 0, :], d[1]
            for i in range(7):
                kvs[2 * i][:, :, pos] = d[2 + 2 * i][:, :, 0]
                kvs[2 * i + 1][:, :, pos] = d[2 + 2 * i + 1][:, :, 0]
            pos += 1
        lat = torch.cat(latents, dim=1)  # [1,2N,64]
        vae_lat = lat / self.scale - self.bias
        vae_lat = vae_lat.transpose(1, 2)  # [1,64,2N]
        n = vae_lat.shape[-1]
        bucket = 64 if n <= 64 else 128 if n <= 128 else 256
        padded = torch.zeros(1, 64, bucket)
        padded[:, :, :n] = vae_lat
        wav = self.vae_dec(padded)[0, 0, : n * 1920]
        return wav, lat, bucket, padded, n

    def _dump_lm(self, d, tag, step, embed, mask, cos, sin, kvs):
        os.makedirs(f"{d}/lm_decode", exist_ok=True)
        np.save(f"{d}/lm_decode/embed_{tag}_{step}.npy", embed.numpy())
        np.save(f"{d}/lm_decode/mask_{tag}_{step}.npy", mask.numpy())
        np.save(f"{d}/lm_decode/cos_{tag}_{step}.npy", cos.numpy())
        np.save(f"{d}/lm_decode/sin_{tag}_{step}.npy", sin.numpy())
        for i in range(7):
            np.save(f"{d}/lm_decode/pk{i}_{tag}_{step}.npy", kvs[2 * i].numpy())
            np.save(f"{d}/lm_decode/pv{i}_{tag}_{step}.npy", kvs[2 * i + 1].numpy())

    def _dump_dit(self, d, tag, step, x0, z, cond, spk, cfg, out):
        os.makedirs(f"{d}/dit", exist_ok=True)
        np.save(f"{d}/dit/x0_{tag}_{step}.npy", x0.numpy())
        np.save(f"{d}/dit/z_{tag}_{step}.npy", z.numpy())
        np.save(f"{d}/dit/cond_{tag}_{step}.npy", cond.numpy())
        np.save(f"{d}/dit/speaker_{tag}_{step}.npy", spk.numpy() if spk is not None else np.zeros((1, 256), np.float32))
        np.save(f"{d}/dit/cfg_strength_{tag}_{step}.npy", cfg.numpy())
        np.save(f"{d}/dit/out_{tag}_{step}.npy", out.numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="")
    ap.add_argument("--text", default="The voice generated by this model sounds amazing!")
    ap.add_argument("--mode", default="voice_clone")
    ap.add_argument("--out", default="scratch/refpipe.wav")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--no-hf-check", action="store_true")
    a = ap.parse_args()

    cfg, model, vae, spk = E.load_all()
    import models as _m
    E.dh.F.scaled_dot_product_attention = _m.manual_sdpa
    from cutetts.modeling.processor import CuteTTSProcessor
    from cutetts.modeling.segments import SegmentManagerConfig
    seg = SegmentManagerConfig(**cfg["processor"]["segment"])
    proc = CuteTTSProcessor(
        acoustic_vae_path=os.path.join(E.ORIGIN, "weights", "audio_vae"),
        tokenizer=os.path.join(E.ORIGIN, "tokenizer"), segment_cfg=seg,
        speech_compress_rate=int(cfg["processor"]["speech_compress_rate"]),
        feature_extractor_internal_batchsize=None, feature_extractor_sort_batches=True,
        sequence_style="auto_reference", enable_condition_drop=False,
        text_suffix_token="<|endofprompt|>", acoustic_feature_no_sampling=True,
        packing_enabled=False, speaker_embedding_enabled=True,
        speaker_embedding_source="precomputed", speaker_embedding_dim=256,
        lm_speaker_linear_enabled=True, text_only_prompt_style="instruction",
        reset_previous_cond_at_target_start=True, self_reference_mode="combined",
    ).float().eval()

    wav, sr = sf.read(REF_WAV, dtype="float32", always_2d=True)
    wav24 = torch.from_numpy(wav.T)[None]
    pipe = RefPipe(cfg, model, vae, spk)
    embeds, dit_spk, ref_lat = build_prompt(model, proc, spk, wav24, a.text, a.mode == "voice_clone")
    print("prompt len", embeds.shape[1])
    w, lat, bucket, padded, n = pipe.run(embeds, dit_spk, a.max_new, calib_dir=a.calib or None, tag="u")
    sf.write(a.out, w.numpy(), 24000)
    print("audio", len(w) / 24000, "s, latents", n, "bucket", bucket)

    if not a.no_hf_check:
        from cutetts import CuteTTS
        m = CuteTTS.from_pretrained(f"{E.ROOT}/origin", device="cpu")
        ref = m.generate(a.text, mode=a.mode, diffusion_steps=4, cfg_strength=2.0, seed=42,
                         show_progress=False,
                         reference_audio=REF_WAV if a.mode == "voice_clone" else None)
        r = ref.waveform.squeeze(0).numpy()
        sf.write(a.out.replace(".wav", "_orig.wav"), r, 24000)
        mlen = min(len(r), len(w))
        a_, b_ = r[:mlen], w.numpy()[:mlen]
        cs = float(a_ @ b_ / (np.linalg.norm(a_) * np.linalg.norm(b_) + 1e-12))
        print(f"cosine vs original: {cs:.6f}  len {mlen/24000:.2f}s (orig {len(r)/24000:.2f}s)")


if __name__ == "__main__":
    main()
