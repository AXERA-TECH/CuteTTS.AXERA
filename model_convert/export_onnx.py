#!/usr/bin/env python3
"""Export CuteTTS-distill submodels to static-shape ONNX + verify vs torch."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.environ.get("CUTETTS_SRC", "/ai_data/AX615/hy/workspace/TTS_Train/TTS_quant/CuteTTS/src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cutetts.modeling.diffusion_head as dh  # noqa: E402
from cutetts.audio_codec.model.audio_vae import AudioVAE  # noqa: E402
from cutetts.audio_codec.model.speaker_encoder import FbankECAPAStudent  # noqa: E402
from cutetts.modeling.configuration import CuteTTSConfig  # noqa: E402
from cutetts.modeling.model import CuteTTSModel  # noqa: E402
from models import (  # noqa: E402
    B, HEAD_DIM, LOCENC_PROMPT_N, SPK_T, TP, VAE_ENC_T, DiTHead, LMDecode, LMPrefill,
    LocEncProj, SpeakerEncoder, VAEDecoder, VAEEncoder, logmel_numpy,
)
from safetensors.torch import load_file  # noqa: E402

ROOT = os.environ.get("CUTETTS_QUANT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORIGIN = os.path.join(ROOT, "origin")
EXPORT = os.path.join(ROOT, "export")
ONNX = os.path.join(EXPORT, "onnx")
os.makedirs(ONNX, exist_ok=True)

OPSET = 17


def load_all():
    cfg = json.load(open(os.path.join(ORIGIN, "config.json")))
    arch = dict(cfg["architecture"])
    arch["attn_implementation"] = "eager"
    arch["use_pretrained_lm"] = False
    arch["lm_model_name"] = None
    model = CuteTTSModel(CuteTTSConfig(**arch))
    sd = load_file(os.path.join(ORIGIN, "weights", "tts", "model.safetensors"))
    model.load_state_dict(sd, strict=True)

    vcfg = json.load(open(os.path.join(ORIGIN, "weights", "audio_vae", "config.json")))
    vcfg.pop("component")
    vae = AudioVAE(**vcfg)
    vae.load_state_dict(load_file(os.path.join(ORIGIN, "weights", "audio_vae", "model.safetensors")), strict=True)

    scfg = json.load(open(os.path.join(ORIGIN, "weights", "speaker_encoder", "config.json")))
    scfg.pop("component")
    spk = FbankECAPAStudent(**scfg)
    spk.load_state_dict(load_file(os.path.join(ORIGIN, "weights", "speaker_encoder", "model.safetensors")), strict=True)

    from models import patch_rmsnorm
    from models import stuff_dilated_depthwise
    print("stuffed dilated depthwise convs:", stuff_dilated_depthwise(vae))
    npatch = patch_rmsnorm(model) + patch_rmsnorm(vae) + patch_rmsnorm(spk)
    print("patched RMSNorm modules:", npatch)
    model = model.float().eval()
    vae = vae.float().eval()
    spk = spk.float().eval()
    for m in (model, vae, spk):
        for p in m.parameters():
            p.requires_grad_(False)
    return cfg, model, vae, spk


def export(mod, args, names, path, out_names, input_names):
    t0 = time.time()
    mod.eval()
    with torch.no_grad():
        torch.onnx.export(
            mod, args, path, input_names=input_names, output_names=out_names,
            opset_version=OPSET, do_constant_folding=True, dynamo=False,
        )
    print(f"  exported {os.path.basename(path)} {os.path.getsize(path)/1e6:.1f} MB in {time.time()-t0:.1f}s")


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def verify(path, inputs, torch_outs, names, rtol=2e-3):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    feed = {n: t.numpy() for n, t in inputs.items()}
    outs = sess.run(None, feed)
    if not isinstance(torch_outs, (list, tuple)):
        torch_outs = [torch_outs]
    res = {}
    for name, ref, got in zip(names, torch_outs, outs):
        c = cosine(ref.detach().numpy(), got)
        res[name] = c
        flag = "OK " if c > 0.999 else "!! "
        print(f"    {flag}{name}: cosine {c:.7f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--ver", action="store_true", help="also dump i/o samples for board tests")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None

    cfg, model, vae, spk = load_all()
    # SDPA -> explicit matmul/softmax for ONNX/Pulsar2 friendliness (dh.F is torch.nn.functional)
    import models as _m
    dh.F.scaled_dot_product_attention = _m.manual_sdpa
    meta = {
        "variant": cfg["variant"],
        "sample_rate": 24000,
        "speech_compress_rate": cfg["processor"]["speech_compress_rate"],
        "speech_scaling_factor": float(model.speech_scaling_factor),
        "speech_bias_factor": float(model.speech_bias_factor),
        "shapes": {
            "vae_enc_T": VAE_ENC_T, "spk_T": SPK_T, "locenc_prompt_n": LOCENC_PROMPT_N,
            "lm_TP": TP, "lm_B": B,
        },
        "latent_patch": 2,
        "vae_hop": 1920,
        "lm_layers": 7,
    }
    json.dump(meta, open(os.path.join(EXPORT, "model_meta.json"), "w"), indent=2)
    print("scaling:", meta["speech_scaling_factor"], meta["speech_bias_factor"])

    evaldir = os.path.join(EXPORT, "eval_io")
    os.makedirs(evaldir, exist_ok=True)

    # ---------------- real reference data ----------------
    import soundfile as sf
    from scipy.signal import resample_poly
    ref_path = os.environ.get("CUTETTS_REF_WAV", os.path.join(os.path.dirname(os.path.dirname(ROOT)), "CuteTTS", "assets", "default_reference.wav"))
    wav, sr = sf.read(ref_path, dtype="float32", always_2d=True)
    wav = wav.T
    wav24 = torch.from_numpy(wav)[None]  # [1,1,T]
    if sr != 24000:
        wav24 = torch.from_numpy(resample_poly(wav, 24000, sr, axis=-1).astype(np.float32))[None]
    wav16 = torch.from_numpy(resample_poly(wav, 16000, sr, axis=-1).astype(np.float32))[None]

    # ---------------- vae encoder ----------------
    if only is None or "vae_enc" in only:
        print("== vae encoder")
        pad = VAE_ENC_T - wav24.shape[-1]
        w = torch.nn.functional.pad(wav24, (0, pad))
        enc = VAEEncoder(vae)
        with torch.no_grad():
            lat = enc(w)
        print("   latent", tuple(lat.shape))
        export(enc, (w,), None, f"{ONNX}/vae_enc.onnx", ["latent"], ["wav"])
        verify(f"{ONNX}/vae_enc.onnx", {"wav": w}, [lat], ["latent"])
        np.save(f"{evaldir}/vae_enc_wav.npy", w.numpy())
        np.save(f"{evaldir}/vae_enc_latent.npy", lat.numpy())

    # ---------------- speaker encoder ----------------
    if only is None or "spk" in only:
        print("== speaker encoder")
        lm_mod = spk.feature_extract
        with torch.no_grad():
            mel = lm_mod(wav16)[0]  # [80,T]
            mel = mel[None]
        print("   mel", tuple(mel.shape))
        T = mel.shape[-1]
        mel_pad = torch.nn.functional.pad(mel, (0, SPK_T - T))
        se = SpeakerEncoder(spk)
        with torch.no_grad():
            emb = se(mel_pad)
        export(se, (mel_pad,), None, f"{ONNX}/speaker_enc.onnx", ["embedding"], ["logmel"])
        verify(f"{ONNX}/speaker_enc.onnx", {"logmel": mel_pad}, [emb], ["embedding"])
        np.save(f"{evaldir}/spk_wav16.npy", wav16.numpy())
        np.save(f"{evaldir}/spk_logmel.npy", mel_pad.numpy())
        np.save(f"{evaldir}/spk_embedding.npy", emb.numpy())
        np.save(f"{evaldir}/spk_mel_filter.npy", lm_mod.mel_filter.numpy())
        np.save(f"{evaldir}/spk_window.npy", lm_mod.window.numpy())

    # ---------------- prompt embeds (torch reference) ----------------
    from cutetts.modeling.processor import CuteTTSProcessor
    from cutetts.modeling.segments import SegmentManagerConfig
    from cutetts.inference.conditioning import build_guidance_plan, build_prefix_segment, lm_speaker_for_branch, dit_speaker_for_plan
    seg = SegmentManagerConfig(**cfg["processor"]["segment"])
    proc = CuteTTSProcessor(
        acoustic_vae_path=os.path.join(ORIGIN, "weights", "audio_vae"),
        tokenizer=os.path.join(ORIGIN, "tokenizer"),
        segment_cfg=seg,
        speech_compress_rate=int(cfg["processor"]["speech_compress_rate"]),
        feature_extractor_internal_batchsize=None,
        feature_extractor_sort_batches=True,
        sequence_style="auto_reference",
        enable_condition_drop=False,
        text_suffix_token="<|endofprompt|>",
        acoustic_feature_no_sampling=True,
        packing_enabled=False,
        speaker_embedding_enabled=True,
        speaker_embedding_source="precomputed",
        speaker_embedding_dim=256,
        lm_speaker_linear_enabled=True,
        text_only_prompt_style="instruction",
        reset_previous_cond_at_target_start=True,
        self_reference_mode="combined",
    ).float().eval()

    text = "The voice generated by this model sounds amazing!"
    plan = build_guidance_plan("voice_clone", "nocfg", 0.0)
    with torch.no_grad():
        ref_lat = proc.acoustic_feature_forward(wav24)  # [1,L,64]
        ref_lat = ref_lat.float()
        spk_emb = spk(wav16, 16000)["embedding"].float()
        prefix = build_prefix_segment(proc, plan.conditional, target_text=text, reference_features=ref_lat[0])
        embeds, contains_speech, _ = model.prepare_input_embeds(
            prefix, lm_speaker_embedding=lm_speaker_for_branch(plan.conditional, spk_emb))
        dit_spk = dit_speaker_for_plan(plan, spk_emb)
    L = embeds.shape[1]
    print("prompt len", L, "speech latents", tuple(ref_lat.shape))
    assert L <= TP, L
    np.save(f"{evaldir}/prompt_embeds.npy", embeds.numpy())
    np.save(f"{evaldir}/prompt_len.npy", np.array([L], dtype=np.int32))
    np.save(f"{evaldir}/ref_latent.npy", ref_lat.numpy())
    np.save(f"{evaldir}/dit_speaker.npy", dit_spk.numpy())

    # ---------------- locenc ----------------
    if only is None or "locenc" in only:
        print("== locenc")
        le = LocEncProj(model)
        # patch reference latents (pad to even)
        lat = ref_lat[0]
        if lat.shape[0] % 2:
            lat = torch.cat([lat, torch.zeros(1, 64)], 0)
        patches = lat.reshape(1, -1, 2, 64)
        p1 = torch.zeros(1, LOCENC_PROMPT_N, 2, 64)
        p1[:, :patches.shape[1]] = patches
        with torch.no_grad():
            y = le(p1)
        export(le, (p1,), None, f"{ONNX}/locenc_prompt.onnx", ["embeds"], ["patches"])
        verify(f"{ONNX}/locenc_prompt.onnx", {"patches": p1}, [y], ["embeds"])
        one = patches[:, :1]
        with torch.no_grad():
            yo = le(one)
        export(le, (one,), None, f"{ONNX}/locenc_step.onnx", ["embeds"], ["patches"])
        verify(f"{ONNX}/locenc_step.onnx", {"patches": one}, [yo], ["embeds"])
        np.save(f"{evaldir}/locenc_patches.npy", patches.numpy())
        np.save(f"{evaldir}/locenc_out.npy", y.numpy())

    # ---------------- lm prefill ----------------
    if only is None or "lm" in only:
        print("== lm prefill")
        pre = LMPrefill(model.qwen_backbone, model.stop_predictor)
        e = torch.zeros(1, TP, 1024)
        e[:, :L] = embeds
        with torch.no_grad():
            outs = pre(e)
        # ground truth: HF backbone full forward
        with torch.no_grad():
            hf = model.qwen_backbone(inputs_embeds=e, position_ids=torch.arange(TP)[None],
                                     use_cache=False, return_dict=True).last_hidden_state
        print("    HF hidden cosine", cosine(hf.numpy(), outs[0].numpy()))
        names = ["hidden", "stop"] + [x for i in range(7) for x in (f"k{i}", f"v{i}")]
        export(pre, (e,), None, f"{ONNX}/lm_prefill.onnx", names, ["embeds"])
        verify(f"{ONNX}/lm_prefill.onnx", {"embeds": e}, outs, names)
        np.save(f"{evaldir}/lm_prefill_embeds.npy", e.numpy())
        for n, o in zip(names, outs):
            np.save(f"{evaldir}/lm_prefill_{n}.npy", o.detach().numpy())

        # ---------------- lm decode ----------------
        print("== lm decode")
        dec = LMDecode(model.qwen_backbone, model.stop_predictor)
        h = outs[0][:, L - 1, :]
        # pretend next input embed = locenc of zero latent (shape-correct)
        le = LocEncProj(model)
        with torch.no_grad():
            nb = le(torch.zeros(1, 1, 2, 64))
        kvs = []
        mask = torch.full((1, 1, 1, B + 1), -30.0)
        mask[..., :L] = 0.0
        mask[..., -1] = 0.0
        for i in range(7):
            kk = torch.zeros(1, 8, B, 128)
            vv = torch.zeros(1, 8, B, 128)
            kk[:, :, :L] = outs[2 + 2 * i][:, :, :L]
            vv[:, :, :L] = outs[3 + 2 * i][:, :, :L]
            kvs += [kk, vv]
        pos = L
        from models import rope_tables
        cos, sin = rope_tables(torch.tensor([pos]), HEAD_DIM, 1e6)
        cos = cos.view(1, 1, 1, HEAD_DIM)
        sin = sin.view(1, 1, 1, HEAD_DIM)
        with torch.no_grad():
            douts = dec(nb, mask, cos, sin, *kvs)
        dnames = ["hidden", "stop"] + [x for i in range(7) for x in (f"k{i}", f"v{i}")]
        export(dec, (nb, mask, cos, sin, *kvs), None, f"{ONNX}/lm_decode.onnx", dnames,
               ["embed", "mask_add", "cos", "sin"] + [x for i in range(7) for x in (f"pk{i}", f"pv{i}")])
        from transformers.cache_utils import DynamicCache
        hf_cache = DynamicCache()
        for i in range(7):
            hf_cache.update(outs[2 + 2 * i][:, :, :L].contiguous(), outs[3 + 2 * i][:, :, :L].contiguous(), i)
        with torch.no_grad():
            hf_d = model.qwen_backbone(inputs_embeds=nb, attention_mask=None,
                                       position_ids=torch.tensor([[L]]), past_key_values=hf_cache,
                                       use_cache=True, return_dict=True).last_hidden_state
        print("    HF decode hidden cosine", cosine(hf_d.numpy(), douts[0].numpy()))
        verify(f"{ONNX}/lm_decode.onnx", {"embed": nb, "mask_add": mask, "cos": cos, "sin": sin,
                                          **{f"pk{i}": kvs[2 * i] for i in range(7)},
                                          **{f"pv{i}": kvs[2 * i + 1] for i in range(7)}}, douts, dnames)
        np.save(f"{evaldir}/lm_decode_embed.npy", nb.numpy())
        np.save(f"{evaldir}/lm_decode_mask.npy", mask.numpy())
        np.save(f"{evaldir}/lm_decode_cos.npy", cos.numpy())
        np.save(f"{evaldir}/lm_decode_sin.npy", sin.numpy())
        for i in range(7):
            np.save(f"{evaldir}/lm_decode_pk{i}.npy", kvs[2 * i].numpy())
            np.save(f"{evaldir}/lm_decode_pv{i}.npy", kvs[2 * i + 1].numpy())
        for n, o in zip(dnames, douts):
            np.save(f"{evaldir}/lm_decode_{n}.npy", o.detach().numpy())

        # ---------------- dit head ----------------
        print("== dit head")
        dest = DiTHead(model.head)
        g = torch.Generator().manual_seed(42)
        x0 = torch.randn(1, 2, 64, generator=g)
        z = h  # [1,1024]
        cond = torch.zeros(1, 2, 64)
        cfgt = torch.tensor([2.0])
        with torch.no_grad():
            y = dest(x0, z, cond, dit_spk, cfgt)
            # reference through the real sampler
            ref = model.head.sample(z, cfg=0.0, num_sampling_steps=4, cond=cond,
                                    speaker_embedding=dit_spk, distilled_cfg_strength=2.0,
                                    sampling_condition_cache=None)
        print("   ref sample", tuple(ref.shape))
        export(dest, (x0, z, cond, dit_spk, cfgt), None, f"{ONNX}/dit_head.onnx", ["latent"],
               ["x0", "z", "cond", "speaker", "cfg_strength"])
        verify(f"{ONNX}/dit_head.onnx", {"x0": x0, "z": z, "cond": cond,
                                         "speaker": dit_spk, "cfg_strength": cfgt}, [y], ["latent"])
        np.save(f"{evaldir}/dit_x0.npy", x0.numpy())
        np.save(f"{evaldir}/dit_z.npy", z.numpy())
        np.save(f"{evaldir}/dit_cond.npy", cond.numpy())
        np.save(f"{evaldir}/dit_speaker.npy", dit_spk.numpy())
        np.save(f"{evaldir}/dit_cfg.npy", cfgt.numpy())
        np.save(f"{evaldir}/dit_out.npy", y.numpy())

    # ---------------- vae decoder ----------------
    if only is None or "vae_dec" in only:
        print("== vae decoder")
        dec = VAEDecoder(vae)
        for lb in (64, 128, 256):
            lat = torch.zeros(1, 64, lb)
            lat[:, :, :ref_lat.shape[1]] = ref_lat[0].T
            with torch.no_grad():
                w = dec(lat)
            print(f"   L={lb} -> wav {tuple(w.shape)}")
            export(dec, (lat,), None, f"{ONNX}/vae_dec_l{lb}.onnx", ["wav"], ["latent"])
            verify(f"{ONNX}/vae_dec_l{lb}.onnx", {"latent": lat}, [w], ["wav"])
            if lb == 256:
                np.save(f"{evaldir}/vae_dec_latent256.npy", lat.numpy())
                np.save(f"{evaldir}/vae_dec_wav256.npy", w.numpy())

    print("done")


if __name__ == "__main__":
    main()
