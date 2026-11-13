"""ONNX-exportable wrappers for CuteTTS-distill submodels (AX650 target).

Every wrapper is static-shape and uses only ops that map cleanly to AX NPU:
MatMul/Gemm, Conv/ConvTranspose, RMSNorm decomposed, Softmax, SiLU(=Sigmoid*Mul),
Sin (Snake activation), Tanh, etc.

Layout conventions (board side):
  vae encoder   wav [1,1,VAE_ENC_T]            -> latent [1,64,L]
  speaker enc   logmel [1,80,SPK_T]            -> embedding [1,256] (L2 normed)
  locenc        patches [1,N,2,64]             -> lm embeds [1,N,1024]
  lm prefill    embeds [1,TP,1024]             -> hidden [1,TP,1024], stop [1,2], k/v x7
  lm decode     embed [1,1,1024], kv, mask01   -> hidden [1,1,1024], stop [1,2], k_new/v_new x7
  dit head      x0 [1,2,64], z [1,1,1024], cond [1,2,64], spk [1,256], cfg [1]
                                               -> latent [1,2,64]  (4 Euler steps fused)
  vae decoder   latent [1,64,LB]               -> wav [1,1,LB*1920]
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

HEAD_DIM = 128
N_HEADS = 16
N_KV = 8
HIDDEN = 1024
LATENT = 64
PATCH = 2
NUM_LAYERS = 7
VAE_HOP = 1920

# static shapes
VAE_ENC_T = 120000      # 5 s @ 24 kHz reference audio
SPK_T = 800             # 8 s @ 16 kHz -> 800 mel frames (hop 160)
LOCENC_PROMPT_N = 128   # up to 128 prompt patches (256 latents = 20.5 s)
TP = 256                # lm prefill bucket
B = 384                 # lm decode kv capacity


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    var = (x * x).mean(dim=-1, keepdim=True)
    return w * (x / torch.sqrt(var + eps))


def rope_tables(positions: torch.Tensor, head_dim: int, theta: float):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = positions.to(torch.float32)[:, None] * inv[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    # q [1,H,T,D], cos [1,1,T,D]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def manual_sdpa(q, k, v, is_causal: bool = False, **kw):
    """SDPA replacement (no mask): q [B,H,T,D], k/v [B,H,S,D]."""
    scale = q.size(-1) ** -0.5
    if k.size(1) != q.size(1):
        rep = q.size(1) // k.size(1)
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if is_causal:
        t, s = q.size(-2), k.size(-2)
        mask = torch.tril(torch.ones(t, s, dtype=scores.dtype))
        p = torch.softmax(scores, dim=-1) * mask
        p = p / p.sum(dim=-1, keepdim=True)
    else:
        p = torch.softmax(scores, dim=-1)
    return torch.matmul(p, v)


def silu(x):
    return x * torch.sigmoid(x)


class Qwen3LoRALessLayer(nn.Module):
    """One Qwen3 decoder layer, explicit cache-friendly implementation."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.eps = layer.input_layernorm.variance_epsilon

    def forward(self, h, past_k=None, past_v=None, mask_add=None, cos=None, sin=None):
        """h [1,T,1024]; past_k/v [1,8,S,128]; mask01 [1,1,T,S+T]; cos/sin [1,1,T,128]."""
        layer = self.layer
        t = h.shape[1]
        hn = rms_norm(h, layer.input_layernorm.weight, self.eps)
        q = layer.self_attn.q_proj(hn).view(1, t, N_HEADS, HEAD_DIM)
        k = layer.self_attn.k_proj(hn).view(1, t, N_KV, HEAD_DIM)
        v = layer.self_attn.v_proj(hn).view(1, t, N_KV, HEAD_DIM)
        q = rms_norm(q, layer.self_attn.q_norm.weight, self.eps).transpose(1, 2)
        k = rms_norm(k, layer.self_attn.k_norm.weight, self.eps).transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        k_new, v_new = k, v
        if past_k is not None:
            k_all = torch.cat([past_k, k], dim=2)
            v_all = torch.cat([past_v, v], dim=2)
        else:
            k_all, v_all = k, v
        # GQA expand
        k_rep = k_all.reshape(1, N_KV, 1, k_all.shape[2], HEAD_DIM).expand(1, N_KV, N_HEADS // N_KV, k_all.shape[2], HEAD_DIM).reshape(1, N_HEADS, k_all.shape[2], HEAD_DIM)
        v_rep = v_all.reshape(1, N_KV, 1, v_all.shape[2], HEAD_DIM).expand(1, N_KV, N_HEADS // N_KV, v_all.shape[2], HEAD_DIM).reshape(1, N_HEADS, v_all.shape[2], HEAD_DIM)
        scores = torch.matmul(q, k_rep.transpose(2, 3)) * (HEAD_DIM ** -0.5)
        if mask_add is not None:
            scores = scores + mask_add
        p = torch.softmax(scores, dim=-1)
        attn = torch.matmul(p, v_rep).transpose(1, 2).contiguous().reshape(1, t, N_HEADS * HEAD_DIM)
        h = h + layer.self_attn.o_proj(attn)
        hn = rms_norm(h, layer.post_attention_layernorm.weight, self.eps)
        h = h + layer.mlp.down_proj(silu(layer.mlp.gate_proj(hn)) * layer.mlp.up_proj(hn))
        return h, k_new, v_new


class LMPrefill(nn.Module):
    """7-layer Qwen3 prefill over TP tokens, causal, no cache."""

    def __init__(self, backbone, stop_predictor, tp=TP):
        super().__init__()
        self.layers = nn.ModuleList([Qwen3LoRALessLayer(l) for l in backbone.layers])
        self.norm = backbone.norm
        self.stop_predictor = stop_predictor
        self.tp = tp
        causal = (torch.tril(torch.ones(tp, tp, dtype=torch.float32)) - 1.0) * 30.0
        causal = causal.view(1, 1, tp, tp)
        self.register_buffer("mask01", causal)
        cos, sin = rope_tables(torch.arange(tp), HEAD_DIM, 1e6)
        self.register_buffer("cos", cos.view(1, 1, tp, HEAD_DIM))
        self.register_buffer("sin", sin.view(1, 1, tp, HEAD_DIM))

    def forward(self, embeds):
        h = embeds
        outs = []
        for layer in self.layers:
            h, k, v = layer(h, None, None, self.mask01, self.cos, self.sin)
            outs.extend([k, v])
        h = rms_norm(h, self.norm.weight)
        stop = self.stop_predictor(h[:, self.tp - 1, :])
        return [h, stop] + outs


class LMDecode(nn.Module):
    """One-token decode with fixed-capacity KV cache (B past entries)."""

    def __init__(self, backbone, stop_predictor, b=B):
        super().__init__()
        self.layers = nn.ModuleList([Qwen3LoRALessLayer(l) for l in backbone.layers])
        self.norm = backbone.norm
        self.stop_predictor = stop_predictor
        self.b = b

    def forward(self, embed, mask_add, cos, sin, *kvs):
        # kvs: k0,v0,k1,v1,... each [1,8,B,128]
        h = embed
        outs = []
        for i, layer in enumerate(self.layers):
            pk, pv = kvs[2 * i], kvs[2 * i + 1]
            h, k, v = layer(h, pk, pv, mask_add, cos, sin)
            outs.extend([k, v])
        h = rms_norm(h, self.norm.weight)
        stop = self.stop_predictor(h[:, 0, :])
        return [h, stop] + outs


class LocEncProj(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.locenc = model.locenc
        self.proj = model.locenc_to_lm_proj

    def forward(self, patches):
        return self.proj(self.locenc(patches))


class DiTHead(nn.Module):
    """4-step step-distilled Euler sampler fused into one graph."""

    def __init__(self, head, steps=4):
        super().__init__()
        self.head = head
        self.steps = steps

    def forward(self, x0, z, cond, speaker, cfg_strength):
        head = self.head
        dt = 1.0 / self.steps
        x = x0
        for step in range(self.steps):
            t = torch.full((1,), step * dt, dtype=torch.float32)
            dtt = torch.full((1,), dt, dtype=torch.float32)
            v = head._predict(
                x,
                t,
                z,
                cond,
                dt=dtt,
                speaker_embedding=speaker,
                cfg_strength=cfg_strength,
                validate_conditions=False,
            )
            x = x + v * dt
        return x


class VAEEncoder(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.encoder = vae.encoder

    def forward(self, wav):
        # [1,C,T] -> [1,T,C] to match the acoustic adapter's latent layout
        return self.encoder(wav, include_logvar=False)["mu"].transpose(1, 2)


class VAEDecoder(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.decoder = vae.decoder

    def forward(self, latents):
        return self.decoder(latents)


class SpeakerEncoder(nn.Module):
    def __init__(self, spk):
        super().__init__()
        self.spk = spk

    def forward(self, logmel):
        s = self.spk
        x = s.instance_norm(logmel)
        out1 = s.layer1(x)
        out2 = s.layer2(out1)
        out3 = s.layer3(out2)
        out4 = s.layer4(out3)
        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(s.conv(out))
        out = s.pooling(out)
        out = F.batch_norm(
            out, s.bn.running_mean, s.bn.running_var, s.bn.weight, s.bn.bias,
            False, s.bn.momentum, s.bn.eps,
        )
        raw = s.linear(out)
        return F.normalize(raw, dim=-1)


def logmel_numpy(wav: "torch.Tensor", mel_filter: "torch.Tensor", window: "torch.Tensor",
                 n_fft=512, hop=160, win=400, eps=1e-6):
    """Reference log-mel used to validate the numpy board implementation."""
    spec = torch.stft(wav, n_fft=n_fft, hop_length=hop, win_length=win, window=window,
                      center=True, return_complex=True)
    power = spec.abs().pow(2.0)
    mel = torch.matmul(mel_filter, power)
    return torch.log(mel.clamp_min(eps))


class RMSNormFn(nn.Module):
    """Drop-in nn.RMSNorm replacement that exports to primitive ops."""

    def __init__(self, norm: nn.RMSNorm):
        super().__init__()
        self.weight = nn.Parameter(norm.weight.detach().clone())
        self.eps = float(getattr(norm, "eps", 1e-6))

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)


def patch_rmsnorm(root: nn.Module) -> int:
    n = 0
    for name, child in list(root.named_children()):
        if isinstance(child, nn.RMSNorm):
            setattr(root, name, RMSNormFn(child))
            n += 1
        else:
            n += patch_rmsnorm(child)
    return n


class DiTStep(nn.Module):
    """One Euler step of the step-distilled Audio DiT (t / dt are inputs)."""

    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, x, z, cond, speaker, cfg_strength, t, dt):
        return self.head._predict(
            x, t, z, cond, dt=dt, speaker_embedding=speaker,
            cfg_strength=cfg_strength, validate_conditions=False,
        )


def stuff_dilated_depthwise(root: nn.Module) -> int:
    """Rewrite depthwise Conv1d with dilation>1 as an equivalent dilation-1 conv.

    Pulsar2 8.0's NPU tiler fails on grouped+dilated convs. A depthwise dilated
    conv is exactly a depthwise conv with the kernel zero-stuffed along time
    (dilation d <=> kernel (k-1)*d+1 with zeros at non-multiples of d).
    """
    n = 0
    for m in root.modules():
        if not isinstance(m, nn.Conv1d) or m.groups == 1:
            continue
        d = int(m.dilation[0])
        if d <= 1:
            continue
        try:
            torch.nn.utils.remove_weight_norm(m)
        except Exception:
            pass
        w = m.weight.detach()
        k = w.shape[-1]
        w2 = torch.zeros(w.shape[0], w.shape[1], (k - 1) * d + 1, dtype=w.dtype)
        w2[:, :, ::d] = w
        m.weight = nn.Parameter(w2)
        m.dilation = (1,)
        n += 1
    return n
