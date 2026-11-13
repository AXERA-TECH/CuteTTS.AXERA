"""Numpy-only audio helpers for the CuteTTS AX650 runtime.

WAV I/O, polyphase resampling and the exact log-mel front end used by the
speaker encoder (matches torch.stft(center=True) + mel matmul + log).
"""

import math
import wave

import numpy as np


def read_wav(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n_ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v >= 1 << 23, v - (1 << 24), v)
        x = v.astype(np.float32) / float(1 << 23)
    elif sw == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / float(1 << 31)
    else:
        raise ValueError(f"unsupported sample width: {sw}")
    if n_ch > 1:
        x = x.reshape(-1, n_ch).mean(axis=1)
    return np.ascontiguousarray(x, dtype=np.float32), sr


def write_wav(path, x, sr):
    x = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    pcm = (x * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())


def _firwin(cutoff, numtaps, beta=5.0):
    n = np.arange(numtaps)
    alpha = (numtaps - 1) / 2.0
    m = n - alpha
    h = cutoff * np.sinc(cutoff * m)
    h *= np.kaiser(numtaps, beta)
    h /= h.sum()
    return h


def resample_poly(x, up, down):
    x = np.asarray(x, dtype=np.float64)
    g = math.gcd(int(up), int(down))
    up, down = int(up) // g, int(down) // g
    if up == down:
        return x.astype(np.float32)
    max_rate = max(up, down)
    half = 10 * max_rate
    h = (_firwin(1.0 / max_rate, 2 * half + 1) * up).astype(np.float64)
    hl = len(h)
    n_out = int(math.ceil(len(x) * up / down))
    base = half + np.arange(n_out, dtype=np.int64) * down
    m0 = (base - hl + 1 + up - 1) // up
    m0 = np.maximum(m0, 0)
    acc = np.zeros(n_out, dtype=np.float64)
    for t in range(hl // up + 2):
        m = m0 + t
        j = base - m * up
        valid = (j >= 0) & (j < hl) & (m < len(x))
        acc[valid] += h[j[valid]] * x[m[valid]]
    return acc.astype(np.float32)


def resample_to(x, sr_in, sr_out):
    if int(sr_in) == int(sr_out):
        return np.asarray(x, dtype=np.float32).copy()
    return resample_poly(x, int(sr_out), int(sr_in))


def logmel(wav, mel_filter, window, n_fft=512, hop=160, win_length=400, eps=1e-6):
    """wav: mono float32 -> [n_mels, frames] float32 (torch-compatible)."""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    pad = n_fft // 2
    if len(wav) <= pad:
        wav = np.pad(wav, (0, pad + 1 - len(wav)))
    x = np.pad(wav, (pad, pad), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop]
    n = 1 + len(wav) // hop
    frames = frames[:n]
    w = np.zeros(n_fft, dtype=np.float32)
    s = (n_fft - win_length) // 2
    w[s:s + win_length] = window
    spec = np.fft.rfft(frames * w[None, :], n=n_fft, axis=-1)
    power = (spec.real ** 2 + spec.imag ** 2).T.astype(np.float32)
    mel = mel_filter.astype(np.float32) @ power
    return np.log(np.maximum(mel, eps)).astype(np.float32)
