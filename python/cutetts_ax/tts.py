"""CuteTTS-distill AX650 runtime (numpy + axengine, no torch/onnxruntime).

Pipeline (all heavy parts run on the NPU):

    reference wav --vae_enc--> latents --locenc--> speech embeddings \
    reference wav --logmel-> speaker_enc --> speaker embedding ---------> prompt embeds
    prompt embeds --lm_prefill--> hidden / kv cache
    loop:
        hidden --dit_head (4 Euler steps)--> latent patch
        latent patch --locenc_step--> next token embedding --lm_decode--\
    latents --vae_dec--> waveform
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from . import audio as A

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")


# --------------------------------------------------------------------- engines
class OnnxEngine:
    """Host-side verification engine (onnxruntime)."""

    def __init__(self, path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.output_names = [o.name for o in self.sess.get_outputs()]

    def run(self, feed):
        return self.sess.run(self.output_names, feed)

    def close(self):
        pass


class AxEngine:
    """AX650 NPU engine (pyaxengine)."""

    def __init__(self, path):
        import axengine as axe
        self.sess = axe.InferenceSession(path, providers=["AxEngineExecutionProvider"])
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.output_names = [o.name for o in self.sess.get_outputs()]

    def run(self, feed):
        return self.sess.run(self.output_names, {k: np.ascontiguousarray(feed[k]) for k in self.input_names})

    def close(self):
        self.sess = None


# ------------------------------------------------------------------ tokenizer
class Tokenizer:
    """SentencePiece with HuggingFace-compatible added-token splitting."""

    def __init__(self, model_path):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        folder = os.path.dirname(model_path)
        self.specials = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3}
        cfg_path = os.path.join(folder, "tokenizer_config.json")
        if os.path.exists(cfg_path):
            cfg = json.load(open(cfg_path))
            for tid, entry in cfg.get("added_tokens_decoder", {}).items():
                self.specials[entry["content"]] = int(tid)
        added = os.path.join(folder, "added_tokens.json")
        if os.path.exists(added):
            for k, v in json.load(open(added)).items():
                self.specials[k] = int(v)
        self._keys = sorted(self.specials, key=len, reverse=True)

    def encode(self, text):
        out, buf, i = [], [], 0
        while i < len(text):
            hit = None
            for k in self._keys:
                if text.startswith(k, i):
                    hit = k
                    break
            if hit is None:
                buf.append(text[i])
                i += 1
                continue
            chunk = "".join(buf).strip()
            if chunk:
                out += list(self.sp.encode(chunk, out_type=int))
            out.append(self.specials[hit])
            buf = []
            i += len(hit)
        chunk = "".join(buf).strip()
        if chunk:
            out += list(self.sp.encode(chunk, out_type=int))
        return out


class CuteTTSAX:
    def __init__(self, model_dir=None, assets_dir=ASSETS, engine="ax", verbose=True):
        t_init = time.time()
        self.root = os.path.abspath(model_dir or os.environ.get("CUTETTS_MODEL_DIR", "."))
        self.assets = os.path.abspath(assets_dir)
        self.cfg = json.load(open(os.path.join(self.assets, "config.json")))
        self.verbose = verbose
        base_mk = AxEngine if engine == "ax" else OnnxEngine

        def mk(path_):
            # AX CMM is a shared pool; retry briefly on transient load failures
            import gc
            last = None
            for _ in range(4):
                try:
                    return base_mk(path_)
                except Exception as exc:  # noqa: BLE001
                    last = exc
                    gc.collect()
                    time.sleep(0.4)
            raise last

        onnx_dir = os.path.join(self.root, "onnx")

        def path(name):
            for cand in (os.path.join(self.root, name + ".axmodel"),
                         os.path.join(self.root, "models", name + ".axmodel"),
                         os.path.join(onnx_dir, name + ".onnx")):
                if os.path.exists(cand):
                    return cand
            raise FileNotFoundError(name)

        self._find = path
        self._vae_enc = mk(path("vae_enc"))
        self.m_vae_stream = None
        self.vae_state_shapes = None
        self._stream_path = None
        sp = os.path.join(self.assets, "vae_stream.json")
        if os.path.exists(sp):
            try:
                self._stream_path = path("vae_dec_stream")
                self.vae_state_shapes = [tuple(int(v) for v in sh)
                                         for sh in json.load(open(sp))["states"]]
            except FileNotFoundError:
                self._stream_path = None
        self.m_spk = mk(path("spk_body"))
        self.m_loc_step = mk(path("locenc_step"))
        self.m_dit = mk(path("dit_step"))
        self.m_pref = mk(path("lm_prefill"))
        self.m_dec = mk(path("lm_decode"))
        self.m_vae_dec = {}
        self._dec_paths = {}
        for lb in (80, 128, 256):
            try:
                self._dec_paths[lb] = path(f"vae_dec_l{lb}")
            except FileNotFoundError:
                pass
        self.embed = np.load(os.path.join(self.assets, "embed_tokens.npy")).astype(np.float32)
        self.spk_proj = np.load(os.path.join(self.assets, "lm_speaker_proj.npy"))
        self.mel_filter = np.load(os.path.join(self.assets, "mel_filter.npy"))
        self.spk_bn = (np.load(os.path.join(self.assets, "spk_bn_mean.npy")),
                       np.load(os.path.join(self.assets, "spk_bn_var.npy")),
                       np.load(os.path.join(self.assets, "spk_bn_w.npy")),
                       np.load(os.path.join(self.assets, "spk_bn_b.npy")),
                       np.load(os.path.join(self.assets, "spk_linear_w.npy")),
                       np.load(os.path.join(self.assets, "spk_linear_b.npy")))
        self.mel_window = np.load(os.path.join(self.assets, "mel_window.npy"))
        self.tok = Tokenizer(self._find_tokenizer())
        self.scale = float(self.cfg["speech_scaling_factor"])
        self.bias = float(self.cfg["speech_bias_factor"])
        self.L = int(self.cfg["lm_prefill_len"])
        self.B = int(self.cfg["lm_kv_capacity"])
        self.sr = int(self.cfg["sample_rate"])
        self.rng = np.random.default_rng(42)
        self.last_timing = {}
        self.init_seconds = time.time() - t_init

    def _find_tokenizer(self):
        for p in (os.path.join(self.assets, "tokenizer", "tokenizer.model"),
                  os.path.join(self.root, "tokenizer", "tokenizer.model"),
                  os.path.join(self.root, "tokenizer.model")):
            if os.path.exists(p):
                return p
        raise FileNotFoundError("tokenizer.model")

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _rms(x):
        return np.sqrt(np.mean(x * x) + 1e-12)

    def _ref_latents(self, wav24):
        cap = int(self.cfg["vae_enc_samples"])
        hop = int(self.cfg["vae_hop"])
        x = np.asarray(wav24[:cap], dtype=np.float32)
        nreal = len(x) // hop
        if len(x) < cap:
            x = np.pad(x, (0, cap - len(x)))
        lat = self._vae_enc.run({"wav": x.reshape(1, 1, -1)})[0].reshape(-1, 64)
        return lat[:max(nreal, 1)].astype(np.float32)

    def _speaker_embedding(self, wav16):
        mel = A.logmel(wav16, self.mel_filter, self.mel_window,
                       n_fft=self.cfg["mel"]["n_fft"], hop=self.cfg["mel"]["hop"],
                       win_length=self.cfg["mel"]["win_length"], eps=self.cfg["mel"]["eps"])
        T = int(self.cfg["speaker_mel_frames"])
        if mel.shape[1] < T:
            mel = np.pad(mel, ((0, 0), (0, T - mel.shape[1])))
        else:
            mel = mel[:, :T]
        stats = self.m_spk.run({"logmel": mel[None]})[0]
        # BN + linear + L2-normalize on host (tiny, keeps the NPU graph conv-only)
        mean, var, w, b, lw, lb = self.spk_bn
        y = (stats - mean) / np.sqrt(var + 1e-5) * w + b
        emb = y @ lw.T + lb
        return (emb / np.linalg.norm(emb, axis=1, keepdims=True)).astype(np.float32)

    def _prompt_embeds(self, text, ref_lat_scaled, spk_emb):
        p = self.cfg["prompt"]
        ids_prefix = self.tok.encode(p["voice_prefix"])
        ids_mid = self.tok.encode(p["voice_mid"])
        ids_suffix = self.tok.encode(p["voice_suffix_fmt"].format(text))
        # speech latents -> patches (pad to even)
        T = ref_lat_scaled.shape[0]
        if T % 2:
            ref_lat_scaled = np.concatenate([ref_lat_scaled, np.zeros((1, 64), np.float32)], 0)
        patches = ref_lat_scaled.reshape(-1, 2, 64)
        n_patch = patches.shape[0]
        if n_patch > int(self.cfg["locenc_prompt_patches"]):
            n_patch = int(self.cfg["locenc_prompt_patches"])
            patches = patches[:n_patch]
        # LocEnc is per-patch: reuse the single-patch graph (cheap, 23..63 calls)
        speech_emb = np.concatenate(
            [self.m_loc_step.run({"patches": patches[i:i + 1][None].astype(np.float32)})[0][0]
             for i in range(n_patch)], axis=0)

        seq_ids = ids_prefix + [0] + ids_mid + [0] * n_patch + ids_suffix
        n = len(seq_ids)
        if n > self.L:
            raise ValueError(f"prompt too long ({n} > {self.L}); use a shorter reference/text")
        emb = self.embed[np.asarray(seq_ids, dtype=np.int64)]
        spk_pos = len(ids_prefix)
        emb[spk_pos] = (spk_emb @ self.spk_proj.T)[0]
        s0 = spk_pos + 1 + len(ids_mid)
        emb[s0:s0 + n_patch] = speech_emb
        out = np.zeros((1, self.L, 1024), np.float32)
        out[0, :n] = emb
        return out, n

    @staticmethod
    def _rope(pos):
        inv = 1.0 / (1e6 ** (np.arange(0, 128, 2, dtype=np.float32) / 128.0))
        freqs = np.float32(pos) * inv
        emb = np.concatenate([freqs, freqs])
        return (np.cos(emb).astype(np.float32).reshape(1, 1, 1, 128),
                np.sin(emb).astype(np.float32).reshape(1, 1, 1, 128))

    def _get_dec(self, bucket):
        if bucket not in self.m_vae_dec:
            if bucket not in self._dec_paths:
                raise ValueError(f"no VAE decoder for {bucket} latents")
            self.m_vae_dec[bucket] = self._find_engine(self._dec_paths[bucket])
        return self.m_vae_dec[bucket]

    def _find_engine(self, path_):
        mk = AxEngine if path_.endswith(".axmodel") else OnnxEngine
        import gc
        last = None
        for _ in range(4):
            try:
                return mk(path_)
            except Exception as exc:  # noqa: BLE001
                last = exc
                gc.collect()
                time.sleep(0.4)
        raise last

    def _get_stream_dec(self):
        if self.m_vae_stream is None and self._stream_path is not None:
            self.m_vae_stream = self._find_engine(self._stream_path)
        return self.m_vae_stream

    def warmup(self):
        """Run one tiny synthesis so the first real request does not pay NPU init cost."""
        try:
            import tempfile, wave as _wave  # noqa: F401
            ref = os.path.join(self.assets, "warmup.wav")
            if not os.path.exists(ref):
                import wave
                pcm = np.zeros(24000, np.int16)
                with wave.open(ref, "wb") as wf:
                    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(self.sr)
                    wf.writeframes(pcm.tobytes())
            for _ in self.synthesize_stream("warm up", ref, max_new=2, seed=1):
                pass
            return True
        except Exception:  # noqa: BLE001
            return False

    def _prepare_reference(self, reference_audio, ref_max_seconds=5.0):
        """Reference latents + speaker embedding, cached per (path, size, seconds)."""
        try:
            key = (os.path.abspath(reference_audio), os.path.getsize(reference_audio),
                   float(ref_max_seconds))
        except OSError:
            key = None
        cache = getattr(self, "_ref_cache", None)
        if cache is None:
            cache = self._ref_cache = {}
        if key is not None and key in cache:
            self.ref_cache_hit = True
            return cache[key]
        # disk cache (shared across processes, like the C++ runtime)
        disk_path = None
        if key is not None:
            import hashlib
            h = hashlib.md5("|".join(str(k) for k in key).encode()).hexdigest()[:16]
            disk_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "..", "cache", "ref", h + ".npz")
            disk_path = os.path.normpath(disk_path)
            if os.path.exists(disk_path):
                try:
                    z = np.load(disk_path)
                    cached = (z["lat"].astype(np.float32), z["spk"].astype(np.float32))
                    cache[key] = cached
                    self.ref_cache_hit = True
                    return cached
                except Exception:  # noqa: BLE001
                    pass
        self.ref_cache_hit = False
        wav, sr = A.read_wav(reference_audio)
        wav24 = A.resample_to(wav, sr, self.sr)
        cap = int(self.cfg["vae_enc_samples"]) / float(self.sr)
        wav24 = wav24[: int(min(ref_max_seconds, cap) * self.sr)]
        if len(wav24) < 2 * self.sr:
            wav24 = np.tile(wav24, int(np.ceil(2 * self.sr / len(wav24))))
        wav16 = A.resample_to(wav, sr, 16000)[: 8 * 16000]
        if len(wav16) < 32000:
            wav16 = np.tile(wav16, int(np.ceil(32000 / len(wav16))))
        lat = self._ref_latents(wav24)
        spk = self._speaker_embedding(wav16)
        if key is not None:
            cache[key] = (lat, spk)
            if disk_path is not None:
                try:
                    os.makedirs(os.path.dirname(disk_path), exist_ok=True)
                    np.savez(disk_path, lat=lat, spk=spk)
                except Exception:  # noqa: BLE001
                    pass
        return lat, spk

    # ------------------------------------------------------------------ streaming
    def _decode_window(self, vae_lat, start, end, length=128):
        """Exact VAE decode of latents [start, end) with a causal sliding window.

        The DAC decoder is causal with a ~60-latent receptive field (measured: zeroing
        the first 60 latents leaves all later samples bit-identical), so any window
        with >=64 latents of context reproduces the offline decode exactly.
        """
        w0 = max(0, end - length)
        win = np.zeros((1, 64, length), np.float32)
        n = end - w0
        win[0, :, :n] = np.stack(vae_lat[w0:end], axis=1)
        out = self._get_dec(length).run({"latent": win})[0][0, 0]
        return out[(start - w0) * 1920:(end - w0) * 1920]

    def _stream_window(self, chunk=16):
        """Smallest exact window that fits 64 context + ``chunk`` new latents."""
        if chunk <= 16 and 80 in self._dec_paths:
            return 80
        if chunk <= 64 and 128 in self._dec_paths:
            return 128
        return max(self._dec_paths)

    # --------------------------------------------------------------------- api
    def synthesize(self, text, reference_audio, max_new=None, cfg_strength=None, seed=42,
                   ref_max_seconds=5.0, rng=None):
        t0 = time.time()
        cfg_strength = self.cfg["cfg_strength"] if cfg_strength is None else float(cfg_strength)
        max_new = int(self.cfg["max_new_patches"] if max_new is None else max_new)
        self.rng = rng if rng is not None else np.random.default_rng(seed)

        # ---- reference features -------------------------------------------------
        t1 = time.time()
        lat, spk = self._prepare_reference(reference_audio, ref_max_seconds)
        t2 = t3 = time.time()
        lat_scaled = (lat + self.bias) * self.scale
        embeds, plen = self._prompt_embeds(text, lat_scaled, spk)
        t4 = time.time()

        # ---- prefill ------------------------------------------------------------
        pre = self.m_pref.run({"embeds": embeds})
        hidden = pre[0][0, plen - 1].reshape(1, 1024)
        stop = pre[1].reshape(2)
        kvs = []
        for i in range(7):
            kk = np.zeros((1, 8, self.B, 128), np.float32)
            vv = np.zeros((1, 8, self.B, 128), np.float32)
            kk[0, :, :plen] = pre[2 + 2 * i][0, :, :plen]
            vv[0, :, :plen] = pre[3 + 2 * i][0, :, :plen]
            kvs += [kk, vv]
        t5 = time.time()

        # ---- autoregressive loop ------------------------------------------------
        mask = np.full((1, 1, 1, self.B + 1), -30.0, np.float32)
        mask[..., :plen] = 0.0
        cond = np.zeros((1, 2, 64), np.float32)
        cfg_t = np.array([cfg_strength], np.float32)
        latents = []
        pos = plen
        n_steps = 0
        t_dit = t_dec = 0.0
        for step in range(max_new):
            stop_now = int(np.argmax(stop)) == 1
            x0 = self.rng.standard_normal((1, 2, 64)).astype(np.float32)
            t_a = time.time()
            x = x0
            for k in range(int(self.cfg["diffusion_steps"])):
                v = self.m_dit.run({"x": x, "z": hidden, "cond": cond, "speaker": spk,
                                    "cfg_strength": cfg_t, "t": np.array([k * 0.25], np.float32),
                                    "dt": np.array([0.25], np.float32)})[0]
                x = x + v * 0.25
            pred = x
            t_dit += time.time() - t_a
            latents.append(pred.copy())
            n_steps += 1
            if stop_now:
                break
            cond = pred
            emb = self.m_loc_step.run({"patches": pred[None].astype(np.float32)})[0]
            if pos >= self.B - 1:
                break
            mask[..., pos] = 0.0
            mask[..., self.B] = 0.0
            cos, sin = self._rope(pos)
            feed = {"embed": emb, "mask_add": mask, "cos": cos, "sin": sin}
            for i in range(7):
                feed[f"pk{i}"] = kvs[2 * i]
                feed[f"pv{i}"] = kvs[2 * i + 1]
            t_a = time.time()
            d = self.m_dec.run(feed)
            t_dec += time.time() - t_a
            hidden = d[0].reshape(1, 1024)
            stop = d[1].reshape(2)
            for i in range(7):
                kvs[2 * i][0, :, pos] = d[2 + 2 * i][0, :, 0]
                kvs[2 * i + 1][0, :, pos] = d[3 + 2 * i][0, :, 0]
            pos += 1
        t6 = time.time()

        # ---- decode -------------------------------------------------------------
        lat = np.concatenate(latents, axis=0)  # [N,2,64]
        self.last_patches = lat.copy()
        vae_lat = (lat / self.scale - self.bias).reshape(-1, 64).T[None]  # [1,64,2N]
        n = vae_lat.shape[-1]
        bucket = next((b for b in sorted(self._dec_paths) if n <= b), max(self._dec_paths))
        if n > bucket:
            raise ValueError(f"audio too long for available decoders ({n} > {bucket} latents)")
        padded = np.zeros((1, 64, bucket), np.float32)
        padded[:, :, :n] = vae_lat
        self.last_latents = vae_lat
        self.last_bucket = bucket
        wav_out = self._get_dec(bucket).run({"latent": padded})[0][0, 0, : n * int(self.cfg["vae_hop"])]
        t7 = time.time()

        self.last_timing = {
            "vae_enc": t2 - t1, "speaker_enc": t3 - t2, "prompt": t4 - t3,
            "prefill": t5 - t4, "ar_loop": t6 - t5, "dit": t_dit, "lm_decode": t_dec,
            "vae_dec": t7 - t6, "total": t7 - t0, "steps": n_steps,
            "audio_seconds": len(wav_out) / self.sr,
        }
        return wav_out.astype(np.float32)

    def synthesize_stream(self, text, reference_audio, max_new=None, cfg_strength=None, seed=42,
                          ref_max_seconds=5.0, rng=None, first_chunk=4, chunk=16, exact=True):
        """True streaming synthesis: yields float32 PCM chunks while the AR loop runs.

        Nothing is sentence-split: the AR loop keeps generating latents and audio is
        decoded/emitted every ``chunk`` latents (2.56 s) after an initial
        ``first_chunk`` latents (1.28 s), using an exact 128-latent sliding window.
        """
        t0 = time.time()
        cfg_strength = self.cfg["cfg_strength"] if cfg_strength is None else float(cfg_strength)
        max_new = int(self.cfg["max_new_patches"] if max_new is None else max_new)
        self.rng = rng if rng is not None else np.random.default_rng(seed)

        lat, spk = self._prepare_reference(reference_audio, ref_max_seconds)
        lat_scaled = (lat + self.bias) * self.scale
        embeds, plen = self._prompt_embeds(text, lat_scaled, spk)

        pre = self.m_pref.run({"embeds": embeds})
        hidden = pre[0][0, plen - 1].reshape(1, 1024)
        stop = pre[1].reshape(2)
        kvs = []
        for i in range(7):
            kk = np.zeros((1, 8, self.B, 128), np.float32)
            vv = np.zeros((1, 8, self.B, 128), np.float32)
            kk[0, :, :plen] = pre[2 + 2 * i][0, :, :plen]
            vv[0, :, :plen] = pre[3 + 2 * i][0, :, :plen]
            kvs += [kk, vv]

        mask = np.full((1, 1, 1, self.B + 1), -30.0, np.float32)
        mask[..., :plen] = 0.0
        cond = np.zeros((1, 2, 64), np.float32)
        cfg_t = np.array([cfg_strength], np.float32)
        vae_lat = []           # VAE-space latents, 64 floats each
        emitted = 0
        win_len = self._stream_window(chunk)
        if exact or self._get_stream_dec() is None:
            states = None
        else:
            states = [np.zeros(sh, np.float32) for sh in self.vae_state_shapes]
        pos = plen
        n_steps = 0
        ttft = None
        n_chunks = 0
        t_dit = t_dec = 0.0

        def emit_patch(pred):
            """stateful per-patch decode (160 ms per call)"""
            nonlocal states, ttft, n_chunks
            vae = pred[0] / self.scale - self.bias      # scaled (LM) space -> VAE space
            lat2 = np.ascontiguousarray(vae.T[None].astype(np.float32))  # [1,64,2]
            feed = {"latent": lat2}
            for i, st in enumerate(states):
                feed[f"s{i}"] = st
            out = self.m_vae_stream.run(feed)
            # copy: the engine reuses its output buffers, and we feed them back next step
            states = [np.array(o, copy=True) for o in out[1:]]
            n_chunks += 1
            if ttft is None:
                ttft = time.time() - t0
            return np.array(out[0][0, 0], copy=True)

        def emit(upto):
            nonlocal emitted, ttft, n_chunks
            audio = self._decode_window(vae_lat, emitted, upto, win_len)
            emitted = upto
            n_chunks += 1
            if ttft is None:
                ttft = time.time() - t0
            return audio

        for step in range(max_new):
            stop_now = int(np.argmax(stop)) == 1
            x0 = self.rng.standard_normal((1, 2, 64)).astype(np.float32)
            t_a = time.time()
            x = x0
            for k in range(int(self.cfg["diffusion_steps"])):
                v = self.m_dit.run({"x": x, "z": hidden, "cond": cond, "speaker": spk,
                                    "cfg_strength": cfg_t, "t": np.array([k * 0.25], np.float32),
                                    "dt": np.array([0.25], np.float32)})[0]
                x = x + v * 0.25
            t_dit += time.time() - t_a
            cond = x
            n_steps += 1
            for c in range(2):
                vae_lat.append((x[0, c] / self.scale - self.bias).astype(np.float32))

            if states is not None:
                yield emit_patch(x)
            if stop_now:
                if states is None and emitted < len(vae_lat):
                    yield emit(len(vae_lat))
                break

            emb = self.m_loc_step.run({"patches": x[None].astype(np.float32)})[0]
            if pos >= self.B - 1:
                if emitted < len(vae_lat):
                    yield emit(len(vae_lat))
                break
            mask[..., pos] = 0.0
            mask[..., self.B] = 0.0
            cos, sin = self._rope(pos)
            feed = {"embed": emb, "mask_add": mask, "cos": cos, "sin": sin}
            for i in range(7):
                feed[f"pk{i}"] = kvs[2 * i]
                feed[f"pv{i}"] = kvs[2 * i + 1]
            t_a = time.time()
            d = self.m_dec.run(feed)
            t_dec += time.time() - t_a
            hidden = d[0].reshape(1, 1024)
            stop = d[1].reshape(2)
            for i in range(7):
                kvs[2 * i][0, :, pos] = d[2 + 2 * i][0, :, 0]
                kvs[2 * i + 1][0, :, pos] = d[3 + 2 * i][0, :, 0]
            pos += 1

            if states is None:
                need = first_chunk if emitted == 0 else chunk
                if len(vae_lat) - emitted >= need:
                    yield emit(len(vae_lat))

        if states is None and emitted < len(vae_lat):
            yield emit(len(vae_lat))

        self.last_timing = {
            "ttft": ttft or (time.time() - t0), "total": time.time() - t0,
            "audio_seconds": len(vae_lat) / 12.5, "steps": n_steps,
            "chunks": n_chunks, "dit": t_dit, "lm_decode": t_dec,
            "streaming": True, "streaming_mode": ("stateful" if states is not None else f"window{win_len}"),
        }
