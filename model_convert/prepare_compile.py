#!/usr/bin/env python3
"""Pack calibration sets and generate Pulsar2 configs for CuteTTS-distill."""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tarfile

import onnx

ROOT = os.environ.get("CUTETTS_QUANT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ONNX = os.path.join(ROOT, "export", "onnx")
CALIB = os.path.join(ROOT, "calib")
COMPILE = os.path.join(ROOT, "compile")
PULSAR2 = os.environ.get("PULSAR2", "/root/.cache/magnetar/pulsar2/8.0/bin/pulsar2")
NPU_OVERRIDE = ""            # --npu：统一覆盖所有子图的 npu_mode（如 NPU2）

PREVIOUS_WEIGHT_WAS_U16 = True   # every graph was compiled with weight_data_type=U16
STREAM_STATES = [[1, 64, 6], [1, 1536, 1], [1, 768, 6], [1, 768, 18], [1, 768, 54], [1, 768, 1], [1, 384, 6], [1, 384, 18], [1, 384, 54], [1, 384, 1], [1, 192, 6], [1, 192, 18], [1, 192, 54], [1, 192, 1], [1, 96, 6], [1, 96, 18], [1, 96, 54], [1, 96, 6]]
DTYPE = "U16"
WEIGHT_DTYPE = "S8"   # previous configs used U16 weights -> switch to the w8a16 recipe below
SMOOTH_QUANT = True
PRECISION_ANALYSIS = True

SPECS = {
    "vae_enc": dict(
        onnx="vae_enc.onnx", inputs=[("wav", "1x1x120000")], calib="vae_enc/wav",
        npu="NPU3", out="vae_enc.axmodel"),
    "locenc_step": dict(
        onnx="locenc_step.onnx", inputs=[("patches", "1x1x2x64")], calib="locenc_step/patches",
        npu="NPU1", out="locenc_step.axmodel"),
    "dit_step": dict(
        onnx="dit_step.onnx",
        inputs=[("x", "1x2x64"), ("z", "1x1024"), ("cond", "1x2x64"),
                ("speaker", "1x256"), ("cfg_strength", "1"), ("t", "1"), ("dt", "1")],
        calib="dit_step", npu="NPU1", out="dit_step.axmodel"),
    "lm_prefill": dict(
        onnx="lm_prefill.onnx", inputs=[("embeds", "1x256x1024")], calib="lm_prefill/embeds",
        npu="NPU1", out="lm_prefill.axmodel"),
    "lm_decode": dict(
        onnx="lm_decode.onnx",
        inputs=[("embed", "1x1x1024"), ("mask_add", "1x1x1x385"), ("cos", "1x1x1x128"),
                ("sin", "1x1x1x128")]
               + [x for i in range(7) for x in ((f"pk{i}", "1x8x384x128"), (f"pv{i}", "1x8x384x128"))],
        calib="lm_decode", npu="NPU1", out="lm_decode.axmodel"),
    "vae_dec_stream": dict(
        onnx="vae_dec_stream.onnx",
        inputs=[("latent", "1x64x2")] + [(f"s{i}", "x".join(str(v) for v in s)) for i, s in enumerate(STREAM_STATES)],
        calib="vae_dec_stream", npu="NPU3", out="vae_dec_stream.axmodel"),
    "vae_dec_l80": dict(
        onnx="vae_dec_l80.onnx", inputs=[("latent", "1x64x80")], calib="vae_dec_l80/latent",
        npu="NPU3", out="vae_dec_l80.axmodel"),
    "vae_dec_l128": dict(
        onnx="vae_dec_l128.onnx", inputs=[("latent", "1x64x128")], calib="vae_dec_l128/latent",
        npu="NPU3", out="vae_dec_l128.axmodel"),
    "vae_dec_l256": dict(
        onnx="vae_dec_l256.onnx", inputs=[("latent", "1x64x256")], calib="vae_dec_l256/latent",
        npu="NPU3", out="vae_dec_l256.axmodel"),
}


def pack(model_name, spec):
    cdir = os.path.join(CALIB, "packed", model_name)
    os.makedirs(cdir, exist_ok=True)
    out = {}
    for name, _shape in spec["inputs"]:
        if "/" in spec["calib"]:
            files = sorted(glob.glob(f"{CALIB}/data/{spec['calib']}/*.npy"))
        else:
            files = sorted(glob.glob(f"{CALIB}/data/{spec['calib']}/{name}/*.npy"))
            if not files:
                files = sorted(glob.glob(f"{CALIB}/data/{spec['calib']}/{name}_*.npy"))
        assert files, (model_name, name)
        tar = os.path.join(cdir, f"{name}.tar.gz")
        with tarfile.open(tar, "w:gz") as tf:
            for f in files:
                tf.add(f, arcname=os.path.basename(f))
        out[name] = (tar, len(files))
    return out


def op_types(path):
    m = onnx.load(path)
    return sorted({n.op_type for n in m.graph.node})


def make_config(name, spec):
    onnx_path = os.path.join(ONNX, spec["onnx"])
    info = pack(name, spec)
    ops = op_types(onnx_path)
    n = next(iter(info.values()))[1]
    assert all(v[1] == n for v in info.values()), (name, {k: v[1] for k, v in info.items()})
    work = os.path.join(COMPILE, "work", name)
    cfg = {
        "input": onnx_path,
        "output_dir": COMPILE,
        "output_name": spec["out"],
        "work_dir": work,
        "model_type": "ONNX",
        "target_hardware": "AX650",
        "npu_mode": NPU_OVERRIDE or spec["npu"],
        "input_shapes": ";".join(f"{k}:{v}" for k, v in spec["inputs"]),
        "onnx_opt": {"disable_onnx_optimization": False, "enable_onnxsim": name == "vae_enc",
                     "model_check": True},
        "quant": {
            "calibration_method": "MinMax",
            "input_configs": [
                {"tensor_name": k, "calibration_dataset": info[k][0],
                 "calibration_format": "Numpy", "calibration_size": info[k][1],
                 "calibration_mean": [], "calibration_std": []}
                for k, _ in spec["inputs"]
            ],
            "layer_configs": [
                {"start_tensor_names": ["DEFAULT"], "end_tensor_names": ["DEFAULT"],
                 "data_type": DTYPE, "weight_data_type": WEIGHT_DTYPE,
                 "output_data_type": DTYPE}
            ] + ([
                # keep the streaming decoder's conv-state path in FP32: quantizing the
                # 18 state tensors every step accumulates drift (11 dB -> 18 dB)
                {"op_types": ["Concat", "Slice"], "data_type": "FP32",
                 "output_data_type": "FP32"},
            ] + [{"start_tensor_names": [f"s{i}"], "end_tensor_names": [f"s{i}"],
                  "data_type": "FP32", "output_data_type": "FP32"}
                 for i in range(len(STREAM_STATES))] if name == "vae_dec_stream" else []) \
            if PREVIOUS_WEIGHT_WAS_U16 else [
                {"op_types": ops, "data_type": DTYPE, "weight_data_type": "U16",
                 "output_data_type": DTYPE}
            ],
            "highest_mix_precision": False,
            **({"transformer_opt_level": 1} if name.startswith("lm_") else {}),
            "calibration_method": "MinMax",
            "enable_smooth_quant": SMOOTH_QUANT,
            "precision_analysis": PRECISION_ANALYSIS,
            "precision_analysis_method": "EndToEnd",
        },
        "input_processors": [
            {"tensor_name": k, "tensor_layout": "NCHW", "src_dtype": "FP32",
             "src_layout": "NCHW"} for k, _ in spec["inputs"]
        ],
        "compiler": {"check": 0},
    }
    path = os.path.join(COMPILE, f"pulsar2_config_{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"{name}: {n} calib samples, {len(ops)} ops -> {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--npu", default="", help="统一覆盖 npu_mode，如 NPU2")
    ap.add_argument("--compile-dir", default="",
                    help="输出目录（默认 <root>/compile），用于编译另一种 npu_mode 的变体")
    a = ap.parse_args()
    global COMPILE, NPU_OVERRIDE
    NPU_OVERRIDE = a.npu
    if a.compile_dir:
        COMPILE = os.path.abspath(a.compile_dir)
    names = a.models.split(",") if a.models else list(SPECS)
    os.makedirs(os.path.join(COMPILE, "logs"), exist_ok=True)
    for name in names:
        cfg = make_config(name, SPECS[name])
        if a.run:
            log = os.path.join(COMPILE, "logs", f"{name}.log")
            print(f"== build {name} (log {log})")
            with open(log, "w") as f:
                r = subprocess.run([PULSAR2, "build", "--config", cfg], stdout=f,
                                   stderr=subprocess.STDOUT)
            print("   rc", r.returncode, "log tail:")
            os.system(f"tail -n 6 {log}")


if __name__ == "__main__":
    main()
