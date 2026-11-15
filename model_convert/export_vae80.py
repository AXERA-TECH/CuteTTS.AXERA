#!/usr/bin/env python3
"""Export the VAE decoder with an 80-latent window (60+ context, 16 new)."""
import os, sys
import numpy as np, torch
sys.path.insert(0,'/ai_data/AX615/hy/workspace/TTS_Train/TTS_quant/CuteTTS/src')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import export_onnx as E
from models import VAEDecoder

cfg, model, vae, spk = E.load_all()
dec = VAEDecoder(vae)
lat = torch.zeros(1, 64, 80)
with torch.no_grad():
    w = dec(lat)
print("L=80 ->", tuple(w.shape))
E.export(dec, (lat,), None, f"{E.ONNX}/vae_dec_l80.onnx", ["wav"], ["latent"])
E.verify(f"{E.ONNX}/vae_dec_l80.onnx", {"latent": lat}, [w], ["wav"])
