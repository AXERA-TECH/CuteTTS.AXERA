#!/usr/bin/env python3
import glob, os, numpy as np, onnxruntime as ort
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAL=f"{ROOT}/calib/data"; os.makedirs(f"{CAL}/spk_pool/feats", exist_ok=True)
s=ort.InferenceSession(f"{ROOT}/export/onnx/spk_front.onnx", providers=["CPUExecutionProvider"])
files=sorted(glob.glob(f"{CAL}/speaker_enc/logmel/*.npy"))
for f in files:
    mel=np.load(f); feats=s.run(None,{"logmel":mel})[0]
    np.save(f"{CAL}/spk_pool/feats/feats_{os.path.basename(f)}", feats)
print(len(files),"feats saved")
