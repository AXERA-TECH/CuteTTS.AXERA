#!/usr/bin/env python3
import glob, os, numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC=f"{ROOT}/calib/data/dit"; DST=f"{ROOT}/calib/data/dit_step"
for name in ("x","z","cond","speaker","cfg_strength","t","dt"):
    os.makedirs(f"{DST}/{name}", exist_ok=True)
tags=[]
for f in sorted(glob.glob(f"{SRC}/x0_*.npy")):
    tag=os.path.basename(f)[3:-4]
    x=np.load(f); z=np.load(f"{SRC}/z_{tag}.npy"); cond=np.load(f"{SRC}/cond_{tag}.npy")
    sp=np.load(f"{SRC}/speaker_{tag}.npy"); cfg=np.load(f"{SRC}/cfg_strength_{tag}.npy")
    for k in range(4):
        o=f"{tag}_s{k}"
        np.save(f"{DST}/x/x_{o}.npy",x.astype(np.float32))
        np.save(f"{DST}/z/z_{o}.npy",z.astype(np.float32))
        np.save(f"{DST}/cond/cond_{o}.npy",cond.astype(np.float32))
        np.save(f"{DST}/speaker/speaker_{o}.npy",sp.astype(np.float32))
        np.save(f"{DST}/cfg_strength/cfg_strength_{o}.npy",cfg.astype(np.float32))
        np.save(f"{DST}/t/t_{o}.npy",np.array([k*0.25],np.float32))
        np.save(f"{DST}/dt/dt_{o}.npy",np.array([0.25],np.float32))
        tags.append(o)
print(len(tags),"samples")
