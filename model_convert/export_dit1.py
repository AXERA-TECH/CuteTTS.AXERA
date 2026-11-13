#!/usr/bin/env python3
import os, sys, numpy as np, torch
sys.path.insert(0,os.environ.get("CUTETTS_SRC", "/ai_data/AX615/hy/workspace/TTS_Train/TTS_quant/CuteTTS/src"))
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import export_onnx as E, models as M
cfg, model, vae, spk = E.load_all()
E.dh.F.scaled_dot_product_attention = M.manual_sdpa
ev=os.path.join(E.EXPORT,'eval_io')
x0=torch.from_numpy(np.load(f'{ev}/dit_x0.npy')); z=torch.from_numpy(np.load(f'{ev}/dit_z.npy'))
cond=torch.from_numpy(np.load(f'{ev}/dit_cond.npy')); sp=torch.from_numpy(np.load(f'{ev}/dit_speaker.npy'))
cfgt=torch.from_numpy(np.load(f'{ev}/dit_cfg.npy')); t=torch.tensor([0.0]); dt=torch.tensor([0.25])
m=M.DiTStep(model.head)
with torch.no_grad(): out=m(x0,z,cond,sp,cfgt,t,dt)
E.export(m,(x0,z,cond,sp,cfgt,t,dt),None,f'{E.ONNX}/dit_step.onnx',["velocity"],["x","z","cond","speaker","cfg_strength","t","dt"])
E.verify(f'{E.ONNX}/dit_step.onnx',{"x":x0,"z":z,"cond":cond,"speaker":sp,"cfg_strength":cfgt,"t":t,"dt":dt},[out],["velocity"])
# full-4-step check with the fused graph
full=M.DiTHead(model.head)
with torch.no_grad(): o4=full(x0,z,cond,sp,cfgt)
x=x0
with torch.no_grad():
    for k in range(4):
        v=m(x,z,cond,sp,cfgt,torch.tensor([k*0.25]),dt); x=x+v*0.25
print('4-step single-step-loop vs fused cos', float((x.flatten()@o4.flatten())/(x.norm()*o4.norm())))
