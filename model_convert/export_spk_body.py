#!/usr/bin/env python3
import os, sys, numpy as np, torch
sys.path.insert(0,os.environ.get("CUTETTS_SRC", "/ai_data/AX615/hy/workspace/TTS_Train/TTS_quant/CuteTTS/src"))
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import export_onnx as E
import torch.nn as nn, torch.nn.functional as F
cfg, model, vae, spk = E.load_all()
ev=os.path.join(E.EXPORT,'eval_io')

class Body(nn.Module):
    """log-mel -> attentive stats [mean|std] (BN/linear/L2 done on host)."""
    def __init__(self,s): super().__init__(); self.s=s
    def forward(self, logmel):
        s=self.s
        x=s.instance_norm(logmel)
        o1=s.layer1(x); o2=s.layer2(o1); o3=s.layer3(o2); o4=s.layer4(o3)
        f=F.relu(s.conv(torch.cat([o2,o3,o4],1)))
        p=s.pooling
        mean=torch.mean(f,dim=-1,keepdim=True).expand_as(f)
        std=torch.sqrt(torch.var(f,dim=-1,keepdim=True)+1e-10).expand_as(f)
        alpha=torch.tanh(p.linear1(torch.cat((f,mean,std),dim=1)))
        alpha=torch.softmax(p.linear2(alpha),dim=2)
        m=torch.sum(alpha*f,dim=2)
        resid=torch.sum(alpha*(f**2),dim=2)-m**2
        return torch.cat([m,torch.sqrt(resid.clamp(min=1e-9))],dim=1)

mel=torch.from_numpy(np.load(f'{ev}/spk_logmel.npy'))
body=Body(spk)
with torch.no_grad(): st=body(mel)
print('stats',tuple(st.shape))
E.export(body,(mel,),None,f'{E.ONNX}/spk_body.onnx',["stats"],["logmel"])
E.verify(f'{E.ONNX}/spk_body.onnx',{"logmel":mel},[st],["stats"])
np.save(f'{ev}/spk_body_stats.npy', st.numpy())
# host head weights
torch.save({}, f'{ev}/_dummy.pt')
np.save(f'{E.ROOT}/sdk/assets/spk_bn_mean.npy', spk.bn.running_mean.numpy())
np.save(f'{E.ROOT}/sdk/assets/spk_bn_var.npy', spk.bn.running_var.numpy())
np.save(f'{E.ROOT}/sdk/assets/spk_bn_w.npy', spk.bn.weight.detach().numpy())
np.save(f'{E.ROOT}/sdk/assets/spk_bn_b.npy', spk.bn.bias.detach().numpy())
np.save(f'{E.ROOT}/sdk/assets/spk_linear_w.npy', spk.linear.weight.detach().numpy())
np.save(f'{E.ROOT}/sdk/assets/spk_linear_b.npy', spk.linear.bias.detach().numpy())
print('bn eps', spk.bn.eps, 'emb', tuple(F.normalize(spk.linear(F.batch_norm(st, spk.bn.running_mean, spk.bn.running_var, spk.bn.weight, spk.bn.bias, False, spk.bn.momentum, spk.bn.eps)),dim=-1).shape))
