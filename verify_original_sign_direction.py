"""Recompute the original quotient-conditioned sign-head direction, Experiments 14/19."""
import io, json, zipfile
from pathlib import Path
import numpy as np
import torch
from model_extracted import HeadComparisonQ8Transformer
p=Path(__file__).resolve().parent
with zipfile.ZipFile(p/'checkpoint_and_config.zip') as z:
 ck=torch.load(io.BytesIO(z.read('conditional/best.pt')),map_location='cpu',weights_only=True)
model=HeadComparisonQ8Transformer('conditional').eval()
model.load_state_dict(ck['model_state_dict'],strict=True)
with zipfile.ZipFile(p/'validation_data.zip') as z:
 with np.load(io.BytesIO(z.read('q8_pilot_v1/validation.npz')),allow_pickle=False) as d:
  tokens=d['tokens'].copy();q=d['quotient_state'].copy();s=d['full_state'].copy()%2;ids=d['pair_id'].copy()
with np.load(p/'validation_activations.npz',allow_pickle=False) as d: saved=d['h'].copy()
_,inv,cnt=np.unique(ids,return_inverse=True,return_counts=True)
assert len(cnt)==512 and np.all(cnt==2)
pairs=np.argsort(inv,kind='stable').reshape(512,2);a,b=pairs.T
assert np.all(q[a]==q[b]) and np.all(s[a]!=s[b])
orientation=np.where(s[a]==1,1.,-1.)
W=model.conditional_sign_head.weight.detach().numpy().reshape(4,2,64)
w=W[:,1]-W[:,0];sel=w[q[a]];wn=np.linalg.norm(sel,axis=-1)
records={k:[] for k in ['input','attention','mlp','output','final_norm']}
with torch.inference_mode():
 for batch in torch.from_numpy(tokens).split(64):
  length=batch.shape[1];mask=torch.triu(torch.ones(length,length,dtype=torch.bool),1)
  h=model.token_embedding(batch)+model.positional_encoding[:length].unsqueeze(0)
  for block in model.blocks[:-1]:h=block(h,mask)
  block=model.blocks[-1];pre=h
  norm=block.attention_norm(pre)
  att,_=block.attention(norm,norm,norm,attn_mask=mask,need_weights=False)
  mlp=block.feedforward(block.feedforward_norm(pre+att))
  out=pre+att+mlp
  for k,v in [('input',pre),('attention',att),('mlp',mlp),('output',out),('final_norm',model.final_norm(out))]:records[k].append(v.numpy())
arr={k:np.concatenate(v,axis=0) for k,v in records.items()}
report={'checkpoint_epoch':int(ck['epoch']),'pairs':512,'direction':'conditional_sign_head.weight.reshape(4,2,64)[:,1]-[:,0], indexed by true quotient at each position','orientation':'h(sign=1)-h(sign=0) at each position','test_split_opened':False,'metrics':{},'saved_activation_max_abs_difference':float(np.max(np.abs(saved-arr['final_norm'])))}
for k,h in arr.items():
 delta=(h[a]-h[b])*orientation[...,None]
 proj=np.sum(delta*sel,axis=-1);norm=np.linalg.norm(delta,axis=-1)
 cosine=proj/np.maximum(norm*wn,1e-12)
 fraction=np.sum((proj/np.maximum(wn,1e-12))**2,axis=0)/np.maximum(np.sum(norm**2,axis=0),1e-12)
 mean_pair_fraction=np.mean((proj/np.maximum(wn,1e-12))**2/np.maximum(norm**2,1e-12),axis=0)
 report['metrics'][k]={'mean_distance':norm.mean(0).tolist(),'mean_signed_cosine':cosine.mean(0).tolist(),'parallel_energy_fraction_ratio_of_sums':fraction.tolist(),'mean_pairwise_parallel_energy_fraction':mean_pair_fraction.tolist(),'mean_signed_logit_projection':proj.mean(0).tolist()}
print('saved activation max abs difference',report['saved_activation_max_abs_difference'])
for k in ['final_norm','attention','mlp','output']:
 r=report['metrics'][k]
 print(k, 'signed cosine p1,p4,p12:',[round(r['mean_signed_cosine'][i],5) for i in [1,4,12]],'mean pairwise fraction p1,p4,p12:',[round(r['mean_pairwise_parallel_energy_fraction'][i],5) for i in [1,4,12]],'signed projection:',[round(r['mean_signed_logit_projection'][i],4) for i in [1,4,12]])
(p/'original_sign_direction_audit.json').write_text(json.dumps(report,indent=2)+'\n')
