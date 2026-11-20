"""Recompute final-block exact residual decomposition on validation, no test access."""
import io,json,zipfile,hashlib
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from model_extracted import HeadComparisonQ8Transformer
p=Path(__file__).resolve().parent
z=zipfile.ZipFile(p/'checkpoint_and_config.zip')
run=json.loads(z.read('conditional/run_config.json'))
ckpt=torch.load(io.BytesIO(z.read('conditional/best.pt')),map_location='cpu',weights_only=True)
model=HeadComparisonQ8Transformer('conditional').eval()
model.load_state_dict(ckpt['model_state_dict'],strict=True)
with zipfile.ZipFile(p/'validation_data.zip') as zd:
 b=zd.read('q8_pilot_v1/validation.npz')
assert hashlib.sha256(b).hexdigest()==run['data_sha256']['validation']
with np.load(io.BytesIO(b),allow_pickle=False) as d: tokens=torch.from_numpy(d['tokens'].copy());q=d['quotient_state'].copy();full=d['full_state'].copy();ids=d['pair_id'].copy()
unique,inv,counts=np.unique(ids,return_inverse=True,return_counts=True)
assert len(unique)==512 and np.all(counts==2)
ix=np.argsort(inv,kind='stable').reshape(-1,2)
assert np.all(q[ix[:,0]]==q[ix[:,1]]) and np.all((full[ix[:,0]]%2)!=(full[ix[:,1]]%2))
records={k:[] for k in ['input','attention','mlp','output']}
max_error=0.
with torch.inference_mode():
 for batch in tokens.split(64):
  length=batch.shape[1];mask=torch.triu(torch.ones(length,length,dtype=torch.bool),diagonal=1)
  h=model.token_embedding(batch)+model.positional_encoding[:length].unsqueeze(0)
  for block in model.blocks[:-1]:h=block(h,mask)
  block=model.blocks[-1];pre=h
  norm=block.attention_norm(pre)
  att,_=block.attention(norm,norm,norm,attn_mask=mask,need_weights=False)
  mlp=block.feedforward(block.feedforward_norm(pre+att))
  out=pre+att+mlp
  actual=block(pre,mask)
  max_error=max(max_error,float((out-actual).abs().max()))
  for k,v in [('input',pre),('attention',att),('mlp',mlp),('output',out)]:records[k].append(v.cpu())
arrays={k:torch.cat(v).numpy() for k,v in records.items()}
assert max_error<1e-5
# Paired-difference directional alignment: unit direction defined as normalized average paired difference at each position.
# Orientation: positive-sign member first for consistent averaging.
order=ix.copy(); swap=(full[order[:,0],1]%2==1);order[swap]=order[swap][:,::-1]
D={k:v[order[:,0]]-v[order[:,1]] for k,v in arrays.items()}
reference=D['output']; direction=reference.mean(axis=0);direction/=np.maximum(np.linalg.norm(direction,axis=-1,keepdims=True),1e-12)
report={'checkpoint_epoch':int(ckpt['epoch']),'pairs':512,'max_residual_reconstruction_error':max_error,'reference_direction':'normalized mean final-block output difference (positive-sign minus negative-sign) at each position','metrics':{},'test_split_opened':False,'causal_interventions_not_recomputed':True}
for k,d in D.items():
 projection=(d*direction[None]).sum(-1)
 energy=(projection**2)/(np.sum(d**2,axis=-1)+1e-12)
 report['metrics'][k]={'mean_paired_euclidean_distance':np.linalg.norm(d,axis=-1).mean(0).tolist(),'mean_signed_projection':projection.mean(0).tolist(),'mean_parallel_energy_fraction':energy.mean(0).tolist()}
(p/'decomposition_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print('max reconstruction error',max_error)
for k in D:print(k,'projection p1/p4/p12',*[round(report['metrics'][k]['mean_signed_projection'][i],4) for i in [1,4,12]])
