"""Validation-only Q8 MLP-restoration controls. No checkpoint training or test access."""
import io,json,zipfile
from pathlib import Path
import numpy as np,torch
from model_extracted import HeadComparisonQ8Transformer
P=Path(__file__).resolve().parent
rng=np.random.default_rng(20261121)
torch.manual_seed(20261121)
with zipfile.ZipFile(P/'checkpoint_and_config.zip') as z:
 ck=torch.load(io.BytesIO(z.read('conditional/best.pt')),map_location='cpu',weights_only=True)
model=HeadComparisonQ8Transformer('conditional').eval();model.load_state_dict(ck['model_state_dict'],strict=True)
with zipfile.ZipFile(P/'validation_data.zip') as z:
 with np.load(io.BytesIO(z.read('q8_pilot_v1/validation.npz')),allow_pickle=False) as d:
  tokens=d['tokens'].copy();q=d['quotient_state'].copy();s=d['full_state'].copy()%2;pair_ids=d['pair_id'].copy()
_,inv,counts=np.unique(pair_ids,return_inverse=True,return_counts=True)
assert len(counts)==512 and np.all(counts==2)
pairs=np.argsort(inv,kind='stable').reshape(512,2)
assert np.all(q[pairs[:,0]]==q[pairs[:,1]]) and np.all(s[pairs[:,0]]!=s[pairs[:,1]])
N,L=tokens.shape
# Preselect independent donors, matching q and s separately at each position.
donors=np.empty((N,L),dtype=int)
donors_other_q=np.empty((N,L),dtype=int)
donors_opp_sign=np.empty((N,L),dtype=int)
for pos in range(L):
 for i in range(N):
  choices=np.flatnonzero((q[:,pos]==q[i,pos])&(s[:,pos]==s[i,pos])&(pair_ids!=pair_ids[i]))
  if not len(choices):raise RuntimeError(f'No matched donor for sample {i}, position {pos}')
  donors[i,pos]=rng.choice(choices)
  other=np.flatnonzero((q[:,pos]!=q[i,pos])&(s[:,pos]==s[i,pos])&(pair_ids!=pair_ids[i]))
  opposite=np.flatnonzero((q[:,pos]==q[i,pos])&(s[:,pos]!=s[i,pos])&(pair_ids!=pair_ids[i]))
  if not len(other) or not len(opposite):raise RuntimeError(f'No contrast donor at {i},{pos}')
  donors_other_q[i,pos]=rng.choice(other)
  donors_opp_sign[i,pos]=rng.choice(opposite)
block=model.blocks[-1]
def zero(m,inp,out):
 if isinstance(out,tuple):return (torch.zeros_like(out[0]),*out[1:])
 return torch.zeros_like(out)
def intervene(x,restore):
 hs=[block.attention.register_forward_hook(zero)]
 if restore is not None:hs.append(block.feedforward.register_forward_hook(lambda m,i,o:restore))
 try:return model(x)
 finally:
  for h in reversed(hs):h.remove()
def prob(out,qt,st):
 logits=out['sign_logits'].gather(2,qt[...,None,None].expand(-1,-1,1,2)).squeeze(2)
 return logits.softmax(-1).gather(-1,st[...,None]).squeeze(-1).numpy()
# Record natural MLP outputs for all validation trajectories, in canonical order.
mlp=[]
with torch.inference_mode():
 for start in range(0,N,64):
  x=torch.as_tensor(tokens[start:start+64],dtype=torch.long)
  h=block.feedforward.register_forward_hook(lambda m,i,o:mlp.append(o.detach().clone()))
  try:model(x)
  finally:h.remove()
mlp=torch.cat(mlp,dim=0)
assert mlp.shape[:2]==(N,L)
# Reorder by pair, so all interventions preserve the matched-pair structure.
order=pairs.reshape(-1);invorder=np.empty(N,int);invorder[order]=np.arange(N)
opp=np.empty(N,int);opp[pairs[:,0]]=pairs[:,1];opp[pairs[:,1]]=pairs[:,0]
position_idx=torch.arange(L)[None,:]
matched=mlp[torch.as_tensor(donors),position_idx]
other_q=mlp[torch.as_tensor(donors_other_q),position_idx]
opp_sign_independent=mlp[torch.as_tensor(donors_opp_sign),position_idx]
# Independent isotropic direction, exact position-wise Euclidean norm of own MLP.
random_direction=torch.randn(mlp.shape)
random_direction=random_direction/(random_direction.norm(dim=-1,keepdim=True)+1e-12)
random_norm_matched=random_direction*mlp.norm(dim=-1,keepdim=True)
conditions={'attention_zero':None,'own':mlp,'opposite':mlp[torch.as_tensor(opp)],'matched_same_q_sign':matched,'random_equal_norm':random_norm_matched,'same_sign_other_q':other_q,'same_q_opposite_sign_independent':opp_sign_independent}
probs={c:np.empty((N,L),dtype=np.float32) for c in conditions}
with torch.inference_mode():
 for start in range(0,N,64):
  idx=order[start:start+64]
  x=torch.as_tensor(tokens[idx],dtype=torch.long)
  qt=torch.as_tensor(q[idx],dtype=torch.long);st=torch.as_tensor(s[idx],dtype=torch.long)
  for name,restore in conditions.items():
   out=intervene(x,None if restore is None else restore[idx]);probs[name][idx]=prob(out,qt,st)
# Pair-level resampling, same as original intervention audit.
bs=np.random.default_rng(20261119).integers(0,512,size=(5000,512))
def summary(a):
 pair=a[pairs].mean(axis=1);early=pair[:,1:6].mean(axis=1)
 return {'early_mean':float(early.mean()),'early_ci95':np.percentile(early[bs].mean(axis=1),[2.5,97.5]).tolist(),'by_position':pair.mean(axis=0).tolist()}
report={'checkpoint_epoch':int(ck['epoch']),'validation_trajectories':N,'pairs':512,'test_split_opened':False,'donor_matching':'same q/sign, same sign different q, same q opposite sign: independent pair donor at same position','random_control':'isotropic Gaussian direction scaled to exact own MLP output norm per sample/position','seed_donors':20261121,'seed_random_vectors':20261121,'bootstrap_seed':20261119,'bootstrap_replicates':5000,'conditions':{},'contrasts':{}}
for k,v in probs.items():report['conditions'][k]=summary(v)
for a,b in [('own','opposite'),('matched_same_q_sign','opposite'),('own','matched_same_q_sign'),('own','random_equal_norm'),('matched_same_q_sign','random_equal_norm'),('matched_same_q_sign','same_sign_other_q'),('matched_same_q_sign','same_q_opposite_sign_independent'),('same_sign_other_q','same_q_opposite_sign_independent')]:report['contrasts'][a+'_minus_'+b]=summary(probs[a]-probs[b])
(P/'factorial_donor_control_results.json').write_text(json.dumps(report,indent=2)+'\n')
np.savez_compressed(P/'factorial_donor_control_arrays.npz',**probs,donors=donors,donors_other_q=donors_other_q,donors_opp_sign=donors_opp_sign,pairs=pairs)
print('Checkpoint',report['checkpoint_epoch'],'N',N)
for k,v in report['conditions'].items():print('condition',k,round(v['early_mean'],6),v['early_ci95'])
for k,v in report['contrasts'].items():print('contrast',k,round(v['early_mean'],6),v['early_ci95'])
