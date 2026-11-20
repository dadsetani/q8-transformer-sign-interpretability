"""Validation-only paired action-permutation control; recompute every prefix label."""
import io,json,zipfile
from pathlib import Path
import numpy as np, torch
import sys
P=Path(__file__).resolve().parent
ROOT=P.parent
sys.path.insert(0, str(ROOT))
from model_extracted import HeadComparisonQ8Transformer
with zipfile.ZipFile(ROOT/'checkpoint_and_config.zip') as z:
 ck=torch.load(io.BytesIO(z.read('conditional/best.pt')),map_location='cpu',weights_only=True)
model=HeadComparisonQ8Transformer('conditional').eval();model.load_state_dict(ck['model_state_dict'],strict=True)
with zipfile.ZipFile(ROOT/'validation_data.zip') as z:
 with np.load(io.BytesIO(z.read('q8_pilot_v1/validation.npz')),allow_pickle=False) as d:
  tokens=d['tokens'].copy();full=d['full_state'].copy();quot=d['quotient_state'].copy();pid=d['pair_id'].copy()
N,L=tokens.shape
pairs=np.stack([np.flatnonzero(pid==p) for p in np.unique(pid)])
assert pairs.shape==(512,2)
assert np.all(tokens[pairs[:,0],1:]==tokens[pairs[:,1],1:])
# Recover right-multiplication transition table from all observed original prefixes.
transition=np.full((8,4),-1,dtype=int)
for row in range(N):
 for pos in range(1,L):
  old=int(full[row,pos-1]); act=int(tokens[row,pos]-8); new=int(full[row,pos]); assert 0<=act<4
  if transition[old,act]>=0:assert transition[old,act]==new
  transition[old,act]=new
assert np.all(transition>=0), 'Incomplete transition table; cannot safely reconstruct targets.'
assert np.all(quot==full//2)
# Verify complete original path reproduction before any permutation.
def labels_for(x):
 f=np.empty_like(x);f[:,0]=x[:,0]
 for t in range(1,L):f[:,t]=transition[f[:,t-1],x[:,t]-8]
 return f,f//2
assert np.array_equal(labels_for(tokens)[0],full)
# q transition must be independent of sign, and quotient composition commutative.
qtrans=np.empty((4,4),int)
for q in range(4):
 for a in range(4):
  vals=np.unique(transition[2*q:2*q+2,a]//2);assert len(vals)==1
  qtrans[q,a]=vals[0]
for q in range(4):
 for a in range(4):
  for b in range(4):assert qtrans[qtrans[q,a],b]==qtrans[qtrans[q,b],a]
@torch.inference_mode()
def predict(x):
 qpred=np.empty_like(x);spred=np.empty_like(x);fpred=np.empty_like(x)
 for start in range(0,N,64):
  out=model(torch.as_tensor(x[start:start+64],dtype=torch.long))
  qlog=out['quotient_logits']; qhat=qlog.argmax(-1)
  slog=out['sign_logits']; shat=slog.gather(2,qhat[...,None,None].expand(-1,-1,1,2)).squeeze(2).argmax(-1)
  qpred[start:start+64]=qhat.numpy();spred[start:start+64]=shat.numpy();fpred[start:start+64]=(2*qhat+shat).numpy()
 return qpred,spred,fpred
original_preds=predict(tokens)
def metrics(preds,f,q):
 qp,sp,fp=preds
 return {'quotient_accuracy':(qp==q).mean(axis=0).tolist(),
 'sign_accuracy':(sp==f%2).mean(axis=0).tolist(),
 'full_accuracy':(fp==f).mean(axis=0).tolist()}
base=metrics(original_preds,full,quot)
rng=np.random.default_rng(20261121)
permuted_results=[]
for repeat in range(5):
 x=tokens.copy()
 for a,b in pairs:
  perm=rng.permutation(L-1)+1
  x[a,1:]=tokens[a,perm];x[b,1:]=tokens[b,perm]
 f,q=labels_for(x)
 assert np.array_equal(q[:,-1],quot[:,-1]),'Endpoint quotient not invariant'
 assert np.array_equal(q[pairs[:,0]],q[pairs[:,1]])
 assert np.all((f[pairs[:,0]]%2)!=(f[pairs[:,1]]%2))
 p=predict(x);m=metrics(p,f,q)
 m['repeat']=repeat;m['final_sign_target_changed_fraction']=float(np.mean((f[:,-1]%2)!=(full[:,-1]%2)))
 m['intermediate_quotient_target_changed_fraction']=float(np.mean(q[:,1:-1]!=quot[:,1:-1]))
 m['end_quotient_prediction_agreement_with_original']=float(np.mean(p[0][:,-1]==original_preds[0][:,-1]))
 permuted_results.append(m)
report={'checkpoint_epoch':int(ck['epoch']),'validation_trajectories':N,'matched_pairs':512,'permutation_repeats':5,'permutation_seed':20261121,'sealed_test_opened':False,'action_transition_table':transition.tolist(),'baseline':base,'permutations':permuted_results}
for k in ('quotient_accuracy','sign_accuracy','full_accuracy'):
 arr=np.array([r[k] for r in permuted_results]);report['permuted_'+k+'_mean']=arr.mean(axis=0).tolist();report['permuted_'+k+'_sd_across_repeats']=arr.std(axis=0,ddof=1).tolist()
(P/'permutation_control_results.json').write_text(json.dumps(report,indent=2)+'\n')
print('checkpoint',ck['epoch'],'N',N,'repeats',5)
for k in ('quotient_accuracy','sign_accuracy','full_accuracy'):
 print(k,'original final',round(base[k][-1],5),'permuted final',round(report['permuted_'+k+'_mean'][-1],5),'original pos1-12',round(np.mean(base[k][1:]),5),'permuted pos1-12',round(np.mean(report['permuted_'+k+'_mean'][1:]),5))
print('fraction changed final sign',np.mean([r['final_sign_target_changed_fraction'] for r in permuted_results]))
print('fraction changed intermediate q',np.mean([r['intermediate_quotient_target_changed_fraction'] for r in permuted_results]))
