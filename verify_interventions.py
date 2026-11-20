"""Independent replay of notebook Experiments 23–24, validation only."""
import io,json,zipfile
from pathlib import Path
import numpy as np
import torch
from model_extracted import HeadComparisonQ8Transformer
P=Path(__file__).resolve().parent
with zipfile.ZipFile(P/'checkpoint_and_config.zip') as z:
 ck=torch.load(io.BytesIO(z.read('conditional/best.pt')),map_location='cpu',weights_only=True)
model=HeadComparisonQ8Transformer('conditional').eval()
model.load_state_dict(ck['model_state_dict'],strict=True)
with zipfile.ZipFile(P/'validation_data.zip') as z:
 with np.load(io.BytesIO(z.read('q8_pilot_v1/validation.npz')),allow_pickle=False) as d:
  tokens=d['tokens'].copy();q=d['quotient_state'].copy();s=d['full_state'].copy()%2;pair_ids=d['pair_id'].copy()
_,inv,counts=np.unique(pair_ids,return_inverse=True,return_counts=True)
assert len(counts)==512 and np.all(counts==2)
pairs=np.argsort(inv,kind='stable').reshape(512,2)
assert np.all(q[pairs[:,0]]==q[pairs[:,1]]) and np.all(s[pairs[:,0]]!=s[pairs[:,1]])
block=model.blocks[-1]
conditions=['baseline','attention_ablation','own_mlp_restore','opposite_sign_mlp_restore']
results={c:{k:[] for k in ('prob','sign_pred','q_pred')} for c in conditions}
def zero_output(module,inputs,output):
 if isinstance(output,tuple):return (torch.zeros_like(output[0]),*output[1:])
 return torch.zeros_like(output)
def evaluate(out,q_true,s_true):
 selected=out['sign_logits'].gather(2,q_true[...,None,None].expand(-1,-1,1,2)).squeeze(2)
 prob=selected.softmax(-1).gather(-1,s_true[...,None]).squeeze(-1)
 return prob.numpy(),selected.argmax(-1).numpy(),out['quotient_logits'].argmax(-1).numpy()
with torch.inference_mode():
 for start in range(0,len(pairs),64):
  idx=pairs[start:start+64].reshape(-1)
  x=torch.as_tensor(tokens[idx],dtype=torch.long);qt=torch.as_tensor(q[idx],dtype=torch.long);st=torch.as_tensor(s[idx],dtype=torch.long)
  saved=[]
  h=block.feedforward.register_forward_hook(lambda m,i,o:saved.append(o.detach().clone()))
  try:baseline=model(x)
  finally:h.remove()
  assert len(saved)==1
  own=saved[0];opp=own.reshape(-1,2,*own.shape[1:]).flip(1).reshape_as(own)
  def intervene(restore=None):
   hooks=[block.attention.register_forward_hook(zero_output)]
   if restore is not None:
    def hook(m,i,o):
     assert o.shape==restore.shape
     return restore
    hooks.append(block.feedforward.register_forward_hook(hook))
   try:return model(x)
   finally:
    for h in reversed(hooks):h.remove()
  outs=dict(zip(conditions,[baseline,intervene(),intervene(own),intervene(opp)]))
  for c,out in outs.items():
   for k,v in zip(('prob','sign_pred','q_pred'),evaluate(out,qt,st)):results[c][k].append(v)
for c in conditions:
 for k in results[c]:results[c][k]=np.concatenate(results[c][k],axis=0)
prob_diff=results['own_mlp_restore']['prob']-results['opposite_sign_mlp_restore']['prob']
pair_diff=prob_diff[pairs].mean(axis=1)
rng=np.random.default_rng(20261119)
bootstrap_indices=rng.integers(0,512,size=(5000,512))
boot=pair_diff[bootstrap_indices].mean(axis=1)
early=pair_diff[:,1:6].mean(axis=1)
early_boot=early[bootstrap_indices].mean(axis=1)
report={'epoch':int(ck['epoch']),'validation_trajectories':1024,'pairs':512,'bootstrap_replicates':5000,'bootstrap_seed':20261119,'test_split_opened':False,'conditions':{},'contrasts':[],'early_positions_1_to_5':{'mean':float(early.mean()),'ci95':np.percentile(early_boot,[2.5,97.5]).tolist()}}
for c in conditions:
 r=results[c]
 report['conditions'][c]={
  'mean_true_sign_probability':r['prob'].mean(axis=0).tolist(),
  'conditional_sign_accuracy':(r['sign_pred']==s).mean(axis=0).tolist(),
  'quotient_accuracy':(r['q_pred']==q).mean(axis=0).tolist(),
  'full_state_accuracy':((r['sign_pred']==s)&(r['q_pred']==q)).mean(axis=0).tolist()}
for i in range(13):report['contrasts'].append({'position':i,'own_minus_opposite_mean':float(pair_diff[:,i].mean()),'ci95':np.percentile(boot[:,i],[2.5,97.5]).tolist()})
np.savez_compressed(P/'intervention_bootstrap_arrays.npz',pair_difference=pair_diff,bootstrap_indices=bootstrap_indices,bootstrap_means=boot)
(P/'intervention_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print('epoch',report['epoch'],'early mean and CI',report['early_positions_1_to_5'])
for i in (1,4,5,6,12):print('position',i,report['contrasts'][i],'own prob',report['conditions']['own_mlp_restore']['mean_true_sign_probability'][i],'opposite prob',report['conditions']['opposite_sign_mlp_restore']['mean_true_sign_probability'][i])
