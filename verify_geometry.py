"""Recompute paired hidden-state Euclidean distances from saved validation activations.
This is a geometry-only reproducibility check, NOT an intervention rerun.
"""
import json, numpy as np
from pathlib import Path
p=Path(__file__).resolve().parent
with np.load(p/'validation_activations.npz',allow_pickle=False) as a:
 h=a['h']; ids=a['pair_id']; q=a['quotient_state']; full=a['full_state']
assert h.shape==(1024,13,64)
unique,inv,counts=np.unique(ids,return_inverse=True,return_counts=True)
assert len(unique)==512 and np.all(counts==2)
idx=np.argsort(inv,kind='stable').reshape(-1,2)
assert np.all(q[idx[:,0]]==q[idx[:,1]])
assert np.all((full[idx[:,0]]%2)!=(full[idx[:,1]]%2))
diff=h[idx[:,0]]-h[idx[:,1]]
mean_distance=np.linalg.norm(diff,axis=-1).mean(axis=0)
report={'source':'saved validation_activations.npz','pairs':512,'positions':list(range(13)),'mean_paired_euclidean_distance':[float(v) for v in mean_distance],'sign_direction_not_recomputed':True,'causal_interventions_not_recomputed':True,'test_split_opened':False}
(p/'geometry_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
