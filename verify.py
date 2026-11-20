import ast,io,json,zipfile,hashlib,math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pathlib import Path
base=Path(__file__).resolve().parent
notebook=json.loads((base/'source_notebook.ipynb').read_text())
source=''.join(notebook['cells'][16]['source'])
parsed=ast.parse(source)
classes=[n for n in parsed.body if isinstance(n,ast.ClassDef) and n.name in ('CausalBlock','HeadComparisonQ8Transformer')]
assert len(classes)==2
z=zipfile.ZipFile(base/'checkpoint_and_config.zip')
run=json.loads(z.read('conditional/run_config.json'))
MODEL_CONFIG=run['model_config']
namespace={'torch':torch,'nn':nn,'F':F,'math':math,'MODEL_CONFIG':MODEL_CONFIG}
exec(compile(ast.Module(body=classes,type_ignores=[]),'recovered_notebook_cell16','exec'),namespace)
model=namespace['HeadComparisonQ8Transformer']('conditional')
ckpt=torch.load(io.BytesIO(z.read('conditional/best.pt')),map_location='cpu',weights_only=True)
load=model.load_state_dict(ckpt['model_state_dict'],strict=True)
model.eval()
zdata=zipfile.ZipFile(base/'validation_data.zip')
member='q8_pilot_v1/validation.npz'; b=zdata.read(member)
assert hashlib.sha256(b).hexdigest()==run['data_sha256']['validation']
with np.load(io.BytesIO(b),allow_pickle=False) as d:
 tokens=torch.from_numpy(d['tokens'].copy()); q=torch.from_numpy(d['quotient_state'].copy()); full=torch.from_numpy(d['full_state'].copy())
with torch.inference_mode():
 outs=[]
 for chunk in tokens.split(128):outs.append(model(chunk))
 out={k:torch.cat([o[k] for o in outs],0) for k in outs[0]}
 qpred=out['quotient_logits'].argmax(-1)
 sign_logits=out['sign_logits'].gather(2,q[:,:,None,None].expand(-1,-1,1,2)).squeeze(2)
 signacc=(sign_logits.argmax(-1)==full%2).float().mean(0)
 quotientacc=(qpred==q).float().mean(0)
 fullpred=out['full_scores'].argmax(-1)
 fullacc=(fullpred==full).float().mean(0)
 qloss=F.cross_entropy(out['quotient_logits'][:,1:].reshape(-1,4),q[:,1:].reshape(-1))
 floss=F.nll_loss(out['full_scores'][:,1:].reshape(-1,8),full[:,1:].reshape(-1))
 joint=.5*(qloss+floss)
report={'checkpoint_epoch':ckpt['epoch'],'strict_load':str(load),'trainable_parameters':sum(p.numel() for p in model.parameters()),'checkpoint_validation':ckpt['validation'],'recomputed_validation_joint_loss':float(joint),'recomputed_quotient_accuracy_positions_1_to_12':float(quotientacc[1:].mean()),'recomputed_full_accuracy_positions_1_to_12':float(fullacc[1:].mean()),'quotient_accuracy_by_position':[round(float(x),7) for x in quotientacc],'conditional_sign_accuracy_by_position':[round(float(x),7) for x in signacc],'full_accuracy_by_position':[round(float(x),7) for x in fullacc],'validation_sha256':hashlib.sha256(b).hexdigest(),'notebook_cell':16,'test_split_opened':False}
(base/'recomputed_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
