"""Controlled Q8 training comparison: original versus equal quotient/sign weights.
Usage: python run_loss_weight_control.py --archive /path/to/q8_pilot_backup_20261005_102000.zip --epochs 200
This never reads test.npz. It writes new files only under --out.
"""
import argparse, hashlib, io, json, random, sys, zipfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from model_extracted import HeadComparisonQ8Transformer

ARCHIVE_HASH='bb1678db82f93282acd69d426fb46c348dd9ee5ec902ed8b77f87b2c3d2ada50'
HASHES={'train':'8ff1da09a7935b9db3dd0ba3e212c258dd024a9fb21112b25ec95730e283c1a4', 'validation':'1ab60070506a3941f385d72c2ca36760e3bbe5998aecae3a8a766c86d74c9a8b'}

def load(archive):
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=ARCHIVE_HASH: raise ValueError('Archive SHA256 mismatch')
    out={}
    with zipfile.ZipFile(archive) as z:
        for split in HASHES:
            raw=z.read('q8_pilot_v1/'+split+'.npz')
            if hashlib.sha256(raw).hexdigest()!=HASHES[split]: raise ValueError(split+' hash mismatch')
            with np.load(io.BytesIO(raw),allow_pickle=False) as d:
                out[split]={k:torch.from_numpy(d[k].copy()).long() for k in ('tokens','full_state','quotient_state')}
    return out

def metrics(model,data,device,batch=256):
    model.eval(); qc=np.zeros(12,dtype=np.int64); sc=np.zeros(12,dtype=np.int64); fc=np.zeros(12,dtype=np.int64)
    n=len(data['tokens']); loss_sum=0.
    with torch.no_grad():
        for start in range(0,n,batch):
            t=data['tokens'][start:start+batch].to(device); q=data['quotient_state'][start:start+batch,1:].to(device); f=data['full_state'][start:start+batch,1:].to(device)
            o=model(t); qlog=o['quotient_logits'][:,1:]; slog=o['sign_logits'][:,1:]
            qpred=qlog.argmax(-1); spred=slog.gather(2,q[...,None,None].expand(-1,-1,1,2)).squeeze(2).argmax(-1)
            fpred=o['full_scores'][:,1:].argmax(-1)
            qc+=(qpred==q).sum(0).cpu().numpy();sc+=(spred==(f%2)).sum(0).cpu().numpy();fc+=(fpred==f).sum(0).cpu().numpy()
            loss_sum+=(F.nll_loss(o['full_scores'][:,1:].reshape(-1,8),f.reshape(-1),reduction='sum').item())
    return {'quotient_accuracy':(qc/n).tolist(),'sign_accuracy_true_q':(sc/n).tolist(),'full_accuracy':(fc/n).tolist(),'full_nll':loss_sum/(n*12)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--out',type=Path,default=HERE/'loss_weight_control_output');p.add_argument('--epochs',type=int,default=200);p.add_argument('--device',default='auto');p.add_argument('--smoke-test',action='store_true');args=p.parse_args()
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device))
    if device.type=='cpu':torch.set_num_threads(min(4,torch.get_num_threads()))
    data=load(args.archive); print('Verified original archive and both split hashes. Test split not read.',flush=True)
    if args.out.exists(): raise FileExistsError(f'Refusing to overwrite {args.out}')
    args.out.mkdir(parents=True)
    config={'seed':11,'shuffle_seed':1011,'batch_size':256,'lr':.001,'weight_decay':.01,'gradient_clip':1.,'epochs':args.epochs,'device':str(device),'torch':torch.__version__,'smoke_test':args.smoke_test,'original_loss':'0.5*(full_nll+q_ce)','symmetric_loss':'full_nll','note':'Full NLL = quotient CE + conditional sign CE; full_nll gives equal component weights.'}
    (args.out/'config.json').write_text(json.dumps(config,indent=2))
    for arm in ('original','symmetric'):
        random.seed(11);np.random.seed(11);torch.manual_seed(11)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(11)
        model=HeadComparisonQ8Transformer('conditional').to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        g=torch.Generator();g.manual_seed(1011);history=[]; best=float('inf')
        for epoch in range(1,args.epochs+1):
            model.train();ordering=torch.randperm(len(data['train']['tokens']),generator=g)
            for start in range(0,len(ordering),256):
                idx=ordering[start:start+256];t=data['train']['tokens'][idx].to(device);q=data['train']['quotient_state'][idx,1:].to(device);f=data['train']['full_state'][idx,1:].to(device)
                o=model(t); full=F.nll_loss(o['full_scores'][:,1:].reshape(-1,8),f.reshape(-1));qce=F.cross_entropy(o['quotient_logits'][:,1:].reshape(-1,4),q.reshape(-1))
                loss=.5*(full+qce) if arm=='original' else full
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
                if args.smoke_test:break
            m=metrics(model,data['validation'],device);m['epoch']=epoch;history.append(m)
            if m['full_nll']<best:
                best=m['full_nll'];torch.save({'model_state_dict':model.state_dict(),'epoch':epoch,'arm':arm},args.out/f'{arm}_best.pt')
            print(arm,epoch,'sign p1,p6,p12:',*[round(m['sign_accuracy_true_q'][i],4) for i in (0,5,11)],flush=True)
            if args.smoke_test:break
        (args.out/f'{arm}_history.json').write_text(json.dumps(history,indent=2))
    print('Finished. Exploratory validation only; no test data used.')
if __name__=='__main__':main()
