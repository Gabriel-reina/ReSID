"""Training recipes for the shared GR and frozen-encoder semantic projection.

Use new output directories. Training reads TRAIN and VALID only; never TEST.
Checkpoint replay is the exact-results track; fresh training is a separate track.
"""
import argparse,json,math,random,shutil,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader,TensorDataset
from .data import DEFAULT_ROOT,read_rows,histories
from .model import GenerativeRecommender,ModelConfig
from .decoding import PaddedSIDTrie,UserProjection
from .cli import new_output

def split_tensors(root,dataset,split,codes,pad):
    rows=read_rows(root,dataset,split);targets=np.array([row[2] for row in rows])
    return {"history":histories(rows,codes,pad),"sid":torch.tensor([codes[i] for i in targets]),"targets":targets}

@torch.no_grad()
def validation_nll(model,data,pad):
    model.eval();loss=0.;count=0
    for a in range(0,len(data['targets']),256):
        y=data['sid'][a:a+256];logits=model(data['history'][a:a+256],y);loss+=float(F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=pad,reduction='sum'));count+=int((y!=pad).sum())
    return loss/count

def train(model,train,valid,trie,out,phase,epochs,lr,negatives=None):
    optim=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4);loader=DataLoader(TensorDataset(train['history'],train['sid'],torch.from_numpy(train['targets'])),batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(42));rng=np.random.default_rng(42);steps=len(loader)*epochs;current=0;best=math.inf;records=[]
    for epoch in range(1,epochs+1):
        start=time.perf_counter();model.train();tot=[]
        for h,y,ids in loader:
            warm=max(1,int(.1*steps));factor=(current+1)/warm if current<warm else .1+.9*.5*(1+math.cos(math.pi*min(1,(current-warm)/max(1,steps-warm))))
            for group in optim.param_groups:group['lr']=lr*factor if phase=='nll' else lr
            mem,pad=model.encode(h);logits=model.decode(mem,pad,y[:,:-1]);nll=F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=trie.pad);ul=nll*0
            if negatives is not None:
                eligible=np.flatnonzero(negatives[ids.numpy(),0]>=0)
                if len(eligible):
                    chosen=negatives[ids.numpy()[eligible],rng.integers(0,5,size=len(eligible))];ny=torch.tensor([trie.padded[i] for i in chosen]);nl=model.decode(mem[eligible],pad[eligible],ny[:,:-1]);mask=ny!=trie.pad;safe=ny.clamp_max(trie.vocab-1);prob=nl.log_softmax(-1).gather(-1,safe[:,:,None]).squeeze(-1).exp();ul=(-torch.log((1-prob).clamp_min(1e-5))*mask).sum()/mask.sum()
            loss=nll+.1*ul;assert torch.isfinite(loss);optim.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step();current+=1;tot.append([float(nll.detach()),float(ul.detach())])
        val=validation_nll(model,valid,trie.pad);row={'phase':phase,'epoch':epoch,'train_nll_ul':np.mean(tot,0).tolist(),'validation_nll':val,'seconds':time.perf_counter()-start};records.append(row);print(json.dumps(row),flush=True)
        state={'model':model.state_dict(),'config':model.checkpoint_config(),'phase':phase,'epoch':epoch,'validation_nll':val,'eos':trie.eos,'pad':trie.pad}
        torch.save(state,out/f'{phase}-last.pt')
        if val<best:best=val;torch.save(state,out/f'{phase}-best.pt')
        (out/f'{phase}-trace.json').write_text(json.dumps(records,indent=2))
    state=torch.load(out/f'{phase}-best.pt',weights_only=True);model.load_state_dict(state['model']);return {'phase':phase,'epoch':state['epoch'],'validation_nll':state['validation_nll']}

class LowRank(nn.Module):
 def __init__(self,w):
  super().__init__();self.A=nn.Parameter(torch.randn(8,w.shape[1])*.01);self.B=nn.Parameter(torch.zeros(w.shape[0],8))
 def forward(self,w):return w+2*(self.B@self.A)

def add_lora(model):
 for p in model.parameters():p.requires_grad=False
 for m in model.modules():
  if isinstance(m,nn.MultiheadAttention):
   parametrize.register_parametrization(m,'in_proj_weight',LowRank(m.in_proj_weight));parametrize.register_parametrization(m.out_proj,'weight',LowRank(m.out_proj.weight))
 for layer in list(model.code_embeddings)+list(model.output_heads):
  for p in layer.parameters():p.requires_grad=True

def tree_edges(codes):
 return [[(int(p),np.unique(codes[codes[:,l]==p,l+1]).tolist()) for p in np.unique(codes[:,l])] for l in range(codes.shape[1]-1)]

def tree_loss(model,edges):
 loss=model.bos.sum()*0
 # Model uses separate position tables. e(child at l+1) follows Eq9.
 for l,parents in enumerate(edges):
  for _,children in parents:
   emb=model.code_embeddings[l+1].weight[children];loss=loss+(emb-emb.mean(0)).square().sum(-1).mean()
 return loss


def train_plain(model,trainset,valid,pad,out,epochs=15,classic=False,edges=None):
    # Ours preserves the original LambdaLR and global-generator DataLoader.
    # Classic EOS baselines preserve their explicit generator and manual schedule.
    kwargs={"generator":torch.Generator().manual_seed(42)} if classic else {}
    loader=DataLoader(TensorDataset(trainset["history"],trainset["sid"],torch.from_numpy(trainset["targets"])),batch_size=256,shuffle=True,**kwargs)
    is_crab=edges is not None;lr=1e-4 if is_crab else 1e-3
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=.01 if is_crab else 1e-4)
    total=len(loader)*epochs;warm=max(1,int(total*.1));step=0
    def factor(s):return (s+1)/warm if s<warm else .1+.9*.5*(1+math.cos(math.pi*min(1,(s-warm)/max(1,total-warm))))
    scheduler=None if classic else torch.optim.lr_scheduler.LambdaLR(opt,factor)
    best=math.inf;trace=[]
    for epoch in range(1,epochs+1):
        model.train();losses=[]
        for h,y,_ in loader:
            if classic and not is_crab:
                for g in opt.param_groups:g["lr"]=lr*factor(step)
            logits=model(h,y);ce=F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=pad)
            loss=ce+.2*tree_loss(model,edges) if is_crab else ce
            if not torch.isfinite(loss):raise ValueError("Nonfinite loss")
            opt.zero_grad(set_to_none=not classic);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            if scheduler:scheduler.step()
            step+=1;losses.append(float(loss.detach()))
        if classic:val=validation_nll(model,valid,pad)
        else:
            # Original ours validation iterates a DataLoader, consuming its base seed.
            model.eval();value=0.;count=0
            with torch.no_grad():
                for h,y,_ in DataLoader(TensorDataset(valid["history"],valid["sid"],torch.from_numpy(valid["targets"])),batch_size=256):
                    value+=float(F.cross_entropy(model(h,y).flatten(0,1),y.flatten(),ignore_index=pad,reduction="sum"));count+=int((y!=pad).sum())
            val=value/count
        row={"epoch":epoch,"validation_nll":val,"train_loss":float(np.mean(losses))};trace.append(row);print("GR",row,flush=True)
        state={"model":model.state_dict(),"config":model.checkpoint_config(),"epoch":epoch}
        torch.save(state,out/"last.pt")
        if val<best-(0 if classic else 1e-5):best=val;torch.save(state,out/"model.pt")
        (out/"trace.json").write_text(json.dumps(trace,indent=2))
    model.load_state_dict(torch.load(out/"model.pt",weights_only=True)["model"])
    if is_crab:
        for module in list(model.modules()):
            if parametrize.is_parametrized(module):
                for key in list(module.parametrizations.keys()):parametrize.remove_parametrizations(module,key,leave_parametrized=True)
        state=torch.load(out/"model.pt",weights_only=True);state["model"]=model.state_dict();torch.save(state,out/"model.pt")

def init_crab(model,codes,eos,base,maps):
    old=torch.load(base/"model.pt",weights_only=True);oldcodes=json.loads((base/"codes.json").read_text())
    oldeos=old["config"]["code_size"]-1;oldpad=oldeos+1;trie=PaddedSIDTrie(codes,eos);state=model.state_dict()
    for key,value in state.items():
        if key.startswith(("code_embeddings.","output_heads.")):
            layer=int(key.split(".")[1]);ov=old["model"][key];lookup=maps[layer] if layer<4 else {}
            for token in range(value.shape[0]):
                source=oldpad if token==trie.pad else oldeos if token==trie.eos else int(lookup.get(str(token),token))
                if source<ov.shape[0]:value[token]=ov[source]
        elif value.shape==old["model"][key].shape:state[key]=old["model"][key]
    model.load_state_dict(state);add_lora(model)

@torch.no_grad()
def contexts(model,history):
    result=[];model.eval()
    for h in history.split(512):
        memory,pad=model.encode(h);w=(~pad).float();result.append((memory*w[:,:,None]).sum(1)/w.sum(1)[:,None])
    return torch.cat(result)

def projection(root,dataset,artifact,out,epochs=8):
    state=torch.load(artifact/"model.pt",weights_only=True);encoder=GenerativeRecommender(ModelConfig(**state["config"])).eval();encoder.load_state_dict(state["model"])
    codes=json.loads((artifact/"codes.json").read_text());data=split_tensors(root,dataset,"train",codes,encoder.config.code_size)
    with np.load(root/"artifacts"/dataset/"ours/text_features.npz") as z:features={k:z[k] for k in z.files};text=torch.from_numpy(features["item_vectors"])
    # PCA is fixed catalogue content. Reuse the distributed PCA to isolate GR/projection training.
    np.savez_compressed(out/"text_features.npz",**features);x=contexts(encoder,data["history"]);y=torch.from_numpy(data["targets"])
    torch.manual_seed(42);model=UserProjection("linear");opt=torch.optim.Adam(model.parameters(),lr=.001);rng=np.random.default_rng(42);trace=[]
    for epoch in range(1,epochs+1):
        model.train();order=rng.permutation(len(x));losses=[]
        for a in range(0,len(x),256):
            ix=order[a:a+256];score=model(x[ix])@text.T/.07;loss=F.cross_entropy(score,y[ix]);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();losses.append(float(loss.detach()))
        trace.append({"epoch":epoch,"loss":float(np.mean(losses))});torch.save(model.state_dict(),out/f"projection-ep{epoch}.pt")
        print("PROJECTION",trace[-1],flush=True)
    shutil.copyfile(out/f"projection-ep{epochs}.pt",out/"projection.pt");(out/"trace.json").write_text(json.dumps(trace,indent=2))

def main(argv=None,root=DEFAULT_ROOT):
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=root);ap.add_argument("--dataset",choices=["industrial","office"],required=True)
    ap.add_argument("--method",choices=["ours","rqvae","tiger","onerec","crab","ghost","sa2crq","projection"],required=True)
    ap.add_argument("--artifact",type=Path,help="Input SID/model directory; defaults to frozen artifacts");ap.add_argument("--tiger-artifact",type=Path);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--epochs",type=int);ap.add_argument("--threads",type=int,default=2)
    args=ap.parse_args(argv);root=Path(args.root);out=new_output(args.output);torch.set_num_threads(args.threads);torch.manual_seed(42);np.random.seed(42);random.seed(42)
    artifact=args.artifact or root/"artifacts"/args.dataset/("ours" if args.method=="projection" else args.method)
    if args.method=="projection":projection(root,args.dataset,artifact,out,args.epochs or 8);return
    codes=json.loads((artifact/"codes.json").read_text());shutil.copyfile(artifact/"codes.json",out/"codes.json")
    is_ours=args.method=="ours"
    if is_ours:pcodes=codes;pad=16;config=ModelConfig(levels=6,code_size=16)
    else:
        meta=json.loads((artifact/"sid_metadata.json").read_text());eos=meta["EOS"];trie=PaddedSIDTrie(codes,eos);pcodes=trie.padded;pad=trie.pad;config=ModelConfig(levels=trie.levels,code_size=trie.vocab)
        shutil.copyfile(artifact/"sid_metadata.json",out/"sid_metadata.json")
    trainset=split_tensors(root,args.dataset,"train",pcodes,pad);valid=split_tensors(root,args.dataset,"valid",pcodes,pad)
    model=GenerativeRecommender(config)
    if args.method=="crab":
        maps=json.loads((artifact/"parent-maps.json").read_text());init_crab(model,codes,eos,args.tiger_artifact or root/"artifacts"/args.dataset/"tiger",maps)
        train_plain(model,trainset,valid,pad,out,args.epochs or 10,True,tree_edges(np.array(codes)))
    elif args.method in ("ghost","sa2crq"):
        train(model,trainset,valid,trie,out,"nll",args.epochs or 15,1e-3)
        if args.method=="ghost":
            negatives=artifact/"ghost-negative-items.npy"
            if not negatives.exists():raise FileNotFoundError("Ghost requires training-only negative table: "+str(negatives))
            selection=train(model,trainset,valid,trie,out,"auo",4,3e-5,np.load(negatives))
            selected=out/"auo-best.pt"
        else:selected=out/"nll-best.pt"
        shutil.copyfile(selected,out/"model.pt")
    else:train_plain(model,trainset,valid,pad,out,args.epochs or 15,classic=not is_ours)
    (out/"training_protocol.json").write_text(json.dumps({"seed":42,"dataset":args.dataset,"method":args.method,"selection":"minimum VALID token NLL; no early stopping; never TEST","cpu_recipe":True},indent=2))

if __name__=="__main__":main()
