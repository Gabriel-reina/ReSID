"""Catalogue-only SID construction recipes. No TEST interactions are consumed."""
import argparse,json,math,shutil,time,random
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .data import DEFAULT_ROOT,frequency




from .kmeans import nearest, torch_kmeans
from third_party.minionerec.rqvae import RQVAE
from third_party.minionerec.vq import VectorQuantizer

def load_initialized(model,state):
    model.load_state_dict(state)
    for module in model.modules():
        if isinstance(module,VectorQuantizer):module.initted=True
    return model

class ParallelQuantizer(nn.Module):
    """Each subspace uses the same upstream trainable VQ and loss as RQ."""
    def __init__(self, levels, size, dimension):
        super().__init__()
        if dimension % levels:
            raise ValueError('Latent dimension must be divisible by every SID length')
        self.levels = levels
        self.vq_layers = nn.ModuleList([VectorQuantizer(size, dimension // levels,
            beta=.25, kmeans_init=True, kmeans_iters=100, sk_epsilon=0., sk_iters=50)
            for _ in range(levels)])

    def forward(self, x, use_sk=True):
        outputs = [layer(block, use_sk=use_sk) for layer, block in zip(self.vq_layers, x.chunk(self.levels, -1))]
        return torch.cat([o[0] for o in outputs], -1), torch.stack([o[1] for o in outputs]).mean(), torch.stack([o[2] for o in outputs], -1)

def make_model(dimension, mode, levels, size):
    model = RQVAE(in_dim=dimension, num_emb_list=[size] * levels, e_dim=96,
        layers=[512, 256, 128], dropout_prob=0., bn=False, loss_type='mse',
        quant_loss_weight=1., beta=.25, kmeans_init=True, kmeans_iters=100,
        sk_epsilons=[0.] * levels, sk_iters=50)
    if mode == 'parallel':
        model.rq = ParallelQuantizer(levels, size, 96)
    return model

def code_stats(codes):
    counts = Counter(map(tuple, codes.tolist()))
    n = len(codes)
    return {'unique_sids': len(counts), 'extra_rate': (n - len(counts)) / n,
            'participating_rate': sum(c for c in counts.values() if c > 1) / n,
            'largest_group': max(counts.values())}

def train_one(x, args, mode, levels, size, initial):
    name = f'{mode}{levels}x{size}'
    directory = args.output / name
    directory.mkdir(exist_ok=False)
    torch.manual_seed(42)
    model = make_model(x.shape[1], mode, levels, size)
    model.load_state_dict(initial, strict=False)
    # Initialize all codebooks on the full catalogue before gradient training.
    model.train()
    with torch.no_grad():
        model(x)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.)
    records, frontier = [], []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for group in optimizer.param_groups:
            group['lr'] = 1e-3 * min(1., epoch / 50)
        optimizer.zero_grad(set_to_none=True)
        out, qloss, _ = model(x)
        loss, _ = model.compute_loss(out, qloss, xs=x)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite tokenizer loss')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                recon, qloss, codes = model(x, use_sk=False)
                mse = F.mse_loss(recon, x).item()
            stat = code_stats(codes)
            row = {'epoch': epoch, 'mse': mse, 'quantization_loss': qloss.item(), **stat}
            records.append(row)
            # Keep catalogue-quality Pareto checkpoints so final loss constraint can be applied exactly.
            dominated = any(r['mse'] <= mse and r['extra_rate'] <= stat['extra_rate'] for r in frontier)
            if not dominated:
                survivors = []
                for r in frontier:
                    if mse <= r['mse'] and stat['extra_rate'] <= r['extra_rate']:
                        (directory / r['filename']).unlink()
                    else:
                        survivors.append(r)
                filename = f'pareto-epoch{epoch}.pt'
                torch.save({'state': model.state_dict(), 'epoch': epoch, 'mode': mode,
                            'levels': levels, 'size': size, 'dimension': x.shape[1]}, directory / filename)
                frontier = survivors + [{**row, 'filename': filename}]
            (directory / 'training-history.json').write_text(json.dumps(records, indent=2))
            if epoch == 1 or epoch % 100 == 0 or epoch == args.epochs:
                print(name, f'epoch={epoch}/{args.epochs} mse={mse:.8f} collision={stat["participating_rate"]*100:.2f}%', flush=True)
    best_mse = min(frontier, key=lambda r: (r['mse'], r['extra_rate']))
    best_collision = min(frontier, key=lambda r: (r['extra_rate'], r['mse']))
    eligible = [r for r in frontier if r['mse'] <= best_mse['mse'] * 1.05]
    chosen = min(eligible, key=lambda r: (r['extra_rate'], r['mse']))
    for label, r in [('best-mse', best_mse), ('best-collision', best_collision), ('selected', chosen)]:
        shutil.copyfile(directory / r['filename'], directory / f'{label}.pt')
    state = torch.load(directory / 'selected.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(state['state'])
    model.eval()
    with torch.no_grad():
        recon, _, codes = model(x, use_sk=False)
    codes = codes.numpy()
    np.save(directory / 'codes.npy', codes)
    np.save(directory / 'reconstruction-errors.npy', (recon - x).square().mean(-1).numpy())
    (directory/'codes.json').write_text(json.dumps(codes.astype(int).tolist()))
    unchanged = 'Inputs are read-only; output contains codes only'
    positions = []
    for j in range(levels):
        counts = np.bincount(codes[:, j], minlength=size)
        p = counts[counts > 0] / len(codes)
        positions.append({'position': j + 1, 'used': int((counts > 0).sum()),
                          'entropy_bits': float(-(p * np.log2(p)).sum()), 'largest_cluster': int(counts.max())})
    result = {'name': name, 'mode': mode, 'levels': levels, 'size': size,
              'nominal_capacity': size ** levels, 'selected': chosen, 'best_mse': best_mse,
              'best_collision': best_collision, 'positions': positions,
              'autoencoder_parameters': sum(p.numel() for n, p in model.named_parameters() if not n.startswith('rq.')),
              'codebook_parameters': sum(p.numel() for n, p in model.named_parameters() if n.startswith('rq.')),
              'non_sid_fields_unchanged': unchanged, 'seconds': time.perf_counter() - start}
    (directory / 'summary.json').write_text(json.dumps(result, indent=2))
    print(name, 'tokenizer complete', json.dumps(result), flush=True)
    return result

def head_split(f):
    # Match author's strict >80th-percentile frequency threshold, use this
    # project's TRAIN-target count convention. Evaluation groups stay unchanged.
    values=np.sort(f[f>0]);threshold=int(values[int(np.ceil(.8*len(values)))-1])
    return np.flatnonzero(f>threshold),np.flatnonzero(f<=threshold),threshold

@torch.no_grad()
def lloyd(x,k,seed):
    rng=np.random.default_rng(seed);c=x[rng.choice(len(x),k,replace=len(x)<k)].clone()
    for _ in range(30):
        d=torch.cdist(x,c).square();ids=d.argmin(1);count=torch.bincount(ids,minlength=k);new=torch.zeros_like(c);new.index_add_(0,ids,x);live=count>0;new[live]/=count[live,None];new[~live]=c[~live]
        if torch.allclose(new,c,atol=1e-5):c=new;break
        c=new
    return c

class AnchoredRQ(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Sequential(nn.Linear(32,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,32));self.decoder=nn.Sequential(nn.Linear(32,64),nn.ReLU(),nn.Linear(64,128),nn.ReLU(),nn.Linear(128,32))
        self.capacities=[64,128,256,256];self.head=nn.ParameterList([nn.Parameter(torch.randn(k,32)*.01) for k in self.capacities]);self.extension=nn.ParameterList();self.tail_phase=False;self.prior={};self.head_total=1;self.tau=2e-6

    def books(self):return [torch.cat([h.detach(),e],0) for h,e in zip(self.head,self.extension)] if self.tail_phase else list(self.head)

    def forward(self,x):
        z=self.encoder(x);qsum=torch.zeros_like(z);indices=[];loss=z.sum()*0;active=torch.ones(len(x),dtype=torch.bool,device=x.device);depth=torch.full((len(x),),4,dtype=torch.long)
        for d,book in enumerate(self.books()):
            # Paper's reported operational SARQ: test first N=2 prefix ratio.
            # Head empirical priors frozen before tail training; unseen=0 stops.
            if d==2:
                prefix=torch.stack(indices,1).detach().cpu().tolist();mass=torch.tensor([self.prior.get(tuple(p),0)/self.head_total for p in prefix],device=x.device)
                stopped=mass<=self.tau;depth[stopped]=2;active=~stopped
            residual=z-qsum.detach();ids=torch.cdist(residual,book).square().argmin(1);q=F.embedding(ids,book);w=active.float()[:,None]
            cb=((q-residual.detach()).square()*w).sum()/w.sum().clamp_min(1)/32
            commit=((residual-q.detach()).square()*w).sum()/w.sum().clamp_min(1)/32
            loss=loss+cb+.25*commit;qsum=qsum+q*w;indices.append(ids)
        zq=z+(qsum-z).detach();reconstruction=self.decoder(zq)
        return reconstruction,loss,torch.stack(indices,1),depth

    @torch.no_grad()
    def initialize(self,x):
        z=self.encoder(x);res=z.clone()
        for d,h in enumerate(self.head):
            h.copy_(lloyd(res,len(h),42+d));ids=torch.cdist(res,h).argmin(1);res=res-h[ids]
        self.update_prior(x)

    @torch.no_grad()
    def update_prior(self,x):
        # Build first2 paths without gating; probabilities are catalogue counts.
        z=self.encoder(x);paths=[];res=z
        for book in list(self.head)[:2]:
            ids=torch.cdist(res,book).argmin(1);paths.append(ids);res=res-book[ids]
        self.prior=dict(Counter(map(tuple,torch.stack(paths,1).cpu().tolist())));self.head_total=len(x)

    @torch.no_grad()
    def start_tail(self,x):
        z=self.encoder(x);res=z.clone();extensions=[]
        for d,h in enumerate(self.head):
            count=256-len(h);new=lloyd(res,count,142+d) if count else torch.empty(0,32);extensions.append(nn.Parameter(new));book=torch.cat([h,new],0);ids=torch.cdist(res,book).argmin(1);res=res-book[ids]
        self.extension=nn.ParameterList(extensions);self.tail_phase=True
        for p in self.head:p.requires_grad_(False)

def sa2crq_tokenizer(x,f,out):
    torch.manual_seed(42);head,tail,threshold=head_split(f);model=AnchoredRQ();tx=torch.from_numpy(x);model.initialize(tx[head]);records=[];frozen_codes=None;start=time.perf_counter()
    for phase,ids in [('head',head),('tail',tail)]:
        if phase=='tail':
            model.eval();model.update_prior(tx[head])
            with torch.no_grad():_,_,hc,hd=model(tx[head])
            frozen_codes=[c[:int(d)].tolist() for c,d in zip(hc,hd)];anchors=[p.detach().clone() for p in model.head];model.start_tail(tx[tail])
        optim=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-4,weight_decay=0);rng=np.random.default_rng(42)
        for epoch in range(1,101):
            model.train();losses=[]
            for ix in np.array_split(rng.permutation(ids),max(1,int(np.ceil(len(ids)/1024)))):
                recon,qloss,_,_=model(tx[ix]);loss=F.mse_loss(recon,tx[ix])+qloss;assert torch.isfinite(loss);optim.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step();losses.append(float(loss.detach()))
            if phase=='head' and epoch%10==0:model.update_prior(tx[head])
            if epoch==1 or epoch%20==0:
                row={'phase':phase,'epoch':epoch,'loss':float(np.mean(losses))};records.append(row);print('SA2CRQ_TOKENIZER',row,flush=True)
        if phase=='head':torch.save(model.state_dict(),out/'sa2crq-head-stage.pt')
    assert all(torch.equal(a,b) for a,b in zip(anchors,model.head));model.eval()
    with torch.no_grad():_,_,tc,td=model(tx[tail])
    codes=[None]*len(x)
    for i,c in zip(head,frozen_codes):codes[int(i)]=c
    for i,c,d in zip(tail,tc,td):codes[int(i)]=c[:int(d)].tolist()
    torch.save({'model':model.state_dict(),'head_prefix_counts':model.prior,'head_total':model.head_total,'tau':model.tau},out/'sa2crq-tokenizer.pt');(out/'sa2crq-tokenizer-trace.json').write_text(json.dumps(records,indent=2))
    return codes,{'head_threshold_strict_gt':threshold,'head_items':len(head),'tail_items':len(tail),'max_depth':4,'head_capacities':model.capacities,'tail_total_capacities':[256]*4,'tau':model.tau,'check_prefix_N':2,'head_epochs':100,'tail_epochs':100,'head_anchors_bitwise_frozen':True,'final_stats':sid_stats(codes),'head_lengths':dict(Counter(map(len,frozen_codes))),'tail_lengths':dict(Counter(int(d) for d in td)),'seconds':time.perf_counter()-start,'provenance':'Paper-based local implementation; no official repository located in current search','adaptations':['catalogue PCA32 input and compact AE128/64/32','TRAIN-target frequency split shared with Ghost port','200 total tokenizer epochs split100 head/100 tail; paper omits phase-specific counts','operational N=2 threshold from paper Table1 note; generic per-layer entropy-budget schedule not specified','preserve head assignments from head stage; no alias token appended, retain one-to-many mapping']}


def sid_stats(codes):
    counts=Counter(map(tuple,codes));n=len(codes)
    return {"items":n,"unique_sids":len(counts),"extra_collision_rate":1-len(counts)/n,
            "collision_participation_rate":sum(v for v in counts.values() if v>1)/n,
            "max_aliases":max(counts.values()),"length_histogram":dict(Counter(map(len,codes)))}

def save_codes(out,codes,detail):
    codes=[[int(t) for t in c] for c in codes]
    (out/"codes.json").write_text(json.dumps(codes))
    (out/"sid_metadata.json").write_text(json.dumps({**detail,**sid_stats(codes),"EOS":max(max(c) for c in codes)+1},indent=2))

# Independent implementation of the constrained residual clustering construction.
# The external constrained solver is optional and is NOT vendored here.
def constrained_residual(x,layers):
    from k_means_constrained import KMeansConstrained
    residual=x.astype(np.float32).copy();codes=[];books=[]
    for layer in range(layers):
        n=len(x);seed=int(np.random.RandomState(42+layer).randint(0,2**31-1))
        km=KMeansConstrained(n_clusters=256,size_min=max(1,n//256-1),size_max=n//256+1,
              max_iter=100,tol=1e-4,random_state=seed,n_init=3,n_jobs=1)
        ids=km.fit_predict(residual);centers=km.cluster_centers_
        codes.append(ids);books.append(centers);residual-=centers[ids]
    return np.stack(codes,1),books,x-residual

def ghost(x,f,out):
    head,tail,threshold=head_split(f);hc,books,recon=constrained_residual(x[head],4)
    # Same squared Euclidean nearest-head computation, torch float32 in blocks.
    nearest_ids=[];h=F.normalize(torch.from_numpy(x[head]),p=2,dim=1)
    for a in range(0,len(tail),512):
        t=F.normalize(torch.from_numpy(x[tail[a:a+512]]),p=2,dim=1)
        distance=torch.cdist(t,h)
        nearest_ids.extend(distance.argmin(1).tolist())
    nearest_ids=np.array(nearest_ids);tc,tbooks,_=constrained_residual(x[tail]-recon[nearest_ids],2)
    codes=[None]*len(x)
    for i,c in zip(head,hc):codes[int(i)]=c.tolist()
    for i,a,c in zip(tail,hc[nearest_ids],tc):codes[int(i)]=a.tolist()+c.tolist()
    buckets=defaultdict(list)
    for i,c in enumerate(codes):buckets[tuple(c)].append(i)
    for ids in buckets.values():
        for rank,i in enumerate(ids,1):codes[i].append(256+rank)
    sim=x[tail]@x[head].T;negative=np.full((len(x),5),-1,np.int64)
    def lcp(a,b):
        for j,(u,v) in enumerate(zip(a,b)):
            if u!=v:return j
        return min(len(a),len(b))
    for j,t in enumerate(tail):
        rough=np.argsort(-sim[j],kind="stable")[:200]
        ids=sorted(rough,key=lambda k:lcp(codes[int(t)],codes[int(head[k])]))
        negative[t]=head[ids[:5]]
    np.save(out/"ghost-negative-items.npy",negative)
    np.savez_compressed(out/"ghost-codebooks.npz",**{f"head{i}":b for i,b in enumerate(books)},**{f"tail{i}":b for i,b in enumerate(tbooks)},head_ids=head,tail_ids=tail,nearest_head_ids=head[nearest_ids])
    return codes,{"construction":"head 4 residual codes; tail inherits nearest head +2 residual codes; all SID ordinal suffix","head_threshold_strict_gt":threshold}

def onerec(x,out):
    residual=torch.from_numpy(x.copy());rng=np.random.default_rng(42);codes=[];books=[];trace=[];n=len(x);k=256
    caps=np.full(k,n//k);caps[:n%k]+=1
    for layer in range(3):
        cent=residual[rng.choice(n,k,replace=False)].clone();previous=None
        for it in range(100):
            distances=torch.cdist(residual,cent).square().numpy();lab=np.full(n,-1,np.int32);available=np.ones(n,bool)
            for j in range(k):
                ids=np.flatnonzero(available);take=ids[np.argsort(distances[ids,j],kind="stable")[:caps[j]]];lab[take]=j;available[take]=False
            for j in range(k):cent[j]=residual[lab==j].mean(0)
            change=n if previous is None else int((lab!=previous).sum());trace.append({"layer":layer,"iteration":it+1,"changed":change})
            if change==0:break
            previous=lab.copy()
        codes.append(lab);books.append(cent.numpy());residual=residual-cent[lab]
    np.savez_compressed(out/"quantizer.npz",books=np.stack(books))
    (out/"trace.json").write_text(json.dumps(trace,indent=2))
    return np.stack(codes,1).tolist(),{"construction":"balanced sequential residual k-means; 3x256; normalized raw text"}

def classic_rqvae(x,out,epochs=20000):
    from third_party.minionerec import layers
    from sklearn.cluster import KMeans
    def sklearn_init(samples,num_clusters,num_iters=100):
        km=KMeans(n_clusters=num_clusters,max_iter=num_iters).fit(samples.detach().cpu().numpy())
        return torch.from_numpy(km.cluster_centers_).to(samples.device)
    # VQ imports layers.kmeans at module import: replace its binding temporarily.
    from third_party.minionerec import vq
    old=vq.kmeans;vq.kmeans=sklearn_init
    config=dict(in_dim=x.shape[1],num_emb_list=[256]*3,e_dim=32,layers=[512,256,128],beta=.25,
       kmeans_init=True,kmeans_iters=100,sk_epsilons=[0.]*3,sk_iters=50)
    try:
        model=RQVAE(**config);tx=F.normalize(torch.from_numpy(x),dim=1)
        opt=torch.optim.Adagrad(model.parameters(),lr=.4,initial_accumulator_value=.1)
        with torch.no_grad():model(tx[:1024],use_sk=False)
    finally:vq.kmeans=old
    best=math.inf;trace=[]
    for epoch in range(1,epochs+1):
        model.train()
        for ix in torch.randperm(len(tx)).split(1024):
            recon,q,_=model(tx[ix],use_sk=False);loss,_=model.compute_loss(recon,q,xs=tx[ix]);opt.zero_grad();loss.backward();opt.step()
        if epoch==1 or epoch%500==0 or epoch==epochs:
            model.eval()
            with torch.no_grad():recon,_,codes=model(tx,use_sk=False);mse=float(F.mse_loss(recon,tx))
            trace.append({"epoch":epoch,"mse":mse});print("RQ_VAE",epoch,mse,flush=True)
            if mse<best:best=mse;torch.save({"model":model.state_dict(),"config":config,"epoch":epoch},out/"tokenizer.pt")
            (out/"trace.json").write_text(json.dumps(trace,indent=2))
    load_initialized(model,torch.load(out/"tokenizer.pt",weights_only=True)["model"]);model.eval()
    with torch.no_grad():return model.get_indices(tx,use_sk=False).numpy().tolist()

def constrained_split(x,n,f,M):
 # Weighted centroid distortion + lambda * squared frequency imbalance.
 # Lambda scale is prespecified from geometry and TRAIN frequencies, never tuned on test.
 rng=np.random.default_rng(42);cent=x[rng.choice(len(x),M,replace=False)].copy();z=np.argmin(((x[:,None]-cent[None])**2).sum(-1),1)
 for m in range(M):z[m]=m
 scale=max(float((n*((x-np.average(x,axis=0,weights=n))**2).sum(1)).sum()),1e-12)
 lam=scale/max(float(f.sum()**2),1.);trace=[]
 def obj(a):
  cs=np.array([np.average(x[a==m],axis=0,weights=n[a==m]) for m in range(M)]);pop=np.bincount(a,weights=f,minlength=M)
  return float((n*((x-cs[a])**2).sum(1)).sum()+lam*((pop-f.sum()/M)**2).sum())
 val=obj(z);trace.append(val)
 for it in range(100):
  changes=0
  for j in range(len(z)):
   old=z[j]
   if (z==old).sum()==1:continue
   best=old;bestval=val
   for m in range(M):
    if m==old:continue
    a=z.copy();a[j]=m;v=obj(a)
    if v<bestval-1e-12:best,bestval=m,v
   if best!=old:z[j]=best;val=bestval;changes+=1
  trace.append(val)
  if changes==0:break
 assert np.all(np.diff(trace)<=1e-9)
 return z,{'lambda':lam,'objective_trace':trace}


def crab(x,f,out,tiger):
    base=Path(tiger);c=np.array(json.loads((base/"codes.json").read_text()))
    state=torch.load(base/"tokenizer.pt",weights_only=True);model=RQVAE(**state["config"])
    load_initialized(model,state["model"]);model.eval()
    with torch.no_grad():res=model.encoder(F.normalize(torch.from_numpy(x),dim=1)).numpy()
    residuals=[]
    for l,book in enumerate(model.rq.vq_layers):
        residuals.append(res.copy());res=res-book.embedding.weight.detach().numpy()[c[:,l]]
    maps=[{int(t):int(t) for t in np.unique(c[:,l])} for l in range(4)];details=[]
    for l in range(3):
        original=c[:,l].copy();tokens=np.unique(original);pop={int(t):int(f[original==t].sum()) for t in tokens};mean=float(np.mean(list(pop.values())))
        top=sorted(tokens,key=lambda t:(-pop[int(t)],int(t)))[:math.ceil(.1*len(tokens))];nextid=int(c[:,l].max())+1
        for t in top:
            ids=np.flatnonzero(original==t);children=np.unique(c[ids,l+1]);m=min(3,max(1,math.ceil(pop[int(t)]/max(mean,1e-12))),len(children))
            if m<2:continue
            clusters=[ids[c[ids,l+1]==ch] for ch in children]
            means=np.array([residuals[l][ii].mean(0) for ii in clusters]);n=np.array([len(ii) for ii in clusters]);freq=np.array([f[ii].sum() for ii in clusters]);z,diag=constrained_split(means,n,freq,m)
            newtokens=list(range(nextid,nextid+m));nextid+=m
            for ii,a in zip(clusters,z):c[ii,l]=newtokens[int(a)]
            for nt in newtokens:maps[l][nt]=int(t)
            details.append({"layer":l,"parent":int(t),"M":m,"new_tokens":newtokens,**diag})
    (out/"parent-maps.json").write_text(json.dumps(maps));(out/"split-details.json").write_text(json.dumps(details,indent=2))
    return c.tolist(),{"construction":"CRAB mechanism port: 10% popular parent splits, max3, indivisible child groups; suffix unchanged"}

def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=DEFAULT_ROOT);ap.add_argument("--dataset",choices=["industrial","office"],required=True)
    ap.add_argument("--method",choices=["controlled","tiger","rqvae","onerec","ghost","sa2crq","crab"],required=True);ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--epochs",type=int);ap.add_argument("--tiger-artifact",type=Path);ap.add_argument("--threads",type=int,default=2)
    args=ap.parse_args(argv)
    from .cli import new_output
    out=new_output(args.output);torch.set_num_threads(args.threads);torch.manual_seed(42);np.random.seed(42);random.seed(42)
    x=np.load(args.root/"data"/args.dataset/"item_embeddings.npy").astype(np.float32)
    if args.method=="controlled":
        common=make_model(x.shape[1],"serial",3,256);initial={k:t.clone() for k,t in common.state_dict().items() if k.startswith(("encoder.","decoder."))}
        args.output=out;args.epochs=args.epochs or 3000;args.eval_every=50
        for mode,levels,size in [("serial",3,256),("parallel",3,256),("serial",6,16),("parallel",6,16)]:train_one(torch.from_numpy(x),args,mode,levels,size,initial)
        return
    if args.method in ("rqvae","tiger"):
        codes=classic_rqvae(x,out,args.epochs or 20000)
        if args.method=="tiger":
            seen=Counter();full=[]
            for c in codes:full.append(c+[seen[tuple(c)]]);seen[tuple(c)]+=1
            codes=full
        save_codes(out,codes,{"construction":"3x256 RQ-VAE"+(" + stable collision suffix" if args.method=="tiger" else " without suffix")});return
    f=frequency(args.root,args.dataset,len(x))
    if args.method in ("ghost","sa2crq"):
        with np.load(args.root/"artifacts"/args.dataset/"ours/text_features.npz") as z:features=z["item_vectors"][:,:32].copy()
        features/=np.linalg.norm(features,axis=1,keepdims=True).clip(1e-12)
        codes,detail=(ghost if args.method=="ghost" else sa2crq_tokenizer)(features,f,out)
    elif args.method=="onerec":codes,detail=onerec(x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12),out)
    else:codes,detail=crab(x,f,out,args.tiger_artifact or args.root/"artifacts"/args.dataset/"tiger")
    save_codes(out,codes,detail)

if __name__=="__main__":main()
