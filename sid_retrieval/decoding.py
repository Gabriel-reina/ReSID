"""Catalogue search kernels adapted without changing CPU ranking semantics."""
from collections import defaultdict
import numpy as np
import torch
from torch import nn
class SIDTrie:
    def __init__(self, sids, code_size, device='cpu'):
        self.levels = len(next(iter(sids.values())))
        self.leaves = defaultdict(list)
        prefixes = [{()}] + [set() for _ in range(self.levels)]
        for item, code in sids.items():
            self.leaves[tuple(code)].append(item)
            for depth in range(1, self.levels + 1):
                prefixes[depth].add(tuple(code[:depth]))
        nodes = [{key: i for i, key in enumerate(sorted(keys))} for keys in prefixes]
        self.transitions = []
        for depth in range(self.levels):
            transition = torch.full((len(nodes[depth]), code_size), -1, dtype=torch.long)
            for key, child in nodes[depth + 1].items():
                transition[nodes[depth][key[:-1]], key[-1]] = child
            self.transitions.append(transition.to(device))
        for items in self.leaves.values():
            items.sort()

@torch.no_grad()
def beam_search(model, history, trie, beam_size=20, target=None):
    memory, padding = model.encode(history)
    batch, device = len(history), history.device
    prefix = torch.empty(batch, 1, 0, dtype=torch.long, device=device)
    score = torch.zeros(batch, 1, device=device)
    nodes = torch.zeros(batch, 1, dtype=torch.long, device=device)
    survives = []
    for depth in range(trie.levels):
        width = prefix.shape[1]
        mem = memory.repeat_interleave(width, dim=0)
        pad = padding.repeat_interleave(width, dim=0)
        logits = model.decode(mem, pad, prefix.reshape(batch * width, depth))[:, -1]
        # Preserve original model probabilities; masking does not renormalize valid children.
        logp = logits.log_softmax(-1).reshape(batch, width, -1)
        next_nodes = trie.transitions[depth][nodes.clamp_min(0)]
        candidates = (score[:, :, None] + logp).masked_fill(next_nodes < 0, -torch.inf)
        count = min(beam_size, candidates.shape[1] * candidates.shape[2])
        score, indices = candidates.flatten(1).topk(count, dim=1)
        parent, token = indices // logits.shape[-1], indices % logits.shape[-1]
        old_prefix = prefix.gather(1, parent[:, :, None].expand(-1, -1, depth))
        prefix = torch.cat([old_prefix, token[:, :, None]], -1)
        nodes = next_nodes.gather(1, parent[:, :, None].expand(-1, -1, next_nodes.shape[-1])).gather(2, token[:, :, None]).squeeze(-1)
        if target is not None:
            survives.append(((prefix == target[:, None, :depth + 1]).all(-1) & score.isfinite()).any(1))
    return prefix, score, torch.stack(survives, 1) if target is not None else None
class PaddedSIDTrie:
    def __init__(self,codes,eos):
        self.eos=eos;self.vocab=eos+1;self.pad=self.vocab;self.levels=max(map(len,codes))+1;self.codes=codes;self.leaves=defaultdict(list)
        self.padded=[c+[eos]+[self.pad]*(self.levels-len(c)-1) for c in codes];prefix=[{()}]+[set() for _ in range(self.levels)]
        for item,c in enumerate(self.padded):
            self.leaves[tuple(c)].append(item)
            for d in range(1,self.levels+1):prefix[d].add(tuple(c[:d]))
        lookup=[{p:i for i,p in enumerate(sorted(ps))} for ps in prefix];self.transitions=[]
        for d in range(self.levels):
            tr=torch.full((len(lookup[d]),self.vocab+1),-1,dtype=torch.long)
            for c,i in lookup[d+1].items():tr[lookup[d][c[:-1]],c[-1]]=i
            self.transitions.append(tr)

@torch.no_grad()
def native_beam(model,history,trie,beam):
    memory,padding=model.encode(history);batch=len(history);prefix=torch.empty(batch,1,0,dtype=torch.long,device=history.device);score=torch.zeros(batch,1,device=history.device);nodes=torch.zeros(batch,1,dtype=torch.long,device=history.device)
    for d,tr in enumerate(trie.transitions):
        width=prefix.shape[1];logits=model.decode(memory.repeat_interleave(width,0),padding.repeat_interleave(width,0),prefix.reshape(batch*width,d))[:,-1].reshape(batch,width,-1)
        nxt=tr[nodes.clamp_min(0)];step=torch.full((batch,width,trie.vocab+1),-torch.inf,device=history.device);step[:,:,:trie.vocab]=logits.log_softmax(-1)
        # A completed EOS path is carried forward with zero additional score;
        # PAD is never generated probabilistically or used as another EOS.
        step[:,:,trie.pad]=torch.where(nxt[:,:,trie.pad]>=0,0.,-torch.inf)
        candidates=(score[:,:,None]+step).masked_fill(nxt<0,-torch.inf);k=min(beam,candidates.shape[1]*candidates.shape[2]);score,index=candidates.flatten(1).topk(k,dim=1);source=index//(trie.vocab+1);token=index%(trie.vocab+1)
        prefix=torch.cat([prefix.gather(1,source[:,:,None].expand(-1,-1,d)),token[:,:,None]],-1);nodes=nxt.gather(1,source[:,:,None].expand(-1,-1,trie.vocab+1)).gather(2,token[:,:,None]).squeeze(-1)
    return prefix,score

def native_items(paths,scores,trie):
    pred=np.full((len(paths),200),-1,np.int32)
    for q,(p,s) in enumerate(zip(paths,scores)):
        result=[]
        for c,sc in zip(p,s):
            if np.isfinite(sc):result.extend(trie.leaves[tuple(c.tolist())])
        assert len(result)==len(set(result));pred[q,:min(200,len(result))]=result[:200]
    return pred
@torch.no_grad()
def length_normalized_beam(model,history,trie,beam):
    memory,padding=model.encode(history);batch=len(history);prefix=torch.empty(batch,1,0,dtype=torch.long,device=history.device);scores=torch.zeros(batch,1,device=history.device);nodes=torch.zeros(batch,1,dtype=torch.long,device=history.device);completed=[[] for _ in range(batch)]
    for d,transition in enumerate(trie.transitions):
        width=prefix.shape[1];logits=model.decode(memory.repeat_interleave(width,0),padding.repeat_interleave(width,0),prefix.reshape(batch*width,d))[:,-1].reshape(batch,width,-1);nxt=transition[nodes.clamp_min(0),:trie.vocab]
        candidates=(scores[:,:,None]+logits.log_softmax(-1)).masked_fill(nxt<0,-torch.inf);top=min(2*beam,candidates.shape[1]*candidates.shape[2]);values,index=candidates.flatten(1).topk(top,dim=1)
        next_prefix=torch.full((batch,beam,d+1),trie.pad,dtype=torch.long,device=history.device);next_score=torch.full((batch,beam),-torch.inf,device=history.device);next_node=torch.full((batch,beam),-1,dtype=torch.long,device=history.device)
        for q in range(batch):
            active=0
            for rank,(value,flat) in enumerate(zip(values[q].tolist(),index[q].tolist())):
                if not np.isfinite(value):continue
                parent,token=divmod(flat,trie.vocab);seq=prefix[q,parent].tolist()+[token]
                if token==trie.eos:
                    # Mirrors HF's candidate-rank rule for completed hypotheses.
                    if rank<beam:completed[q].append((value/len(seq),value,seq))
                elif active<beam:
                    next_prefix[q,active]=torch.tensor(seq,device=history.device);next_score[q,active]=value;next_node[q,active]=nxt[q,parent,token];active+=1
                if active>=beam:break
            completed[q]=sorted(completed[q],key=lambda x:(-x[0],x[2]))[:beam]
        prefix,scores,nodes=next_prefix,next_score,next_node
        if not torch.isfinite(scores).any():break
    paths=np.full((batch,beam,trie.levels),trie.pad,np.int64);norm=np.full((batch,beam),-np.inf,np.float32);raw=np.full_like(norm,-np.inf)
    for q,hyp in enumerate(completed):
        for k,(n,s,seq) in enumerate(hyp):paths[q,k,:len(seq)]=seq;norm[q,k]=n;raw[q,k]=s
    return paths,norm,raw
def conditional_priors(codes,frequency,trie):
    priors=[];statistics=[]
    for depth,transition in enumerate(trie.transitions):
        prefixes=sorted(set(tuple(c[:depth]) for c in codes));lookup={p:i for i,p in enumerate(prefixes)}
        counts=np.zeros(tuple(transition.shape),np.float64)
        for code,f in zip(codes,frequency):counts[lookup[tuple(code[:depth])],code[depth]]+=int(f)
        # Smooth all sixteen token classes, including Trie-forbidden children.
        # This makes logit adjustment a standard full-vocabulary distribution.
        prior=(counts+1)/(counts.sum(1,keepdims=True)+counts.shape[1])
        assert np.allclose(prior.sum(1),1.)
        priors.append(torch.from_numpy(np.log(prior).astype(np.float32)))
        statistics.append({'depth':depth,'prefix_nodes':len(prefixes),'zero_train_prefixes':int((counts.sum(1)==0).sum()),'count_sum':int(counts.sum())})
    return priors,statistics

@torch.no_grad()
def corrected_beam(model,history,trie,priors,strength,beam_size=20):
    memory,padding=model.encode(history);batch=len(history);prefix=torch.empty(batch,1,0,dtype=torch.long,device=history.device);score=torch.zeros(batch,1,device=history.device);nodes=torch.zeros(batch,1,dtype=torch.long,device=history.device)
    for depth in range(trie.levels):
        width=prefix.shape[1];logits=model.decode(memory.repeat_interleave(width,0),padding.repeat_interleave(width,0),prefix.reshape(batch*width,depth))[:,-1]
        next_nodes=trie.transitions[depth][nodes.clamp_min(0)]
        # Normalize all sixteen adjusted logits, then apply original Trie mask.
        # Do NOT renormalize only allowed children; lambda zero is exactly original.
        adjusted=logits.reshape(batch,width,-1)
        if strength!=0:adjusted=adjusted-strength*priors[depth][nodes.clamp_min(0)]
        step=adjusted.log_softmax(-1)
        candidates=(score[:,:,None]+step).masked_fill(next_nodes<0,-torch.inf);count=min(beam_size,candidates.shape[1]*candidates.shape[2]);score,indices=candidates.flatten(1).topk(count,dim=1);parent,token=indices//logits.shape[-1],indices%logits.shape[-1]
        previous=prefix.gather(1,parent[:,:,None].expand(-1,-1,depth));prefix=torch.cat([previous,token[:,:,None]],-1);nodes=next_nodes.gather(1,parent[:,:,None].expand(-1,-1,next_nodes.shape[-1])).gather(2,token[:,:,None]).squeeze(-1)
    return prefix,score
def prefix_groups(codes):
    inverse=[];counts=[]
    for depth in range(codes.shape[1]+1):
        prefixes=sorted(set(tuple(c[:depth]) for c in codes));lookup={p:i for i,p in enumerate(prefixes)}
        index=np.array([lookup[tuple(c[:depth])] for c in codes]);inverse.append(torch.from_numpy(index));counts.append(torch.from_numpy(np.bincount(index).astype(np.float32)))
    return inverse,counts

@torch.no_grad()
def completion_potentials(query,text,inverse,counts):
    # Same cosine temperature as frozen semantic CE training, no new temperature fit.
    score=query@text.T/.07
    # Per-user constant shift cancels in Phi(child)-Phi(parent).
    score=score-score.max(1,keepdim=True).values;weight=score.exp();result=[]
    for index,count in zip(inverse,counts):
        mass=torch.zeros(len(query),len(count),dtype=score.dtype,device=score.device)
        mass.scatter_add_(1,index[None].expand(len(query),-1),weight)
        phi=(mass/count[None]).clamp_min(1e-30).log()
        assert torch.isfinite(phi).all();result.append(phi)
    return result

@torch.no_grad()
def guided_beam(model,projection,text,history,trie,priors,inverse,counts,beta,strength=.5,beam_size=20):
    if beta==0:return corrected_beam(model,history,trie,priors,strength,beam_size)
    memory,padding=model.encode(history);w=(~padding).float()
    query=projection((memory*w[:,:,None]).sum(1)/w.sum(1)[:,None])
    potential=completion_potentials(query,text,inverse,counts)
    batch=len(history);prefix=torch.empty(batch,1,0,dtype=torch.long,device=history.device);score=torch.zeros(batch,1,device=history.device);nodes=torch.zeros(batch,1,dtype=torch.long,device=history.device)
    for depth in range(trie.levels):
        width=prefix.shape[1];logits=model.decode(memory.repeat_interleave(width,0),padding.repeat_interleave(width,0),prefix.reshape(batch*width,depth))[:,-1].reshape(batch,width,-1)
        next_nodes=trie.transitions[depth][nodes.clamp_min(0)]
        step=(logits-strength*priors[depth][nodes.clamp_min(0)]).log_softmax(-1)
        parent=potential[depth].gather(1,nodes.clamp_min(0))
        child=potential[depth+1].gather(1,next_nodes.clamp_min(0).reshape(batch,-1)).reshape_as(next_nodes)
        candidates=(score[:,:,None]+step+beta*(child-parent[:,:,None])).masked_fill(next_nodes<0,-torch.inf)
        count=min(beam_size,candidates.shape[1]*candidates.shape[2]);score,index=candidates.flatten(1).topk(count,dim=1);source,token=index//logits.shape[-1],index%logits.shape[-1]
        prefix=torch.cat([prefix.gather(1,source[:,:,None].expand(-1,-1,depth)),token[:,:,None]],-1)
        nodes=next_nodes.gather(1,source[:,:,None].expand(-1,-1,next_nodes.shape[-1])).gather(2,token[:,:,None]).squeeze(-1)
    return prefix,score
class UserProjection(nn.Module):
 def __init__(self,kind):
  super().__init__();self.net=nn.Linear(128,128) if kind=='linear' else nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.Linear(128,128))
 def forward(self,x):return nn.functional.normalize(self.net(x),dim=-1)
