"""Frozen prediction API. recommend() accepts histories, never target labels."""
import json
from pathlib import Path
import numpy as np
import torch
from .model import GenerativeRecommender,ModelConfig
from .data import settings,frequency,histories
from .decoding import (SIDTrie,PaddedSIDTrie,conditional_priors,prefix_groups,
 guided_beam,native_beam,native_items,length_normalized_beam,UserProjection)

def partial_items(codes,paths,scores,max_mask=3,budget=200):
    if not 0<=max_mask<=codes.shape[1]:raise ValueError("Invalid mask count")
    result=np.full((len(paths),budget),-1,np.int32)
    for q in range(len(paths)):
        good=paths[q][np.isfinite(scores[q])]
        if not len(good):continue
        distance=(codes[:,None,:]!=good[None,:,:]).sum(2)
        minimum=distance.min(1);earliest=distance.argmin(1)
        ids=np.flatnonzero(minimum<=max_mask)
        chosen=ids[np.lexsort((ids,earliest[ids],minimum[ids]))[:budget]]
        result[q,:len(chosen)]=chosen
    return result

class Retriever:
    def __init__(self,root,dataset,method="ours",device="cpu",beam=None,
                 prior=None,semantic=None,max_mask=None,artifact=None):
        self.root=Path(root);self.dataset=dataset;self.method=method;self.device=torch.device(device)
        registry=settings(root)
        entries={**registry["datasets"][dataset]["methods"],**registry["datasets"][dataset].get("variants",{})}
        if method not in entries:raise ValueError("Method is not available for the requested dataset")
        self.spec=dict(entries[method])
        if artifact is not None:self.spec["artifact"]=str(artifact)
        self.artifact=self.root/self.spec["artifact"]
        if artifact is not None and self.spec["eos"] is not None:
            self.spec["eos"]=json.loads((self.artifact/"sid_metadata.json").read_text())["EOS"]
        state=torch.load(self.artifact/"model.pt",map_location="cpu",weights_only=True)
        self.model=GenerativeRecommender(ModelConfig(**state["config"]))
        self.model.load_state_dict(state["model"]);self.model.to(self.device).eval()
        self.codes=json.loads((self.artifact/"codes.json").read_text())
        self.beam=self.spec["beam"] if beam is None else beam
        if self.beam<1:raise ValueError("beam must be positive")
        if self.spec["eos"] is None:
            self.codes_array=np.array(self.codes);self.pad=self.model.config.code_size
            self.trie=SIDTrie(dict(enumerate(self.codes)),self.pad,device=self.device)
            self.priors,_=conditional_priors(self.codes_array,frequency(root,dataset,len(self.codes)),self.trie)
            self.priors=[p.to(self.device) for p in self.priors]
            self.inverse,self.counts=prefix_groups(self.codes_array)
            self.inverse=[x.to(self.device) for x in self.inverse];self.counts=[x.to(self.device) for x in self.counts]
            self.prior=self.spec.get("prior_strength",0) if prior is None else prior
            self.semantic=self.spec.get("semantic_strength",0) if semantic is None else semantic
            self.max_mask=self.spec.get("max_mask",0) if max_mask is None else max_mask
            self.projection=UserProjection("linear").to(self.device).eval()
            self.projection.load_state_dict(torch.load(self.artifact/"projection.pt",map_location=self.device,weights_only=True))
            with np.load(self.artifact/"text_features.npz") as z:self.text=torch.from_numpy(z["item_vectors"]).to(self.device)
            self.history_codes=self.codes
        else:
            if any(x is not None for x in (prior,semantic,max_mask)):raise ValueError("Ours-only parameters on baseline")
            self.trie=PaddedSIDTrie(self.codes,self.spec["eos"]);self.pad=self.trie.pad
            self.trie.transitions=[x.to(self.device) for x in self.trie.transitions]
            self.history_codes=self.trie.padded

    def history_tensor(self,raw_histories):
        return histories([(str(i),h,0) for i,h in enumerate(raw_histories)],self.history_codes,self.pad,self.model.config.max_history)

    @torch.no_grad()
    def generate(self,history):
        history=history.to(self.device)
        if self.spec["eos"] is None:
            p,s=guided_beam(self.model,self.projection,self.text,history,self.trie,
                  self.priors,self.inverse,self.counts,self.semantic,self.prior,self.beam)
            return p.detach().cpu().numpy(),s.detach().cpu().numpy()
        if self.spec["decoding"]=="length_normalized":
            p,s,_=length_normalized_beam(self.model,history,self.trie,self.beam)
            return p,s
        p,s=native_beam(self.model,history,self.trie,self.beam)
        return p.detach().cpu().numpy(),s.detach().cpu().numpy()

    def rank(self,paths,scores):
        if self.spec["eos"] is None:return partial_items(self.codes_array,paths,scores,self.max_mask)
        return native_items(paths,scores,self.trie)

    def recommend(self,raw_histories,batch_size=None):
        if not raw_histories:return np.empty((0,200),np.int32)
        batch_size=batch_size or (128 if self.spec["eos"] is None else 16)
        if batch_size<1:raise ValueError("batch_size must be positive")
        h=self.history_tensor(raw_histories);result=[]
        for a in range(0,len(h),batch_size):result.append(self.rank(*self.generate(h[a:a+batch_size])))
        return np.concatenate(result)
