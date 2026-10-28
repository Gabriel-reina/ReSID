"""Baseline-only motivation analyses. Targets are retrospective diagnostics only."""
import argparse,ast,csv,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from .data import DEFAULT_ROOT,settings
from .model import GenerativeRecommender,ModelConfig
from .decoding import SIDTrie,beam_search
from .metrics import metrics,coverage_metrics
from .cli import new_output
KS=[5,10,20,50,100,200]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,x):Path(path).write_text(json.dumps(x,indent=2))
def rows(split):
    with (RAW/(split+".csv")).open() as f:return list(csv.DictReader(f))
def frequency(n):return np.bincount([int(r["item_id"]) for r in rows("train")],minlength=n)
def groups(f):return np.where(f>=12,"head",np.where((f>=1)&(f<=6),"tail",np.where(f==0,"zero","middle")))
def masks(f):
    g=groups(f);return {"all":np.ones(len(f),bool),"head":g=="head","tail":g=="tail"}
def data(split,codes,size):
    rr=rows(split);hist=[]
    for row in rr:
        ids=ast.literal_eval(row["history_item_id"])[-10:];hist.append([codes[i].tolist() for i in ids]+[[size]*codes.shape[1]]*(10-len(ids)))
    return {"history":torch.tensor(hist),"targets":np.array([int(row["item_id"]) for row in rr]),"users":np.array([row["user_id"] for row in rr])}
def load_model(path):
    state=torch.load(path,map_location="cpu",weights_only=True);model=GenerativeRecommender(ModelConfig(**state["config"])).eval();model.load_state_dict(state["model"]);return model,state

def sources():
    cfg=settings(ROOT)["datasets"][DATASET];base={}
    for name,path in cfg["representations"].items():
        artifact=ROOT/(cfg["methods"]["ours"]["artifact"] if path=="ours" else path)
        base[name]={"token":artifact,"checkpoint":artifact/"model.pt"}
    return base

@torch.no_grad()
def trace_beam(model,history,trie):
    # Deliberately no target, popularity, projection, prior, or guidance input.
    memory,padding=model.encode(history);batch=len(history)
    prefix=torch.empty(batch,1,0,dtype=torch.long);score=torch.zeros(batch,1)
    nodes=torch.zeros(batch,1,dtype=torch.long);trace={}
    for depth in range(trie.levels):
        width=prefix.shape[1]
        logits=model.decode(memory.repeat_interleave(width,0),padding.repeat_interleave(width,0),
                            prefix.reshape(batch*width,depth))[:,-1]
        logp=logits.log_softmax(-1).reshape(batch,width,-1)
        children=trie.transitions[depth][nodes.clamp_min(0)]
        candidates=(score[:,:,None]+logp).masked_fill(children<0,-torch.inf)
        score,index=candidates.flatten(1).topk(min(20,width*logits.shape[-1]),dim=1)
        parent,token=index//logits.shape[-1],index%logits.shape[-1]
        prefix=torch.cat([prefix.gather(1,parent[:,:,None].expand(-1,-1,depth)),token[:,:,None]],-1)
        nodes=children.gather(1,parent[:,:,None].expand(-1,-1,children.shape[-1])).gather(2,token[:,:,None]).squeeze(-1)
        trace[f'paths{depth+1}']=prefix.numpy().astype(np.int16)
        trace[f'scores{depth+1}']=score.numpy()
    return trace

def generate(model,dd,codes,out,tag):
    assert not out.exists(),f'Refuse overwrite {out}'
    trie=SIDTrie(dict(enumerate(codes.tolist())),model.config.code_size);parts=[]
    for a in range(0,len(dd['targets']),128):
        h=dd['history'][a:a+128];trace=trace_beam(model,h,trie)
        if a==0:
            # Independent unmodified baseline generation check on every variant/split.
            pp,ss,_=beam_search(model,h,trie,20)
            assert np.array_equal(pp.numpy(),trace[f'paths{codes.shape[1]}'])
            assert np.array_equal(ss.numpy(),trace[f'scores{codes.shape[1]}'])
        parts.append(trace)
        if a%1024==0: print('GENERATE',tag,min(a+128,len(dd['targets'])),flush=True)
    trace={k:np.concatenate([t[k] for t in parts]) for k in parts[0]}
    np.savez_compressed(out,**trace,targets=dd['targets'],users=dd['users'])
    return trace

def ranked_items(codes,paths,scores,max_mask):
    items=np.full((len(paths),200),-1,np.int32)
    for q in range(len(paths)):
        beam=paths[q][np.isfinite(scores[q])]
        d=(codes[:,None,:]!=beam[None,:,:]).sum(2)
        minimum=d.min(1);first=d.argmin(1)
        ids=np.flatnonzero(minimum<=max_mask)
        order=np.lexsort((ids,first[ids],minimum[ids]))
        chosen=ids[order[:200]];items[q,:len(chosen)]=chosen
    return items

def evaluate_items(items,dd,freq):
    cat=groups(freq);labels=cat[dd['targets']]
    cover,rank=coverage_metrics(items,dd['targets'],labels,cat,dd['users'])
    return {'metrics':metrics(rank,labels),'coverage':cover,
            'mean_returned_items':float((items>=0).sum(1).mean())},rank

def support_lookup():
    # Bit k indicates that all kept positions in subset k match one SAME beam.
    return np.array([sum(1<<k for k in range(64) if e&k==k) for e in range(64)],dtype=np.uint64)

def pattern_support(codes,paths,scores,lut):
    equal=(codes[:,None,:]==paths[None,:,:])
    bits=(equal*(1<<np.arange(6))).sum(-1)
    evidence=lut[bits];evidence[:,~np.isfinite(scores)]=0
    support=np.bitwise_or.reduce(evidence,axis=1)
    return support,equal

def different_user_permutation(users,rng):
    p=rng.permutation(len(users))
    for i in np.flatnonzero(users[p]==users):
        if users[p[i]]!=users[i]:continue
        candidates=np.flatnonzero((users[p]!=users[i])&(users!=users[p[i]]))
        j=int(rng.choice(candidates));p[i],p[j]=p[j],p[i]
    assert np.all(users[p]!=users) and len(np.unique(p))==len(p)
    return p

def diagnosis(out):
    torch.set_num_threads(2);torch.manual_seed(42)
    out=new_output(out)
    src=sources()['parallel6x16'];codes=np.array(json.loads((src['token']/'codes.json').read_text()),dtype=np.int16);N=len(codes)
    model,st=load_model(src['checkpoint']);freq=frequency(N);qmasks=masks(freq)
    files=[src['checkpoint'],src['token']/'codes.json',RAW/'item_embeddings.npy']
    files += [RAW/f'{s}.csv' for s in ['train','valid','test']]
    files += [Path(__file__)]
    hashes={str(p):sha(p) for p in files}
    protocol={'dataset':DATASET,'seed':42,'beam':20,'source_checkpoint_epoch':st['epoch'],
      'source_checkpoint':str(src['checkpoint'].relative_to(ROOT)),'SID':'original parallel6x16',
      'generation':'Target-free raw ordinary GR. No priors or semantic guidance. First batch per split compared exactly to unchanged beam_search.',
      'frequency':'TRAIN next-target only: head>=12, tail1..6, middle7..11, zero separately',
      'fixed_pattern_selection':'VALID tail uncapped coverage, tie smaller mean pool then lower integer pattern. Freeze before TEST.',
      'controls':'Exact misses only; equal-size random expansion preserving exact pool. 20 cross-user bijections excluding same user; expected uniform subsampling to per-query minimum budget.',
      'equal_budget_patterns':'For every query/m, use minimum candidate count over all same-m patterns. Coverage expectation = hit * shared_budget / pattern_pool_size.',
      'input_hashes':{(str(Path(k).relative_to(ROOT)) if Path(k).is_relative_to(ROOT) else Path(k).name):v for k,v in hashes.items()}}
    dump(out/'manifest.json',protocol)
    lut=support_lookup();kept=np.arange(64,dtype=np.uint64)
    pop=np.array([bin(k).count('1') for k in range(64)])
    pats={m:np.flatnonzero(pop==6-m) for m in range(7)}
    near=[np.flatnonzero((codes!=c).sum(1)<=3) for c in codes]
    prefixids=[(codes[:,:d]*(16**np.arange(d))).sum(1) for d in range(1,7)]
    summary={'protocol':protocol,'splits':{},'catalogue_position_info':{}}
    for g,ix in qmasks.items():
        summary['catalogue_position_info'][g]=[]
        for d in range(6):
            c=np.bincount(codes[:,d],minlength=16)
            summary['catalogue_position_info'][g].append({'mean_single_position_bucket':float(c[codes[ix,d]].mean()),
                'median_single_position_bucket':float(np.median(c[codes[ix,d]])),'used_tokens':int((c>0).sum())})
    chosen=None;audit=[]
    for split in ['valid','test']:
        start=time.perf_counter();dd=data(split,codes,16)
        trace=generate(model,dd,codes,out/f'{split}-baseline-trace.npz',split)
        paths=trace['paths6'];scores=trace['scores6'];target=dd['targets'];users=dd['users'];n=len(target);qm=masks(freq[target])
        counts=np.empty((n,64),np.int32);support_target=np.empty(n,np.uint64)
        distance=np.empty(n,np.int8);sizes=np.empty((n,7),np.int32)
        top1=np.empty((n,6),bool);anybeam=np.empty((n,6),bool)
        nearest_error=np.empty((n,6),np.float32);mandatory=np.empty((n,6),bool)
        reach=np.empty((n,6),bool);exactreach=np.empty((n,6),bool)
        for q in range(n):
            support,equal=pattern_support(codes,paths[q],scores[q],lut)
            counts[q]=np.unpackbits(support.astype('<u8').view(np.uint8).reshape(-1,8),axis=1,bitorder='little').sum(0)
            support_target[q]=support[target[q]]
            actual=(~equal).sum(-1);valid=np.isfinite(scores[q]);actual[:,~valid]=99
            best=actual.min(1);distance[q]=best[target[q]];sizes[q]=np.bincount(best,minlength=7)[:7].cumsum()
            eq=equal[target[q]];nearest=(actual[target[q]]==distance[q])&valid
            top1[q]=eq[np.flatnonzero(valid)[0]];anybeam[q]=eq[valid].any(0)
            nearest_error[q]=(~eq[nearest]).mean(0);mandatory[q]=(~eq[nearest]).all(0)
            for d in range(1,7):
                beam=trace[f'paths{d}'][q];finite=np.isfinite(trace[f'scores{d}'][q])
                bid=(beam[finite]*(16**np.arange(d))).sum(1)
                reach[q,d-1]=np.isin(prefixids[d-1][near[target[q]]],bid).any()
                exactreach[q,d-1]=((beam[finite]==codes[target[q],:d]).all(1)).any()
            if q in [0,1,123,n-1]:
                # Independent Boolean enumeration of all 64 kept subsets.
                for k in range(64):
                    positions=[d for d in range(6) if k&(1<<d)]
                    brute=(codes[:,None,positions]==paths[q][None,:,positions]).all(-1) if False else None
                    eq_brute=(codes[:,None,:]==paths[q][None,:,:])
                    hit=eq_brute[:,:,positions].all(-1)[:,valid].any(1)
                    assert int(hit.sum())==counts[q,k]
                    assert bool(hit[target[q]])==bool((int(support_target[q])>>k)&1)
                audit.append({'split':split,'query':q,'all64_patterns_verified':True})
            if q%1000==0:print('DIAGNOSE',split,q,flush=True)
        hit_subsets=((support_target[:,None]>>kept)&np.uint64(1)).astype(bool)
        for m in range(7):assert np.array_equal(hit_subsets[:,pats[m]].any(1),distance<=m)
        assert np.all(np.diff(reach.astype(int),axis=1)<=0)
        assert np.all(np.diff(exactreach.astype(int),axis=1)<=0)
        assert np.array_equal(reach[:,-1],distance<=3)
        assert np.array_equal(exactreach[:,-1],distance==0)
        assert np.array_equal(counts[:,63],sizes[:,0])
        if split=='valid':
            chosen={str(m):int(min(pats[m],key=lambda k:(-hit_subsets[qm['tail'],k].mean(),counts[qm['tail'],k].mean(),k))) for m in [1,2,3]}
            dump(out/'frozen-fixed-patterns.json',chosen)
        assert chosen==json.loads((out/'frozen-fixed-patterns.json').read_text())
        items=ranked_items(codes,paths,scores,3);serving,ranks=evaluate_items(items,dd,freq)
        exactitems=ranked_items(codes,paths,scores,0);exactserving,exactranks=evaluate_items(exactitems,dd,freq)
        np.savez_compressed(out/f'{split}-baseline-items.npz',items=items,ranks=ranks,exact_items=exactitems,exact_ranks=exactranks,targets=target)
        result={'positions':{},'mask_patterns':{},'radius':{},'beam_reachability':{},'exact_prefix_survival':{},
                'mask_ceiling_cohorts':{},'controls':{},'equal_budget_patterns':{},'mask3_item_evaluation':serving,'exact_item_evaluation':exactserving}
        for g,ix in qm.items():
            result['positions'][g]={'queries':int(ix.sum()),'top1_match_rate':top1[ix].mean(0).tolist(),
              'anybeam_position_match_rate':anybeam[ix].mean(0).tolist(),
              'nearest_beam_tie_average_mismatch_rate':nearest_error[ix].mean(0).tolist(),
              'nearest_beam_mandatory_mismatch_rate':mandatory[ix].mean(0).tolist(),
              'all6_individually_present_rate':float(anybeam[ix].all(1).mean())}
            result['mask_patterns'][g]={};result['equal_budget_patterns'][g]={}
            for m in [1,2,3]:
                pp=pats[m];rr=hit_subsets[ix][:,pp].mean(0);cc=counts[ix][:,pp].mean(0);k=chosen[str(m)]
                result['mask_patterns'][g][str(m)]={'patterns':[{'pattern':int(kk),'masked':[d+1 for d in range(6) if not kk&(1<<d)],
                    'target_coverage':float(r),'mean_candidates':float(c)} for kk,r,c in zip(pp,rr,cc)],
                    'fixed_coverage':float(hit_subsets[ix,k].mean()),'fixed_mean_candidates':float(counts[ix,k].mean()),
                    'validation_selected_fixed_pattern':k,'union_coverage':float((distance[ix]<=m).mean()),
                    'union_mean_candidates':float(sizes[ix,m].mean()),'min_fixed_coverage':float(rr.min()),'max_fixed_coverage_diagnostic_only':float(rr.max())}
                budget=counts[:,pp].min(1)
                expected=np.divide(hit_subsets[:,pp]*budget[:,None],counts[:,pp],
                    out=np.zeros((n,len(pp)),float),where=counts[:,pp]>0)
                er=expected[ix].mean(0);assert np.all(er<=rr+1e-12)
                result['equal_budget_patterns'][g][str(m)]={'mean_shared_budget':float(budget[ix].mean()),
                   'min_expected_coverage':float(er.min()),'max_expected_coverage':float(er.max()),
                   'pattern_expectations':{str(int(k)):float(v) for k,v in zip(pp,er)}}
            result['radius'][g]=[{'m':m,'target_coverage':float((distance[ix]<=m).mean()),'mean_candidates':float(sizes[ix,m].mean())} for m in range(7)]
            result['exact_prefix_survival'][g]=exactreach[ix].mean(0).tolist()
            result['beam_reachability'][g]={'potentially_reachable_per_depth':reach[ix].mean(0).tolist()}
            hit=(ranks>0)&(ranks<=200)
            result['mask_ceiling_cohorts'][g]={'queries':int(ix.sum()),'hit_at200':int((ix&hit).sum()),
                'pool_present_not_hit_at200':int((ix&~hit&(distance<=3)).sum()),
                'pool_absent_for_every_mask_policy_max3':int((ix&(distance>3)).sum())}
            miss=ix&(distance>0)
            result['controls'][g]={'exact_miss_queries':int(miss.sum()),
                'local_pool_recovery_among_exact_misses':float((distance[miss]<=3).mean()),
                'same_size_random_expansion_expected_recovery':float(((sizes[:,3]-sizes[:,0])[miss]/(N-sizes[miss,0])).mean())}
        rng=np.random.default_rng(42);perm_results={g:[] for g in qm}
        for iteration in range(20):
            perm=different_user_permutation(users,rng)
            d=(paths[perm]!=codes[target,None,:]).sum(-1);d[~np.isfinite(scores[perm])]=99
            other_hit=d.min(1)<=3;a=sizes[:,3];b=a[perm];budget=np.minimum(a,b)
            local=(distance<=3)*budget/a;other=other_hit*budget/b
            for g,ix in qm.items():
                miss=ix&(distance>0)
                perm_results[g].append({'local_equal_budget_expected_coverage':float(local[miss].mean()),
                    'shuffled_equal_budget_expected_coverage':float(other[miss].mean()),'mean_shared_budget':float(budget[miss].mean())})
        for g,rr in perm_results.items():
            result['controls'][g]['different_user_permutations']={k:{'mean':float(np.mean([r[k] for r in rr])),
                 'std_across20_permutations':float(np.std([r[k] for r in rr]))} for k in rr[0]}
        np.savez_compressed(out/f'{split}-position-mask-evidence.npz',candidate_counts=counts,target_subset_support=support_target,
            minimum_distance=distance,radius_sizes=sizes,top1_position_match=top1,anybeam_position_match=anybeam,
            nearest_mismatch_fraction=nearest_error,mandatory_mismatch=mandatory,beam_neighborhood_reachable=reach,
            exact_prefix_survival=exactreach,targets=target,frequency=freq)
        result['seconds']=time.perf_counter()-start;summary['splits'][split]=result
        dump(out/f'{split}-summary.json',result);dump(out/'summary.json',summary)
        print('SPLIT_COMPLETE',split,flush=True)
    # Counterexample: coordinatewise existence across different beams does not imply joint matching.
    toy=np.ones((3,6),int)
    for i in range(3):toy[i,2*i:2*i+2]=0
    assert (toy==0).any(0).all() and ((toy==0).sum(1)<3).all()
    summary['verification']={'source_hashes_unchanged':all(sha(p)==h for p,h in hashes.items()),
        'all7_radii_equal_same_beam_subset_unions_every_query':True,'neighborhood_survival_monotone':True,
        'final_neighborhood_survival_equals_radius3':True,'final_exact_survival_equals_radius0':True,
        'generation_equals_unmodified_beam_search_first_batch_both_splits':True,'independent_boolean_subset_checks':audit,
        'cross_beam_stitching_counterexample_verified':True,'fixed_mask_selection_valid_only':True}
    assert summary['verification']['source_hashes_unchanged']
    dump(out/'summary.json',summary);dump(out/'independent-audit.json',summary['verification'])
    print('DIAGNOSIS_COMPLETE',flush=True)


def representation(out):
    out=new_output(out);results={}
    for name,src in sources().items():
        directory=out/name;directory.mkdir();codes=np.array(json.loads((src["token"]/"codes.json").read_text()))
        model,state=load_model(src["checkpoint"]);dd=data("test",codes,model.config.code_size)
        trace=generate(model,dd,codes,directory/"trace.npz",name)
        items=ranked_items(codes,trace[f"paths{codes.shape[1]}"],trace[f"scores{codes.shape[1]}"],0)
        result,ranks=evaluate_items(items,dd,frequency(len(codes)))
        unique,counts=np.unique(codes,axis=0,return_counts=True)
        result["SID"]={"unique_sids":len(unique),"extra_collision_rate":1-len(unique)/len(codes),"collision_participation_rate":float(counts[counts>1].sum()/len(codes)),"max_aliases":int(counts.max())}
        result["actual_GR_code_size"]=model.config.code_size;result["epoch"]=state["epoch"]
        result["mask_expansion"]={"0":{k:v for k,v in result.items() if k in ["metrics","coverage"]}}
        for m in range(1,(2 if codes.shape[1]==3 else 3)+1):
            expanded=ranked_items(codes,trace[f"paths{codes.shape[1]}"],trace[f"scores{codes.shape[1]}"],m)
            record,rr=evaluate_items(expanded,dd,frequency(len(codes)));result["mask_expansion"][str(m)]=record
            np.savez_compressed(directory/(f"mask{m}-items.npz"),items=expanded,ranks=rr)
        np.savez_compressed(directory/"items.npz",items=items,ranks=ranks);results[name]=result
        dump(out/"results.json",results)

def main():
    global ROOT,RAW,DATASET
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=DEFAULT_ROOT);ap.add_argument("--dataset",choices=["industrial","office"],required=True)
    ap.add_argument("--analysis",choices=["representation","diagnosis"],required=True);ap.add_argument("--output",type=Path,required=True);args=ap.parse_args()
    ROOT=args.root.resolve();DATASET=args.dataset;RAW=ROOT/"data"/DATASET
    torch.set_num_threads(2);torch.manual_seed(42)
    (representation if args.analysis=="representation" else diagnosis)(args.output)

if __name__=="__main__":main()
