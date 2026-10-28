"""Single-target ITEM metrics; SID matches and pool reachability are separate."""
import numpy as np
KS=[5,10,20,50,100,200]
GROUPS=["all","head","tail"]
def coverage_metrics(items,targets,labels,catalogue_groups,users,ks=KS):
    nitems=len(catalogue_groups);support=np.bincount(targets,minlength=nitems);result={g:{} for g in GROUPS}
    ranks=np.zeros(len(targets),np.int16)
    for q,row in enumerate(items):
        where=np.flatnonzero(row==targets[q]);ranks[q]=int(where[0]+1) if len(where) else 0
    for k in ks:
        flat=items[:,:k].reshape(-1);flat=flat[flat>=0];exposure=np.bincount(flat,minlength=nitems)
        good=(ranks>0)&(ranks<=k);hit=np.bincount(targets[good],minlength=nitems)
        ndcg=np.where(good,1/np.log2(np.maximum(ranks,1).astype(float)+1),0)
        itemnd=np.bincount(targets,weights=ndcg,minlength=nitems)
        for g in GROUPS:
            c=np.ones(nitems,bool) if g=='all' else catalogue_groups==g
            rows=np.ones(len(targets),bool) if g=='all' else labels==g
            observed=c&(support>0);ex=exposure[c];total=int(ex.sum());p=ex[ex>0]/max(total,1)
            result[g][str(k)]={
              'samples':int(rows.sum()),'catalogue_items':int(c.sum()),'observed_target_items':int(observed.sum()),
              'Recall':float(good[rows].mean()) if rows.any() else None,'NDCG':float(ndcg[rows].mean()) if rows.any() else None,
              'exposed_unique_items':int((ex>0).sum()),'exposure_catalogue_coverage':float((ex>0).mean()) if c.any() else None,
              'hit_unique_items':int((hit[observed]>0).sum()),'effective_hit_coverage':float((hit[observed]>0).mean()) if observed.any() else None,
              'item_macro_recall':float((hit[observed]/support[observed]).mean()) if observed.any() else None,
              'item_macro_ndcg':float((itemnd[observed]/support[observed]).mean()) if observed.any() else None,
              'exposure_share':float(total/max(len(flat),1)),
              'effective_exposure_items':float(np.exp(-(p*np.log(p)).sum())) if total else 0.,
              'user_hit_coverage':float(len(np.unique(users[good&rows]))/len(np.unique(users[rows]))) if rows.any() else None}
    return result,ranks

def metrics(ranks,labels):
    result={}
    for group in ['all','head','tail','middle','zero']:
        rows=np.ones(len(labels),bool) if group=='all' else labels==group;r=ranks[rows];d={'samples':len(r)}
        for k in KS:
            good=(r>0)&(r<=k);d[f'Recall@{k}']=float(good.mean()) if len(r) else None
            d[f'NDCG@{k}']=float(np.where(good,1/np.log2(np.maximum(r,1).astype(float)+1),0).mean()) if len(r) else None
        result[group]=d
    return result

def evaluate(items,targets,labels,catalogue_groups,users):
    items=np.asarray(items);targets=np.asarray(targets)
    if np.any(items < -1):raise ValueError("Only -1 is a valid padding item")
    if items.ndim!=2 or len(items)!=len(targets):raise ValueError("Prediction shape mismatch")
    for row in items:
        valid=row[row>=0]
        if len(valid)!=len(np.unique(valid)):raise ValueError("Duplicate recommendation")
        if np.any(valid>=len(catalogue_groups)):raise ValueError("Unknown recommended item")
    coverage,ranks=coverage_metrics(items,targets,labels,catalogue_groups,users)
    return {"metrics":metrics(ranks,labels),"coverage":coverage},ranks
