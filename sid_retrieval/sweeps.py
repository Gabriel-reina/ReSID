"""Frozen ablations and one-variable sensitivity; no training or test-based selection."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .data import DEFAULT_ROOT,read_rows,frequency,groups
from .engine import Retriever
from .metrics import evaluate
from .cli import new_output

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=DEFAULT_ROOT);ap.add_argument("--dataset",choices=["industrial","office"],required=True);ap.add_argument("--experiment",choices=["ablation","parameters"],required=True);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--split",choices=["valid","test"]);args=ap.parse_args()
    split=args.split or ("valid" if args.dataset=="office" and args.experiment=="parameters" else "test")
    out=new_output(args.output);torch.set_num_threads(2);torch.manual_seed(42);rows=read_rows(args.root,args.dataset,split);target=np.array([r[2] for r in rows]);users=np.array([r[0] for r in rows]);r=Retriever(args.root,args.dataset)
    history=r.history_tensor([row[1] for row in rows]);cg=groups(frequency(args.root,args.dataset,len(r.codes)));configs={}
    if args.experiment=="ablation":
        configs={"baseline_exact":(0,0,0),"baseline_mask":(0,0,3),"prior_only":(.5,0,3),"semantic_only":(0,.4,3),"full":(.5,.4,3),"full_exact":(.5,.4,0)}
    else:
        for a in [.3,.4,.5,.6,.7]:configs[f"lambda{a:g}-beta0.4-mask3"]=(a,.4,3)
        for b in [.2,.3,.4,.5,.6]:configs[f"lambda0.5-beta{b:g}-mask3"]=(.5,b,3)
        for m in range(5):configs[f"lambda0.5-beta0.4-mask{m}"]=(.5,.4,m)
    results={};cache={}
    for name,(prior,semantic,mask) in configs.items():
        r.prior=prior;r.semantic=semantic;r.max_mask=mask;key=(prior,semantic)
        if key not in cache:
            parts=[r.generate(h) for h in history.split(128)];cache[key]=(np.concatenate([t[0] for t in parts]),np.concatenate([t[1] for t in parts]))
        items=r.rank(*cache[key]);record,ranks=evaluate(items,target,cg[target],cg,users)
        record["parameters"]={"lambda":prior,"beta":semantic,"max_mask":mask};record["split"]=split;results[name]=record
        np.savez_compressed(out/(name+".npz"),items=items,ranks=ranks);(out/"results.json").write_text(json.dumps(results,indent=2));print("COMPLETE",name,flush=True)
    if args.experiment=="ablation" and args.dataset=="office":
        serial=Retriever(args.root,"office","serial_full")
        items=serial.recommend([row[1] for row in rows])
        record,ranks=evaluate(items,target,cg[target],cg,users)
        record["parameters"]={"lambda":.5,"beta":.4,"max_mask":3};record["split"]=split
        results["serial_full"]=record
        np.savez_compressed(out/"serial_full.npz",items=items,ranks=ranks)
        (out/"results.json").write_text(json.dumps(results,indent=2))
        print("COMPLETE serial_full",flush=True)
if __name__=="__main__":main()
