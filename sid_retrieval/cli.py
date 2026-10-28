"""Command line entry points; every experiment requires a new output directory."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from .data import DEFAULT_ROOT,settings,read_rows,frequency,groups
from .metrics import evaluate,KS
from .engine import Retriever

def new_output(path):
    p=Path(path)
    if p.exists():raise FileExistsError("Refusing to overwrite existing output: "+str(p))
    p.mkdir(parents=True,exist_ok=False)
    return p

def compare(actual,expected,prefix=""):
    if isinstance(expected,dict):
        if not isinstance(actual,dict):return [prefix+": expected an object"]
        failures=[]
        for key,value in expected.items():
            if key not in actual:failures.append(prefix+"/"+key+": missing")
            else:failures.extend(compare(actual[key],value,prefix+"/"+key))
        return failures
    if isinstance(expected,list):
        if not isinstance(actual,list) or len(actual)!=len(expected):return [prefix+": list shape differs"]
        return [error for i,(a,b) in enumerate(zip(actual,expected)) for error in compare(a,b,prefix+"/"+str(i))]
    if expected is None:return [] if actual is None else [prefix+": expected null"]
    if isinstance(expected,bool):return [] if type(actual) is bool and actual==expected else [prefix+": boolean differs"]
    if isinstance(expected,(int,float)):
        if not isinstance(actual,(int,float)) or isinstance(actual,bool) or not np.isclose(actual,expected,rtol=0,atol=1e-10,equal_nan=True):return [prefix+": number differs"]
        return []
    return [] if actual==expected else [prefix+": value differs"]

def run_evaluation(args):
    if args.verify and (args.split!="test" or args.limit or any(v is not None for v in [args.beam,args.prior,args.semantic,args.max_mask,args.artifact])):
        raise ValueError("--verify requires the complete frozen TEST protocol")
    if args.artifact is not None and (args.dataset=="all" or args.method=="all"):
        raise ValueError("--artifact requires exactly one dataset and method")
    root=Path(args.root);cfg=settings(root)
    datasets=list(cfg["datasets"]) if args.dataset=="all" else [args.dataset]
    methods=cfg["methods"] if args.method=="all" else [args.method]
    for dataset in datasets:
        available={**cfg["datasets"][dataset]["methods"],**cfg["datasets"][dataset].get("variants",{})}
        if any(method not in available for method in methods):raise ValueError("Method is not available for the requested dataset")
    out=new_output(args.output)
    summary={};failures=[]
    for dataset in datasets:
        rows=read_rows(root,dataset,args.split);targets=np.array([r[2] for r in rows]);users=np.array([r[0] for r in rows]);cg=groups(frequency(root,dataset,cfg["datasets"][dataset]["items"]))
        if args.limit:rows=rows[:args.limit];targets=targets[:args.limit];users=users[:args.limit]
        summary[dataset]={}
        for method in methods:
            print("START",dataset,method,len(rows),flush=True)
            r=Retriever(root,dataset,method,args.device,artifact=args.artifact,beam=args.beam,prior=args.prior,semantic=args.semantic,max_mask=args.max_mask)
            start=time.perf_counter();items=r.recommend([row[1] for row in rows],args.batch_size)
            record,ranks=evaluate(items,targets,cg[targets],cg,users)
            record["elapsed_seconds"]=time.perf_counter()-start
            record["beam"]=r.beam;record["queries"]=len(rows)
            record["inference_configuration"]={"beam":r.beam,"decoding":r.spec["decoding"]}
            if r.spec["eos"] is None:record["inference_configuration"].update({"prior_strength":r.prior,"semantic_strength":r.semantic,"max_mask":r.max_mask})
            name=dataset+"-"+method
            np.savez_compressed(out/(name+"-items.npz"),items=items,ranks=ranks)
            if args.verify:
                if args.split!="test" or args.limit or any(v is not None for v in [args.beam,args.prior,args.semantic,args.max_mask,args.artifact]):raise ValueError("--verify requires the complete frozen TEST protocol")
                ref=json.loads((r.artifact/"reference_metrics.json").read_text())
                delta=compare(record,ref);refitems=np.load(r.artifact/"reference_items.npz")["items"]
                record["reference_check"]={"metric_mismatches":delta,"identical_item_lists":bool(np.array_equal(items,refitems)),"different_item_slots":int((items!=refitems).sum())}
                failures.extend(name+v for v in delta)
                if not np.array_equal(items,refitems):failures.append(name+": item lists differ")
            summary[dataset][method]=record
            (out/"results.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
            print("DONE",name,"R200",record["metrics"]["all"]["Recall@200"],"tail",record["metrics"]["tail"]["Recall@200"],"verified",not failures,flush=True)
    report(summary,out/"results.md")
    (out/"verification.json").write_text(json.dumps({"passed":not failures,"failures":failures,"seed":42,"split":args.split},indent=2))
    if failures:raise SystemExit("Verification failed; inspect verification.json")

def report(summary,path):
    lines=["# Main comparison", "", "Seed 42. The frozen main protocol is Ours beam20, lambda=.5, beta=.4, mask<=3; baselines beam200. This output uses the beams listed per method; exploratory overrides are in results.json.", ""]
    for ds,methods in summary.items():
        lines += ["## "+ds,""]
        for group in ["all","head","tail"]:
            lines += ["### "+group,"","| Method | @5 R/N | @10 R/N | @20 R/N | @50 R/N | @100 R/N | @200 R/N |","|---|---|---|---|---|---|---|"]
            def fmt(method,k,key,scale,precision):
                values={m:round(v["metrics"][group][f"{key}@{k}"]*scale,precision) for m,v in methods.items() if v["metrics"][group][f"{key}@{k}"] is not None}
                if method not in values:return "N/A"
                v=values[method];order=sorted(set(values.values()),reverse=True);s=f"{v:.{precision}f}"
                return "**"+s+"**" if v==order[0] else "".join(c+"\u0332" for c in s) if len(order)>1 and v==order[1] else s
            for method,v in methods.items():lines.append("| "+method+" (beam"+str(v.get("beam",20 if method=="ours" else 200))+") | "+" | ".join(fmt(method,k,"Recall",100,2)+" / "+fmt(method,k,"NDCG",1,5) for k in KS)+" |")
            lines += ["","| Method | K | Effective hit coverage | Exposure catalogue coverage |","|---|---|---|---|"]
            for method,v in methods.items():
                for k in KS:
                    c=v["coverage"][group][str(k)]
                    def percent(x):return "N/A" if x is None else f"{100*x:.2f}%"
                    lines.append(f"| {method} | {k} | {percent(c['effective_hit_coverage'])} | {percent(c['exposure_catalogue_coverage'])} |")
            lines.append("")
    Path(path).write_text("\n".join(lines)+"\n")

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=DEFAULT_ROOT);ap.add_argument("--threads",type=int,default=2)
    sub=ap.add_subparsers(dest="command",required=True)
    ev=sub.add_parser("evaluate");ev.add_argument("--dataset",choices=["all","industrial","office"],default="all");ev.add_argument("--method",choices=["all","ours","rqvae","tiger","onerec","crab","ghost","sa2crq","serial_full"],default="all")
    ev.add_argument("--split",choices=["valid","test"],default="test");ev.add_argument("--output",type=Path,required=True);ev.add_argument("--verify",action="store_true");ev.add_argument("--device",default="cpu");ev.add_argument("--batch-size",type=int);ev.add_argument("--limit",type=int);ev.add_argument("--artifact",type=Path)
    for n,t in [("beam",int),("prior",float),("semantic",float),("max-mask",int)]:ev.add_argument("--"+n,type=t)
    pr=sub.add_parser("predict");pr.add_argument("--dataset",choices=["industrial","office"],required=True);pr.add_argument("--method",default="ours");pr.add_argument("--history",type=int,nargs="*",default=[]);pr.add_argument("--device",default="cpu")
    tr=sub.add_parser("train")
    tr.add_argument("--dataset",choices=["industrial","office"],required=True)
    tr.add_argument("--method",choices=["ours","rqvae","tiger","onerec","crab","ghost","sa2crq","projection"],required=True)
    tr.add_argument("--artifact",type=Path);tr.add_argument("--tiger-artifact",type=Path);tr.add_argument("--output",type=Path,required=True);tr.add_argument("--epochs",type=int)
    args=ap.parse_args();torch.set_num_threads(args.threads);torch.manual_seed(42);np.random.seed(42)
    if args.command=="evaluate":run_evaluation(args)
    elif args.command=="predict":print(json.dumps(Retriever(args.root,args.dataset,args.method,args.device).recommend([args.history])[0].tolist()))
    elif args.command=="train":
        from .training import main as train_main
        argv=["--dataset",args.dataset,"--method",args.method,"--output",str(args.output),"--threads",str(args.threads)]
        for key in ["artifact","tiger_artifact","epochs"]:
            value=getattr(args,key)
            if value is not None:argv.extend(["--"+key.replace("_","-"),str(value)])
        train_main(argv,root=args.root)
