"""End-to-end frozen inference timing; isolated worker per setting.

Timing starts after input tensors/indexes have been prepared. It includes encoding,
semantic potentials, decoding, matching and item ranking. No cached timing is reused.
"""
import argparse,json,time,platform,resource,subprocess,sys,os,hashlib
from pathlib import Path
import numpy as np
import torch
from .data import DEFAULT_ROOT,read_rows,settings
from .engine import Retriever
from .cli import new_output

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=DEFAULT_ROOT);ap.add_argument("--dataset",choices=["industrial","office"],required=True)
    ap.add_argument("--method",choices=["all","ours","rqvae","tiger","onerec","crab","ghost","sa2crq"],default="all");ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--batch-size",type=int,choices=[1,16],default=1);ap.add_argument("--queries",type=int,default=64);ap.add_argument("--rounds",type=int,default=3);ap.add_argument("--worker",action="store_true");args=ap.parse_args()
    if args.queries<1 or args.rounds<1:ap.error("queries and rounds must be positive")
    out=new_output(args.output)
    if not args.worker:
        methods=settings(args.root)["methods"] if args.method=="all" else [args.method]
        schedule=[(m,i) for m in methods for i in range(args.rounds)];np.random.default_rng(42).shuffle(schedule);results=[]
        for m,i in schedule:
            directory=out/(m+"-round"+str(i+1))
            env=dict(os.environ,OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",PYTHONDONTWRITEBYTECODE="1")
            subprocess.run([sys.executable,"-B","-m","sid_retrieval.benchmark","--worker","--root",str(args.root),"--dataset",args.dataset,"--method",m,"--output",str(directory),"--batch-size",str(args.batch_size),"--queries",str(args.queries),"--rounds","1"],check=True,env=env)
            results.append(json.loads((directory/"measurement.json").read_text()))
            (out/"results.json").write_text(json.dumps(results,indent=2))
        return
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(42)
    rows=read_rows(args.root,args.dataset,"test");ids=np.sort(np.random.default_rng(42).choice(len(rows),min(args.queries,len(rows)),replace=False))
    retriever=Retriever(args.root,args.dataset,args.method);h=retriever.history_tensor([rows[i][1] for i in ids]);batches=list(h.split(args.batch_size))
    # Check output before measuring, but don't demand batch1 tie order match batch16.
    preds=np.concatenate([retriever.rank(*retriever.generate(b)) for b in batches])
    ref=np.load(retriever.artifact/"reference_items.npz")["items"][ids]
    target=np.array([rows[i][2] for i in ids])
    def rank(a):
        hit=a==target[:,None];return np.where(hit.any(1),hit.argmax(1)+1,0)
    rank_mismatch=int((rank(preds)!=rank(ref)).sum())
    if rank_mismatch:raise RuntimeError("Timing aborted: target ranks changed for the requested batch")
    for b in batches[:2]:retriever.rank(*retriever.generate(b))
    durations=[];generation=[];ranking=[]
    for b in batches:
        start=time.perf_counter();p,s=retriever.generate(b);mid=time.perf_counter();retriever.rank(p,s);end=time.perf_counter()
        durations.append(end-start);generation.append(mid-start);ranking.append(end-mid)
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if platform.system()=="Darwin" else 1024)
    record={"dataset":args.dataset,"method":args.method,"beam":retriever.beam,"seed":42,"batch_size":args.batch_size,
      "queries":len(ids),"query_ids":ids.tolist(),"durations_seconds":durations,"generation_seconds":generation,"retrieval_ranking_seconds":ranking,
      "queries_per_second":len(ids)/sum(durations),"batch_p50_ms":float(np.median(durations)*1000),"batch_p95_ms":float(np.quantile(durations,.95)*1000),
      "process_peak_RSS_MiB":rss,"different_item_slots":int((preds!=ref).sum()),"different_target_ranks":rank_mismatch,
      "hardware":{"system":platform.system(),"machine":platform.machine(),"cpu_count":os.cpu_count(),"torch":torch.__version__,"numpy":np.__version__,"threads":2},
      "model_sha256":hashlib.sha256((retriever.artifact/"model.pt").read_bytes()).hexdigest(),"fresh_measurement":True}
    (out/"measurement.json").write_text(json.dumps(record,indent=2));print("MEASURED",args.method,record["queries_per_second"],flush=True)
if __name__=="__main__":main()
