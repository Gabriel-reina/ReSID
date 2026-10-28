"""Input handling. Popularity always counts TRAIN next-item targets only."""
import ast,csv,json
from pathlib import Path
import numpy as np
import torch

DEFAULT_ROOT=Path(__file__).resolve().parent.parent

def settings(root=DEFAULT_ROOT):
    return json.loads((Path(root)/"configs/frozen.json").read_text())

def read_rows(root,dataset,split):
    if dataset not in ("industrial","office") or split not in ("train","valid","test"):
        raise ValueError("Unknown dataset or split")
    with (Path(root)/"data"/dataset/(split+".csv")).open(newline="") as f:
        return [(r["user_id"],ast.literal_eval(r["history_item_id"]),int(r["item_id"])) for r in csv.DictReader(f)]

def frequency(root,dataset,nitems):
    return np.bincount([r[2] for r in read_rows(root,dataset,"train")],minlength=nitems)

def groups(counts):
    f=np.asarray(counts)
    return np.where(f>=12,"head",np.where((f>=1)&(f<=6),"tail",np.where(f==0,"zero","middle")))

def histories(rows,codes,pad,max_history=10):
    levels=len(codes[0]);result=[]
    for _,history,_ in rows:
        ids=history[-max_history:]
        if any(i<0 or i>=len(codes) for i in ids):raise ValueError("Unknown history item")
        result.append([codes[i] for i in ids]+[[pad]*levels]*(max_history-len(ids)))
    return torch.tensor(result,dtype=torch.long)
