"""TRAIN-only catalogue frequency summaries for the two released categories."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from .data import DEFAULT_ROOT,settings,frequency,groups
from .cli import new_output

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=DEFAULT_ROOT);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    out=new_output(args.output);results={}
    for ds,cfg in settings(args.root)['datasets'].items():
        f=frequency(args.root,ds,cfg['items']);g=groups(f);results[ds]={}
        for label in ['head','tail','middle','zero']:
            selected=g==label
            results[ds][label]={'catalogue_items':int(selected.sum()),'item_share':float(selected.mean()),'train_targets':int(f[selected].sum()),'train_target_share':float(f[selected].sum()/f.sum())}
        with (out/(ds+'-frequencies.csv')).open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['item_id','train_target_count','group'])
            writer.writerows((i,int(count),str(group)) for i,(count,group) in enumerate(zip(f,g)))
    (out/'results.json').write_text(json.dumps(results,indent=2))
    lines=['# TRAIN catalogue popularity','','Groups: head>=12, tail1..6, middle7..11; zero separate. No VALID/TEST labels enter popularity.','','| Dataset | Group | Items | Item share | TRAIN targets | TRAIN share |','|---|---|---|---|---|---|']
    for ds,gg in results.items():
        for g,v in gg.items():lines.append(f"| {ds} | {g} | {v['catalogue_items']} | {100*v['item_share']:.2f}% | {v['train_targets']} | {100*v['train_target_share']:.2f}% |")
    (out/'README.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
