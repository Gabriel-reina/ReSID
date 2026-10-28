"""Render the English seven-chapter tables from anonymous numeric references."""
import argparse,json
from pathlib import Path
from .data import DEFAULT_ROOT
from .cli import new_output

def read(root,name):return json.loads((Path(root)/'references'/name).read_text())
def table(head,rows):
    return ['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows]+['']
def rn(r,g,k):
    m=r['metrics'][g];return f"{m[f'Recall@{k}']*100:.2f} / {m[f'NDCG@{k}']:.5f}"
def cov(r,g,k):
    c=r['coverage'][g][str(k)];return f"{100*c['effective_hit_coverage']:.2f} / {100*c['exposure_catalogue_coverage']:.2f}"
def marked(value,others,precision,second):
    v=round(value,precision);order=sorted(set(round(x,precision) for x in others),reverse=True);s=f'{v:.{precision}f}'
    if v==order[0]:return '**'+s+'**'
    return ''.join(c+'\u0332' for c in s) if second and len(order)>1 and v==order[1] else s

def result_table(cases,ks,coverage_k,highlight=False,second=False):
    rows=[]
    for g,label in [('all','Overall'),('tail','Tail')]:
        for name,r in cases:
            row=[label,name]
            def metric(key,scale,digits):
                v=r['metrics'][g][key]*scale
                return marked(v,[a['metrics'][g][key]*scale for _,a in cases],digits,second) if highlight else f'{v:.{digits}f}'
            for k in ks:row.append(metric(f'Recall@{k}',100,2)+' / '+metric(f'NDCG@{k}',1,5))
            for key in ['effective_hit_coverage','exposure_catalogue_coverage']:
                v=r['coverage'][g][str(coverage_k)][key]*100
                row.append(marked(v,[a['coverage'][g][str(coverage_k)][key]*100 for _,a in cases],2,second) if highlight else f'{v:.2f}')
            rows.append(row)
    return table(['Group','Method']+[f'@{k} R/N' for k in ks]+[f'EHC@{coverage_k} (%)',f'EC@{coverage_k} (%)'],rows)

def render(root=DEFAULT_ROOT):
    d=read(root,'paper-results.json');ks=[50,100,200]
    lines=['# Experimental report: seven chapters','',
        'Industrial motivation tables are intended for the main paper; matching Office motivation tables are intended for the appendix. Main, ablation and motivation results use TEST. Office sensitivity uses VALID.',
        '', 'Seed42. Tail items have1..6 TRAIN next-target events; head items have at least12. Overall includes middle and zero-frequency items. R is item Recall (%); N is NDCG (raw value). EHC is effective hit coverage: distinct correctly retrieved group targets divided by distinct group targets observed in the evaluation split. EC is exposure coverage: distinct group items shown across all queries divided by catalogue group size. Both coverage values are percentages.',
        '', 'The main method uses parallel6x16 SID, beam20, lambda=.5, beta=.4, mask<=3. Comparison methods use beam200. Best values are bold and second-best use combining underlines in main tables; ablation tables bold the best only. Ties follow displayed precision.',
        '', '## Chapter 1. Motivation analyses','']
    for dataset,label in [('industrial','Industrial: main paper'),('office','Office: appendix')]:
        rep=d['analysis']['representations'][dataset];diag=read(root,dataset+'-motivation.json')['splits']['test']
        lines += ['### '+label,'','#### 1. Does offline construction affect tail retrieval?',
        '', 'Compare serial residual quantization and parallel subspace quantization at a fixed configuration using raw-GR exact matching, beam20. The nominal spaces are equal:256^3=16^6=2^24.','']
        cases=[(n,rep[n]) for n in ['serial3x256','parallel3x256','serial6x16','parallel6x16']]
        lines+=table(['SID']+[f'@{k} tail R/N' for k in ks],[[n]+[rn(v,'tail',k) for k in ks] for n,v in cases])
        lines+=table(['SID']+[f'@{k} tail EHC/EC (%)' for k in ks],[[n]+[cov(v,'tail',k) for k in ks] for n,v in cases])
        lines+=['Parallel construction improves tail R@100/200 and effective hit coverage within both configurations on these datasets.','',
        '#### 2. Is usable local information retained after exact SID failure?','',
        'Compare the target to each complete generated beam and take the minimum Hamming distance. Matching coordinates are evaluated within the same complete beam. Reachability is measured before ranking or truncation.','']
        lines+=table(['Maximum mismatches','Head cumulative reach (%)','Tail cumulative reach (%)'],[[m]+[f"{100*diag['radius'][g][m]['target_coverage']:.2f}" for g in ['head','tail']] for m in range(4)])
        masks=d['analysis'][dataset+'_masks'];local=[('Parallel3x256 exact',rep['parallel3x256'])]+[(f'Parallel6x16 mask0..{m}',masks[str(m)]) for m in range(4)]
        lines+=table(['Configuration']+[f'@{k} tail R/N' for k in ks],[[n]+[rn(v,'tail',k) for k in ks] for n,v in local])
        lines+=table(['Configuration']+[f'@{k} tail EHC/EC (%)' for k in ks],[[n]+[cov(v,'tail',k) for k in ks] for n,v in local])
        lines+=['Incomplete long-tail SIDs retain local agreement that can recover item recall and effective coverage through partial matching.','',
        '#### 3. When do exact prefixes and legal tolerant completions disappear?','',
        'Exact survival asks whether the true prefix is still in the free beam. Tolerant survival asks whether any retained prefix can complete to a legal catalogue SID within3 target mismatches.','']
        if dataset=='industrial':survival=read(root,'industrial-representation-and-motivation.json')['baseline_path_survival'];survival={g:v['per_depth_rate'] for g,v in survival.items()}
        else:survival=diag['exact_prefix_survival']
        rows=[]
        for g in ['head','tail']:
            rows.append([g,'Exact target prefix']+[f'{100*x:.2f}' for x in survival[g]])
            rows.append([g,'Legal radius3 completion']+[f'{100*x:.2f}' for x in diag['beam_reachability'][g]['potentially_reachable_per_depth']])
        lines+=table(['Group','Quantity']+[f'Depth{i} (%)' for i in range(1,7)],rows)
        lines+=['Many tail targets lose every legal tolerant completion before decoding ends. A post-beam mask cannot recover evidence outside the retained neighborhood; this motivates protecting relevant tail prefixes during beam search as well.','']
    methods=[('RQ-VAE','rqvae'),('TIGER-style','tiger'),('OneRec-style','onerec'),('CRAB','crab'),('Ghost','ghost'),('SA2CRQ','sa2crq'),('Ours','ours')]
    lines+=['## Chapter 2. Main comparison','']
    for ds in ['industrial','office']:
        cases=[(n,d['main'][ds][k]) for n,k in methods]
        lines+=['### '+ds,'','Main paper: deep-K retrieval and coverage@200.','']+result_table(cases,ks,200,True,True)
        lines+=['Appendix: shallow-K retrieval and coverage@100.','']+result_table(cases,[5,10,20],100,True,True)
    lines+=['## Chapter 3. Office component ablation','',
        'Keep one table in the paper, or convert it to a figure if space is limited. The first three removals reuse the frozen parallel model. Serial Full changes offline construction to serial6x16, uses its corresponding frozen GR and the same TRAIN-only linear projection recipe for8 epochs.','']
    names={'full':'Full','semantic_only':'Full without prior','prior_only':'Full without semantics','full_exact':'Full without mask','serial_full':'Full with serial SID'}
    lines+=result_table([(names[k],d['ablation'][k]) for k in names],ks,200,True)
    lines+=['Parallel Full improves all tail Recall/NDCG cutoffs relative to serial Full; serial Full has stronger overall shallow retrieval and slightly higher overall NDCG.','',
        '## Chapter 4. Office parameter sensitivity','',
        'Use figures in the paper; these tables supply plotting data. All panels use VALID. Vary one parameter around(.5,.4,3), holding the other two fixed.','']
    p=d['parameters']['results']
    panels=[('TRAIN prior lambda: beta=.4, mask<=3',[(str(v),p[f'lambda{v:g}-beta0.4-mask3']) for v in [.3,.4,.5,.6,.7]]),('Semantic beta: lambda=.5, mask<=3',[(str(v),p[f'lambda0.5-beta{v:g}-mask3']) for v in [.2,.3,.4,.5,.6]]),('Maximum masks: lambda=.5, beta=.4',[(str(v),p[f'lambda0.5-beta0.4-mask{v}']) for v in range(5)])]
    for title,cases in panels:lines+=['### '+title,'']+result_table(cases,ks,200)
    lines+=['## Chapter 5. Relative inference efficiency','',
        'Within each dataset, normalize latency and throughput to Ours=1. Latency is batch1 time per request (lower is better); throughput is batch16 requests per second (higher is better). Historical CPU FP32 measurements use identical64 requests and3 serial rounds, including semantic work, matching and sorting.','']
    efficiency=read(root,'efficiency-historical.json')['results'];rows=[]
    for label,key in methods:
        row=[label,20 if key=='ours' else 200]
        for ds in ['industrial','office']:
            v=efficiency[ds];row += [f"{v[key]['1']['mean_ms_per_query_amortized']/v['ours']['1']['mean_ms_per_query_amortized']:.2f}",f"{v[key]['16']['throughput_queries_per_second']/v['ours']['16']['throughput_queries_per_second']:.2f}"]
        rows.append(row)
    lines+=table(['Method','Beam','Industrial latency','Industrial throughput','Office latency','Office throughput'],rows)
    lines+=['The table reports end-to-end CPU inference with each method using its configured SID length, vocabulary and beam budget.','',
        '## Chapter 6. Dataset popularity statistics','',
        'Appendix. Counts use TRAIN next-target events only. Middle and zero-frequency items remain in the catalogue totals.','']
    popularity=read(root,'popularity/results.json');rows=[]
    counts={'industrial':(3686,36259),'office':(3459,38924)}
    for ds in counts:
        groups=popularity[ds];t=groups['tail'];h=groups['head'];rows.append([ds,*counts[ds],f"{100*t['item_share']:.2f}",f"{100*t['train_target_share']:.2f}",f"{100*h['item_share']:.2f}",f"{100*h['train_target_share']:.2f}"])
    lines+=table(['Dataset','Items','TRAIN events','Tail items (%)','Tail events (%)','Head items (%)','Head events (%)'],rows)
    lines+=['Tail items exceed half the catalogue yet receive only about20..22% of TRAIN target events, while roughly one fifth of items receive more than half the events.','',
        '## Chapter 7. Baseline descriptions','', 'Appendix; one sentence per method.','',
        '- **RQ-VAE:** three residual vector-quantization codes without an extra identity suffix; preserve aliases.',
        '- **TIGER-style SID:** append a stable collision-disambiguation token to the same three residual codes.',
        '- **OneRec-style SID:** construct residual k-means codes with balanced per-cluster item allocations.',
        '- **CRAB:** split popular semantic parent tokens and adapt the shared retriever with structural regularization.',
        '- **Ghost:** let tail items inherit head prefixes and add longer suffixes, together with tail unlikelihood training.',
        '- **SA2CRQ:** combine adaptive code lengths with anchored head codebooks and tail extensions.',
        '', 'Baselines use their SID construction and training mechanisms on the shared ordinary GR backbone, with method-specific vocabularies and decoding lengths. Sources and implementation details are in README.md and NOTICE.']
    return '\n'.join(lines)+'\n'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=DEFAULT_ROOT);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    directory=new_output(args.output);(directory/'paper-report.md').write_text(render(args.root))
if __name__=='__main__':main()
