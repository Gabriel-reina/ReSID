import ast,json,tempfile,unittest,inspect
from pathlib import Path
import numpy as np
import torch
from sid_retrieval.data import DEFAULT_ROOT,frequency,groups,settings,read_rows
from sid_retrieval.engine import Retriever,partial_items
from sid_retrieval.decoding import (SIDTrie,PaddedSIDTrie,native_beam,native_items,beam_search,
 corrected_beam,conditional_priors,completion_potentials,prefix_groups)
from sid_retrieval.model import GenerativeRecommender,ModelConfig
from sid_retrieval.cli import new_output,compare
from sid_retrieval.metrics import evaluate

class ReleaseTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(2);torch.manual_seed(42)
 def test_python_parse(self):
  for base in ['sid_retrieval','third_party','tests']:
   for p in (DEFAULT_ROOT/base).rglob('*.py'):ast.parse(p.read_text())
 def test_fixed_groups(self):
  self.assertEqual(groups(np.array([0,1,6,7,11,12,100])).tolist(),['zero','tail','tail','middle','middle','head','head'])
 def test_new_output_refuses_overwrite(self):
  with tempfile.TemporaryDirectory() as d:
   p=new_output(Path(d)/'new');(p/'keep').write_text('keep')
   with self.assertRaises(FileExistsError):new_output(p)
   self.assertEqual((p/'keep').read_text(),'keep')
 def test_partial_matching_keeps_aliases(self):
  codes=np.array([[0,0],[0,0],[0,1],[1,0]])
  paths=np.array([[[0,0],[1,0]]]);scores=np.array([[0.,-1.]])
  self.assertEqual(partial_items(codes,paths,scores,0)[0,:3].tolist(),[0,1,3])
  self.assertEqual(partial_items(codes,paths,scores,1)[0,:4].tolist(),[0,1,3,2])
 def test_partial_invalid_and_empty(self):
  c=np.array([[0,0]]);p=np.zeros((1,1,2),int);s=np.array([[-np.inf]])
  self.assertTrue((partial_items(c,p,s,1)==-1).all())
  with self.assertRaises(ValueError):partial_items(c,p,s,3)
 def test_target_free_api(self):
  self.assertEqual(list(inspect.signature(Retriever.recommend).parameters),['self','raw_histories','batch_size'])
  self.assertEqual(list(inspect.signature(Retriever.generate).parameters),['self','history'])
 def test_history_uses_only_histories(self):
  r=Retriever(DEFAULT_ROOT,'industrial');a=r.history_tensor([[1,2]]);b=r.history_tensor([[1,2]])
  self.assertTrue(torch.equal(a,b));self.assertEqual(a.shape,(1,10,6))
 def test_decoder_causality(self):
  c=ModelConfig(levels=3,code_size=3,hidden_size=8,heads=2,encoder_layers=1,decoder_layers=1,ffn_size=16,dropout=0)
  m=GenerativeRecommender(c).eval();h=torch.ones(1,10,3,dtype=torch.long)
  with torch.no_grad():
   a=m(h,torch.tensor([[0,1,2]]));b=m(h,torch.tensor([[0,2,1]]))
  self.assertTrue(torch.equal(a[:,:2],b[:,:2]))
 def test_lambda_zero_exact(self):
  r=Retriever(DEFAULT_ROOT,'industrial');h=r.history_tensor([[1,2],[3]])
  with torch.no_grad():
   a,b=corrected_beam(r.model,h,r.trie,r.priors,0);c,d,_=beam_search(r.model,h,r.trie,20)
  self.assertTrue(torch.equal(a,c));self.assertTrue(torch.equal(b,d))
 def test_potential_constant_cancels(self):
  codes=np.array([[0,0],[0,1],[1,0]]);inverse,counts=prefix_groups(codes)
  query=torch.zeros(2,4);text=torch.zeros(3,4);phi=completion_potentials(query,text,inverse,counts)
  for x in phi:self.assertTrue(torch.equal(x,torch.zeros_like(x)))
 def test_eos_not_another_probability(self):
  trie=PaddedSIDTrie([[0],[0,1]],2)
  paths=np.array([[trie.padded[0],trie.padded[1]]]);scores=np.array([[0.,-1.]])
  self.assertEqual(native_items(paths,scores,trie)[0,:2].tolist(),[0,1])
  self.assertEqual(trie.pad,trie.eos+1)
 def test_priors_train_counts_each_depth(self):
  r=Retriever(DEFAULT_ROOT,'industrial');priors,stats=conditional_priors(r.codes_array,frequency(DEFAULT_ROOT,'industrial',len(r.codes)),r.trie)
  for t,st in zip(priors,stats):
   self.assertTrue(torch.allclose(t.exp().sum(1),torch.ones(t.shape[0])));self.assertEqual(st['count_sum'],36259)
 def test_all_reference_archives_independent_metrics(self):
  for ds,cfg in settings()['datasets'].items():
   rows=read_rows(DEFAULT_ROOT,ds,'test');target=np.array([r[2] for r in rows]);users=np.array([r[0] for r in rows]);cg=groups(frequency(DEFAULT_ROOT,ds,cfg['items']))
   for method,spec in cfg['methods'].items():
    a=DEFAULT_ROOT/spec['artifact'];record,_=evaluate(np.load(a/'reference_items.npz')['items'],target,cg[target],cg,users)
    self.assertEqual(compare(record,json.loads((a/'reference_metrics.json').read_text())),[],(ds,method))
 def test_tokenizers_frozen_codes(self):
  from sid_retrieval.tokenization import make_model,load_initialized,RQVAE
  for ds in ['industrial','office']:
   raw=torch.from_numpy(np.load(DEFAULT_ROOT/'data'/ds/'item_embeddings.npy').astype(np.float32))
   a=DEFAULT_ROOT/'artifacts'/ds/'ours';state=torch.load(a/'tokenizer.pt',weights_only=True)
   m=make_model(2560,'parallel',6,16);load_initialized(m,state['state']);m.eval()
   with torch.no_grad():codes=m.get_indices(raw,use_sk=False).numpy()
   self.assertTrue(np.array_equal(codes,np.array(json.loads((a/'codes.json').read_text()))))
   a=DEFAULT_ROOT/'artifacts'/ds/'tiger';state=torch.load(a/'tokenizer.pt',weights_only=True);m=RQVAE(**state['config']);load_initialized(m,state['model']);m.eval()
   with torch.no_grad():codes=m.get_indices(torch.nn.functional.normalize(raw,dim=1),use_sk=False).numpy()
   self.assertTrue(np.array_equal(codes,np.array(json.loads((a/'codes.json').read_text()))[:,:3]))
 def test_training_import_and_no_test_read(self):
  import sid_retrieval.training as t
  self.assertNotIn('"test"',inspect.getsource(t))
 def test_empty_group_reports_none(self):
  record,_=evaluate(np.array([[0,-1]]),np.array([0]),np.array(['head']),np.array(['head','tail']),np.array(['u0']))
  self.assertIsNone(record['metrics']['tail']['Recall@5']);self.assertIsNone(record['coverage']['tail']['5']['effective_hit_coverage'])
 def test_trace_equals_raw_search(self):
  from sid_retrieval.analysis import trace_beam,pattern_support,support_lookup
  r=Retriever(DEFAULT_ROOT,'office');history=r.history_tensor([[1,2],[3]])
  trace=trace_beam(r.model,history,r.trie);a,b,_=beam_search(r.model,history,r.trie,20)
  self.assertTrue(np.array_equal(a.numpy(),trace['paths6']));self.assertTrue(np.array_equal(b.numpy(),trace['scores6']))
  codes=r.codes_array;support,equal=pattern_support(codes,trace['paths6'][0],trace['scores6'][0],support_lookup())
  finite=np.isfinite(trace['scores6'][0]);distance=(~equal).sum(2);distance[:,~finite]=99;minimum=distance.min(1)
  for m in range(7):
   patterns=[k for k in range(64) if bin(k).count('1')==6-m]
   hit=np.array([any((int(s)>>k)&1 for k in patterns) for s in support])
   self.assertTrue(np.array_equal(hit,minimum<=m))
 def test_training_two_tiny_epochs(self):
  from sid_retrieval.training import train_plain
  c=ModelConfig(levels=2,code_size=3,hidden_size=8,heads=2,encoder_layers=1,decoder_layers=1,ffn_size=16,dropout=0)
  model=GenerativeRecommender(c)
  data={'history':torch.zeros(2,10,2,dtype=torch.long),'sid':torch.tensor([[0,1],[1,2]]),'targets':np.array([0,1])}
  with tempfile.TemporaryDirectory() as d:
   train_plain(model,data,data,3,Path(d),epochs=2)
   self.assertEqual(len(json.loads((Path(d)/'trace.json').read_text())),2)
   self.assertTrue((Path(d)/'model.pt').exists())
 def test_baseline_trained_artifact_override(self):
  r=Retriever(DEFAULT_ROOT,'office','rqvae',artifact='artifacts/office/rqvae')
  self.assertEqual(r.spec['eos'],r.model.config.code_size-1)
 def test_anonymous_text(self):
  bad=[chr(47)+'Users'+chr(47),chr(47)+'home'+chr(47)]
  for p in DEFAULT_ROOT.rglob('*'):
   if p.is_file() and p.suffix in ['.py','.md','.json','.toml','.txt','.csv']:
    s=p.read_text().lower()
    # Test constants themselves describe the forbidden patterns, so exclude test source.
    if p==Path(__file__):continue
    for token in bad:self.assertNotIn(token.lower(),s,str(p.relative_to(DEFAULT_ROOT)))

if __name__=='__main__':unittest.main()
