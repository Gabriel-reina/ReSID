"""Entry-point, isolation and report regressions for release 1.1."""
import csv,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from sid_retrieval.data import DEFAULT_ROOT
from sid_retrieval.cli import compare,main
from sid_retrieval.engine import Retriever
from sid_retrieval.paper_report import render
from unittest.mock import patch

class ReleaseV2Tests(unittest.TestCase):
 def run_python(self,*args):
  env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
  return subprocess.run([sys.executable,'-B',*args],cwd=DEFAULT_ROOT,env=env,text=True,capture_output=True,timeout=120)
 def test_tokenizer_fresh_entry_point(self):
  p=self.run_python('-m','sid_retrieval.tokenization','--help');self.assertEqual(p.returncode,0,p.stderr);self.assertIn('--method',p.stdout)
 def test_vendor_fresh_import(self):
  p=self.run_python('-c','from third_party.minionerec.rqvae import RQVAE; from sid_retrieval.tokenization import make_model')
  self.assertEqual(p.returncode,0,p.stderr)
 def test_strict_reference_comparison(self):
  for actual,expected in [(1,None),([1],[1,2]),({'x':False},{'x':True}),('test','valid'),(True,1)]:self.assertTrue(compare(actual,expected))
  self.assertEqual(compare({'x':[None,True,0.1]},{'x':[None,True,0.1]}),[])
 def test_unavailable_method_no_output(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'result'
   with patch.object(sys,'argv',['sid_retrieval','evaluate','--dataset','industrial','--method','serial_full','--output',str(out)]):
    with self.assertRaises(ValueError):main()
   self.assertFalse(out.exists())
 def test_serial_variant_uses_corresponding_artifacts(self):
  r=Retriever(DEFAULT_ROOT,'office','serial_full');p=Retriever(DEFAULT_ROOT,'office','ours')
  self.assertEqual(r.beam,20);self.assertEqual((r.prior,r.semantic,r.max_mask),(.5,.4,3))
  self.assertNotEqual(r.codes,p.codes);self.assertEqual(r.artifact.name,'serial_full')
 def test_seven_chapter_report(self):
  report=render();self.assertEqual(report.count('## Chapter '),7)
  self.assertIn('Parallel3x256 exact',report);self.assertIn('Full with serial SID',report);self.assertIn('VALID',report)
 def test_primary_training_alias_without_test_split(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);data=root/'data'/'office';data.mkdir(parents=True)
   for split in ['train','valid']:
    with (DEFAULT_ROOT/'data'/'office'/(split+'.csv')).open(newline='') as source:
     reader=csv.DictReader(source);rows=[next(reader),next(reader)]
     with (data/(split+'.csv')).open('w',newline='') as target:
      writer=csv.DictWriter(target,fieldnames=reader.fieldnames);writer.writeheader();writer.writerows(rows)
   out=root/'fit';p=self.run_python('-m','sid_retrieval','--root',str(root),'train','--dataset','office','--method','ours','--artifact',str(DEFAULT_ROOT/'artifacts/office/ours'),'--epochs','1','--output',str(out))
   self.assertEqual(p.returncode,0,p.stderr);self.assertTrue((out/'model.pt').exists())
   self.assertFalse((data/'test.csv').exists());self.assertEqual(json.loads((out/'training_protocol.json').read_text())['seed'],42)
 def test_english_public_text(self):
  for p in DEFAULT_ROOT.rglob('*'):
   if p.is_file() and p.suffix in ['.py','.md','.json','.toml','.csv','.txt']:
    self.assertFalse(any('\u3400'<=c<='\u9fff' for c in p.read_text()),str(p.relative_to(DEFAULT_ROOT)))

if __name__=='__main__':unittest.main()
