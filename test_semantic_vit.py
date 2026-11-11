"""CPU checks for data alignment, leakage boundaries, caching, and probe resume.

Run: python3 -m unittest -v test_semantic_vit
These synthetic checks do not validate pretrained GPU inference or Kaggle accuracy.
"""
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

import semantic_vit as m

torch.set_num_threads(1)


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_stratified_disjoint_complete(self):
        y = np.repeat(np.arange(10), 31)
        a,b,c = m.stratified_parts(y)
        self.assertEqual(len(set(a)&set(b)|set(a)&set(c)|set(b)&set(c)), 0)
        np.testing.assert_array_equal(np.sort(np.concatenate([a,b,c])), np.arange(len(y)))
        for split in (a,b,c):
            self.assertEqual(len(np.unique(y[split])),10)

    def test_public_template_parser_does_not_execute(self):
        source = 'raise RuntimeError("must not execute")\nOPENAI_IMAGENET_TEMPLATES=(' + ','.join(['lambda c: f"{c}"']*80) + ')'
        self.assertEqual(m.parse_prompt_bank(source), ['{c}']*80)
        with self.assertRaises(ValueError):
            m.parse_prompt_bank(source.replace('{c}', '{str(c)}', 1))

    def test_descriptor_excludes_registers(self):
        x = torch.zeros(2,8,4)
        x[:,0,0] = 1
        x[:,1:5] = 9999
        x[:,5:,2] = 1
        z = m.dino_descriptor(SimpleNamespace(last_hidden_state=x),4)
        expected = torch.tensor([1,0,0,0,0,0,1,0],dtype=torch.float32)/2**.5
        torch.testing.assert_close(z, expected.expand(2,-1))
        torch.testing.assert_close(z.norm(dim=1),torch.ones(2))

    def test_chunk_roundtrip_and_corruption(self):
        x = np.arange(24).reshape(6,4).astype(np.float16)
        m.save_feature_chunk(x,self.root/'cache',self.root/'tmp',0,6)
        np.testing.assert_array_equal(m.load_feature_chunk(self.root/'cache',0,6,4),x)
        path = self.root/'cache/0000000_0000006.npy'
        with path.open('ab') as f: f.write(b'corrupt')
        with self.assertRaises(ValueError):
            m.load_feature_chunk(self.root/'cache',0,6,4)

    def test_nonfinite_cache_rejected(self):
        with self.assertRaises(ValueError):
            m.save_feature_chunk(np.array([[np.nan]]),self.root/'c',self.root/'t',0,1)

    def test_resumed_extraction_skips_completed_chunks(self):
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.rows = 0
            def forward(self,x):
                self.rows += len(x)
                return x.float().mean((2,3))
        enc=Encoder()
        raw=np.arange(9*32*32*3,dtype=np.uint8).reshape(9,32,32,3)
        spec=dict(kind='mock',split='train',view='full',seed=42)
        args=(enc,lambda x:x,raw,spec,self.root/'c',self.root/'t',2,3,m.Budget(1),100)
        a=m.extract_cached(*args,chunk_size=4)
        self.assertEqual(enc.rows,9)
        b=m.extract_cached(*args,chunk_size=4)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(enc.rows,9)

    def test_cache_identity_includes_transform_and_revision(self):
        self.assertNotEqual(m.cache_key({'revision':'a','size':256}),m.cache_key({'revision':'b','size':256}))
        self.assertNotEqual(m.cache_key({'revision':'a','size':256}),m.cache_key({'revision':'a','size':384}))

    def fixture(self):
        rng=np.random.default_rng(9)
        y=np.tile(np.arange(10),20)
        x=(np.eye(10)[y]+rng.normal(0,.02,(len(y),10))).astype(np.float16)
        return x,y

    def settings(self):
        return dict(seed=42,lr=.04,weight_decay=.01,epochs=5,min_epochs=5,patience=10,batch=32)

    def test_head_learns_and_resume_reproduces(self):
        x,y=self.fixture()
        ids=np.arange(50)
        full=m.train_probe([x],y,x,x,y,ids,self.root/'full',self.settings(),m.Budget(1),device='cpu')
        class Interrupted:
            calls=0
            def require(self,*args):
                self.calls+=1
                if self.calls==3: raise m.BudgetExceeded('test interruption')
        with self.assertRaises(m.BudgetExceeded):
            m.train_probe([x],y,x,x,y,ids,self.root/'resume',self.settings(),Interrupted(),device='cpu')
        resumed=m.train_probe([x],y,x,x,y,ids,self.root/'resume',self.settings(),m.Budget(1),device='cpu')
        for key,value in full.state_dict().items():
            torch.testing.assert_close(value,resumed.state_dict()[key],rtol=0,atol=0)
        final_a=torch.load(self.root/'full/last.pt',weights_only=True)
        final_b=torch.load(self.root/'resume/last.pt',weights_only=True)
        for k,v in final_a['head'].items():
            torch.testing.assert_close(v,final_b['head'][k],rtol=0,atol=0)
        for parameter, state in final_a['optimizer']['state'].items():
            for k,v in state.items():
                torch.testing.assert_close(v,final_b['optimizer']['state'][parameter][k],rtol=0,atol=0)
        self.assertGreater(m.scores(m.predict_head(full,x,'cpu'),y)['accuracy'],.95)

    def test_head_selection_does_not_read_calibration_or_audit_labels(self):
        x,y=self.fixture()
        select=np.arange(50)
        altered=y.copy()
        altered[50:]=(altered[50:]+1)%10
        a=m.train_probe([x],y,x,x,y,select,self.root/'a',self.settings(),m.Budget(1),device='cpu')
        b=m.train_probe([x],y,x,x,altered,select,self.root/'b',self.settings(),m.Budget(1),device='cpu')
        for k,v in a.state_dict().items(): torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)

    def test_changed_head_settings_rejected(self):
        x,y=self.fixture()
        m.train_probe([x],y,x,x,y,np.arange(50),self.root,self.settings(),m.Budget(1),device='cpu')
        settings={**self.settings(),'seed':43}
        with self.assertRaises(ValueError):
            m.train_probe([x],y,x,x,y,np.arange(50),self.root,settings,m.Budget(1),device='cpu')

    def test_temperature_and_blend(self):
        x,y=self.fixture()
        t=m.fit_temperature(x,y)
        p=m.probabilities(x,t)
        np.testing.assert_allclose(p.sum(1),1,atol=1e-6)
        np.testing.assert_allclose(m.blend_probabilities([p,p],[.75,.25]),p,atol=1e-6)
        np.testing.assert_array_equal(p.argmax(1),x.argmax(1))
        with self.assertRaises(ValueError): m.blend_probabilities([p,p],[1,1])
        bad=p.copy();bad[0,:2]=[-1,2]
        with self.assertRaises(ValueError): m.blend_probabilities([bad],[1])

    def test_csv_alignment_follows_template(self):
        template=self.root/'template.csv'
        template.write_text('id,label\n8,0\n2,0\n5,0\n')
        p=np.eye(10)[[1,3,6]]
        m.write_submission(np.array([2,5,8]),p,template,self.root/'out.csv')
        with (self.root/'out.csv').open() as f: rows=list(csv.DictReader(f))
        self.assertEqual(rows,[{'id':'8','label':'6'},{'id':'2','label':'1'},{'id':'5','label':'3'}])
        with self.assertRaises(ValueError):
            m.write_submission(np.array([2,2,8]),p,template,self.root/'bad.csv')

    def test_read_data_uses_only_test_pixels_and_ids(self):
        for name in ('train','validation'):
            np.savez(self.root/f'{name}.npz',images=np.zeros((50,32,32,3),dtype=np.uint8),labels=np.tile(np.arange(10),5))
        np.savez(self.root/'test.npz',images=np.zeros((2,32,32,3),dtype=np.uint8),ids=np.array([12,4]),labels=np.array(['forbidden'],dtype=object))
        (self.root/'sample_submission.csv').write_text('id,label\n4,0\n12,0\n')
        data=m.read_data(self.root)
        self.assertNotIn('test_labels',data)
        np.testing.assert_array_equal(data['test_ids'],[12,4])


if __name__ == '__main__':
    unittest.main()
