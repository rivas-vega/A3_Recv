import json, tempfile, unittest
from pathlib import Path
import numpy as np
import torch
from torch import nn
import mobileclip_s0_cinic as m

class Tests(unittest.TestCase):
    def test_duplicates_and_flips(self):
        x=np.random.default_rng(9).integers(0,256,(3,32,32,3),dtype=np.uint8)
        keys=m.pixel_keys(np.concatenate([x,x[:1,:,::-1,:]]))
        self.assertEqual(keys[0],keys[3])
        ti,vi=m.disjoint_indices(['a','b','b','c','d'],['c','c','e'],['a'])
        self.assertEqual(ti.tolist(),[1,4]); self.assertEqual(vi.tolist(),[0,2])
    def test_alignment(self):
        self.assertEqual(m.align_submission([2,0,1],[8,3,5],['0','1','2']),[3,5,8])
        for ids,ys,ss in [([1,1],[0,1],[1,2]),([1],[10],[1]),([1],[0],[2])]:
            with self.assertRaises(ValueError): m.align_submission(ids,ys,ss)
    def test_selection_rule(self):
        best={'accuracy':.9,'loss':.3}
        self.assertFalse(m.choose_improvement({'accuracy':.89,'loss':.1},best))
        self.assertTrue(m.choose_improvement({'accuracy':.9,'loss':.2},best))
    def test_immutable_protocol(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'protocol.json'; m.lock_json({'seed':42},p); m.lock_json({'seed':42},p)
            with self.assertRaises(RuntimeError): m.lock_json({'seed':43},p)
    def test_training_stages_and_gradients(self):
        model=m.Classifier(nn.Sequential(nn.Linear(4,4),nn.BatchNorm1d(4)),dim=4)
        m.set_stage(model,True)
        self.assertFalse(model.visual.training); self.assertFalse(any(p.requires_grad for p in model.visual.parameters()))
        m.set_stage(model,False)
        self.assertTrue(all(p.requires_grad for p in model.visual.parameters()))
        self.assertFalse(model.visual[1].training)
        before=model.visual[0].weight.detach().clone()
        opt=m.make_optimizer(model,m.DEFAULTS,False)
        loss=nn.functional.cross_entropy(model(torch.randn(3,4)),torch.tensor([1,2,3]));loss.backward();opt.step()
        self.assertFalse(torch.equal(before,model.visual[0].weight))
    def test_prepare_without_test_access(self):
        rng=np.random.default_rng(8);x=rng.integers(0,256,(42,32,32,3),dtype=np.uint8)
        raw={'train_x':x[:20],'train_y':np.tile(np.arange(10),2),'train_names':np.array([str(i) for i in range(20)]),
             'valid_x':x[20:40],'valid_y':np.tile(np.arange(10),2),'valid_names':np.array([str(i) for i in range(20,40)])}
        raw['train_x'][0]=raw['valid_x'][0]
        with tempfile.TemporaryDirectory() as d:
            data,identity=m.prepare_data(raw,x[40:],np.array([0,1]),dict(m.DEFAULTS,selection_per_class=1),d,{'source':'synthetic'})
            self.assertEqual(len(data['train_x']),19); self.assertEqual(len(data['selection_x']),10)
            self.assertFalse(set(m.pixel_keys(data['train_x'])) & set(m.pixel_keys(data['selection_x'])))
            self.assertEqual(identity['counts']['validation_origin'],'unknown; no assumption')
    def test_incomplete_training_cannot_export(self):
        with tempfile.TemporaryDirectory() as d:
            m.atomic_json({'completed':False},Path(d)/'status.json')
            with self.assertRaises(RuntimeError): m.load_selected(nn.Linear(1,1),d,{})
    def test_export_uses_ids_and_does_not_read_labels(self):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__();self.bias=nn.Parameter(torch.arange(10,dtype=torch.float32))
            def forward(self,x):return self.bias.expand(len(x),-1)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); model=Tiny();protocol={'fixed':True}
            m.atomic_json({'completed':True},root/'status.json')
            m.atomic_save({'model':model.state_dict(),'best':{'epoch':2},'protocol_hash':m.json_hash(protocol)},root/'best.pt')
            x=np.zeros((3,32,32,3),np.uint8)
            # If labels were read, allow_pickle=False would reject this object array.
            np.savez(root/'test.npz',images=x,ids=[8,6,7],labels=np.array([object()],dtype=object))
            (root/'sample_submission.csv').write_text('id,label\n6,0\n7,0\n8,0\n')
            transform=lambda im:torch.from_numpy(np.array(im).copy()).permute(2,0,1).float()
            out=m.export_final(model,root,transform,dict(m.DEFAULTS,workers=0),root,protocol)
            self.assertEqual(out.read_text(),'id,label\n6,9\n7,9\n8,9\n')
            self.assertEqual(m.export_final(model,root,transform,dict(m.DEFAULTS,workers=0),root,protocol),out)

if __name__=='__main__':unittest.main()
