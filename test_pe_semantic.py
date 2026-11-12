"""Synthetic CPU checks. These do not execute the full pretrained PE model."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

import pe_semantic as pe
import semantic_vit as sv


class Tests(unittest.TestCase):
    def test_git_blob_verification(self):
        self.assertEqual(pe.git_blob_hash(b''),'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391')
        self.assertNotEqual(pe.git_blob_hash(b'code'),pe.git_blob_hash(b'code\n'))

    def test_state_dictionary_unwraps_only_known_prefix(self):
        tensor=torch.zeros(2,3)
        result=pe.canonical_state_dict({'state_dict':{'module.layer.weight':tensor}})
        self.assertEqual(list(result),['layer.weight'])
        torch.testing.assert_close(result['layer.weight'],tensor)
        with self.assertRaises(ValueError):
            pe.canonical_state_dict({'layer':tensor,'module.layer':tensor})
        with self.assertRaises(ValueError): pe.canonical_state_dict({'bad':'not a tensor'})

    def test_strict_loading_never_accepts_missing_weights(self):
        model=torch.nn.Linear(3,2)
        with self.assertRaises(RuntimeError):
            model.load_state_dict(pe.canonical_state_dict({'weight':torch.zeros(2,3)}),strict=True,assign=True)

    def test_loader_reads_actual_checkpoint_formats_and_freezes_weights(self):
        class TinyCLIP(torch.nn.Module):
            image_size=448
            context_length=72
            def __init__(self):
                super().__init__()
                self.visual=torch.nn.Linear(3,2)
                self.text=torch.nn.Linear(3,2)
            def to(self,*args,**kwargs):
                self.requested_device=kwargs
                return self
            @classmethod
            def from_config(cls,name,pretrained):
                if pretrained: raise AssertionError('Downloads must go through pinned revision')
                return cls()
        core=types.ModuleType('core');core.__path__=[]
        vision=types.ModuleType('core.vision_encoder');vision.__path__=[]
        upstream=types.ModuleType('core.vision_encoder.pe');upstream.CLIP=TinyCLIP
        core.vision_encoder=vision;vision.pe=upstream
        hf=types.ModuleType('huggingface_hub')
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);checkpoint=folder/'weights.pt'
            hf.hf_hub_download=lambda *args,**kwargs:str(checkpoint)
            modules={'core':core,'core.vision_encoder':vision,'core.vision_encoder.pe':upstream,'huggingface_hub':hf}
            original=TinyCLIP().state_dict()
            for zipped in (True,False):
                torch.save({'state_dict':{'module.'+k:v for k,v in original.items()}},checkpoint,_use_new_zipfile_serialization=zipped)
                with patch.dict('sys.modules',modules),patch.object(sv,'pinned_revision',return_value='a'*40):
                    model,revision=pe.load_pe(folder)
                self.assertEqual(revision,'a'*40)
                self.assertTrue(all(not p.requires_grad for p in model.parameters()))
                self.assertEqual(model.requested_device,dict(device='cuda',dtype=torch.float32))
                for key,value in model.state_dict().items(): torch.testing.assert_close(value,original[key])

    def test_uncertainty_uses_margins_with_deterministic_cap(self):
        logits=np.zeros((8,10),dtype=np.float32)
        logits[:,0]=[8,2,.1,4,.05,3,5,6]
        np.testing.assert_array_equal(pe.uncertainty_indices(logits,.25),[2,4])
        np.testing.assert_array_equal(pe.uncertainty_indices(logits,0),np.array([],dtype=np.int64))
        self.assertEqual(len(pe.uncertainty_indices(logits,.3)),3)
        np.testing.assert_array_equal(pe.uncertainty_indices(np.zeros((8,10)),.25),[0,1])
        with self.assertRaises(ValueError): pe.uncertainty_indices(logits,1.1)

    def test_crop_geometry_preserves_pixels(self):
        image=np.arange(32*32*3,dtype=np.uint8).reshape(1,32,32,3)
        boxes=pe.crop_boxes(28)
        self.assertEqual(boxes,[(0,0,28,28),(0,4,28,28),(4,0,28,28),(4,4,28,28),(2,2,28,28)])
        for box in boxes:
            crop=pe.crop_images(image,box)
            self.assertEqual(crop.shape,(1,28,28,3))
            np.testing.assert_array_equal(crop[:,0,0],image[:,box[0],box[1]])
        with self.assertRaises(ValueError): pe.crop_boxes(33)
        with self.assertRaises(ValueError): pe.crop_images(image,(8,0,28,28))

    def test_local_views_only_change_selected_rows(self):
        baseline=np.zeros((6,10),dtype=np.float32)
        local=np.ones((2,10),dtype=np.float32)
        result=pe.merge_local_logits(baseline,np.array([1,4]),local,.25)
        np.testing.assert_array_equal(result[[0,2,3,5]],baseline[[0,2,3,5]])
        np.testing.assert_array_equal(result[[1,4]],np.full((2,10),.25))
        np.testing.assert_array_equal(baseline,np.zeros((6,10)))
        with self.assertRaises(ValueError): pe.merge_local_logits(baseline,np.array([1,1]),local)

    def test_low_level_diagnostics_keep_shape_and_dtype(self):
        images=np.random.default_rng(3).integers(0,256,(4,32,32,3),dtype=np.uint8)
        original=images.copy()
        for mode in ('grayscale','posterize4','blur05'):
            transformed=pe.stress_images(images,mode)
            self.assertEqual(transformed.shape,images.shape)
            self.assertEqual(transformed.dtype,np.uint8)
            self.assertTrue(np.isfinite(transformed).all())
        np.testing.assert_array_equal(images,original)
        q=pe.stress_images(images,'posterize4')
        self.assertTrue((q%16==0).all())

    def test_new_views_cannot_alias_baseline_cache(self):
        base={'view':'full','crop':None,'stress':None,'selected_rows_sha256':None}
        self.assertNotEqual(sv.cache_key(base),sv.cache_key({**base,'crop':[0,0,28,28]}))
        self.assertNotEqual(sv.cache_key(base),sv.cache_key({**base,'selected_rows_sha256':sv.digest([0,2])}))
        self.assertNotEqual(sv.cache_key(base),sv.cache_key({**base,'stress':'grayscale'}))

    def fixture(self,folder):
        data=dict(hashes={'a':'hash'},parts=(np.array([0]),np.array([1]),np.array([2])),
                  test_ids=np.array([3,1]),val=np.zeros((3,32,32,3),dtype=np.uint8))
        manifest=dict(data_hashes=data['hashes'],classes=list(sv.CLASSES),validation_parts=[p.tolist() for p in data['parts']])
        (folder/'experiment.json').write_text(json.dumps(manifest))
        np.savez(folder/'siglip_zero_shot_probabilities.npz',test_ids=data['test_ids'],
                 val=np.ones((3,10))*.1,test=np.ones((2,10))*.1)
        return data

    def test_previous_probabilities_require_matching_data_and_ids(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);data=self.fixture(folder)
            (val,test),provenance=pe.load_previous_siglip(folder,data)
            self.assertEqual(val.shape,(3,10));self.assertEqual(test.shape,(2,10))
            self.assertIn('probability_sha256',provenance)
            data['test_ids']=np.array([1,3])
            with self.assertRaises(ValueError): pe.load_previous_siglip(folder,data)

    def test_previous_probabilities_reject_changed_class_order(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);data=self.fixture(folder)
            path=folder/'experiment.json'
            manifest=json.loads(path.read_text());manifest['classes'].reverse()
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError): pe.load_previous_siglip(folder,data)


if __name__=='__main__': unittest.main()
