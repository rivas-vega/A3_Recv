import io, json, os, tempfile, unittest, zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import mobileclip_s0_cinic as base
import s0_domain_generalization as d

def toy_data():
    rng=np.random.default_rng(123); ys=[]; ds=[]
    for domain in (0,1,2):
        for c in (range(10) if domain==0 else d.COVERED):
            for _ in range(3): ys.append(c); ds.append(domain)
    return dict(x=rng.integers(0,256,(len(ys),32,32,3),dtype=np.uint8),
        y=np.array(ys),domain=np.array(ds))

class ProtocolTests(unittest.TestCase):
    def test_taxonomy_and_unsafe_paths(self):
        rows=d.parse_list("sketch/deer/a.png 90\nsketch/cruise_ship/b.png 85\nsketch/car/c.png 44\n","sketch")
        self.assertEqual([x[1] for x in rows],[1,8])
        self.assertNotIn("deer",d.MAPPING)
        for path in ("../cat/a.png","sketch/../cat/a.png","/sketch/cat/a.png"):
            with self.assertRaises(ValueError): d.parse_list(path+" 1","sketch")
    def test_duplicate_list_and_taxonomy_conflict(self):
        for text in ("clipart/cat/a.png 1\nclipart/cat/a.png 1",
                     "clipart/cat/a.png 1\nclipart/cat/b.png 2"):
            with self.assertRaises(ValueError): d.parse_list(text,"clipart")
    def test_zip_selected_members_and_box_resize(self):
        stream=io.BytesIO(); im=Image.new("RGB",(64,48),(12,20,30)); im.save(stream,format="PNG")
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"x.zip"
            with zipfile.ZipFile(p,"w") as z:
                z.writestr("clipart/cat/one.png",stream.getvalue())
                z.writestr("../never_extract.txt","unsafe")
            pack=d.read_zip(p,[("clipart/cat/one.png",3,64)])
            self.assertEqual(pack["x"].shape,(1,32,32,3))
            self.assertEqual(pack["y"].tolist(),[3])
            self.assertEqual(pack["x"][0,0,0].tolist(),[12,20,30])
            self.assertFalse((Path(tmp).parent/"never_extract.txt").exists())
    def test_fixed_updates_balanced_labels_and_paired_ablation(self):
        data=toy_data(); cfg=dict(d.DEFAULTS,batch_size=20,steps_per_epoch=20)
        a,sa=d.epoch_plan(data,d.ARMS[0],cfg,4)
        b,sb=d.epoch_plan(data,d.ARMS[1],cfg,4)
        c,sc=d.epoch_plan(data,d.ARMS[2],cfg,4)
        np.testing.assert_array_equal(b,c)
        np.testing.assert_array_equal(data["y"][a],data["y"][b])
        self.assertTrue((np.bincount(data["y"][b])==40).all())
        self.assertTrue((data["domain"][a]==0).all())
        self.assertTrue((data["domain"][b[data["y"][b]==4]]==0).all())
        self.assertTrue(np.any(data["domain"][b]>0))
        self.assertFalse(sa.any()); self.assertFalse(sb.any()); self.assertTrue(sc.any())
        np.testing.assert_array_equal(d.epoch_plan(data,d.ARMS[2],cfg,4)[1],sc)
    def test_style_cache_is_required_and_labels_retained(self):
        data=toy_data(); i=np.array([0]); styled=np.array([True])
        with self.assertRaises(ValueError): d.PlannedDataset(data,i,styled,lambda x:x)
        ds=d.PlannedDataset(data,i,styled,lambda x:np.array(x),np.zeros_like(data["x"]))
        x,y=ds[0]; self.assertEqual(x.max(),0); self.assertEqual(y,data["y"][0])
    def test_macro_differs_from_micro(self):
        p=np.array([[.9,.1],[.9,.1],[.8,.2]])
        metrics=d.metrics_from_probabilities(p,np.array([0,0,1]))
        self.assertAlmostEqual(metrics["accuracy"],2/3)
        self.assertAlmostEqual(metrics["macro_accuracy"],.5)
        with self.assertRaises(ValueError): d.aggregate_selection({"sketch":metrics})
    def test_mirror_and_reserved_duplicates(self):
        x=toy_data()["x"][:3]
        keys=base.pixel_keys(x)
        self.assertEqual(keys[0],base.pixel_keys(x[:1,:,::-1,:])[0])
        self.assertEqual(d.unique_keep([keys[0],keys[1],keys[1],keys[2]],{keys[0]},{keys[2]}).tolist(),[1])
    def test_prepare_data_leakage_and_conflicting_labels(self):
        rng=np.random.default_rng(12)
        def group(classes,count,prefix):
            y=np.repeat(list(classes),count)
            return dict(x=rng.integers(0,256,(len(y),32,32,3),dtype=np.uint8),y=y,
                names=np.array([f"{prefix}/{i}" for i in range(len(y))]))
        train=group(range(10),4,"train"); valid=group(range(10),3,"valid")
        audit=group(range(10),1,"audit")
        dn={name:group(d.COVERED,3,name) for name in
            ("clipart_train","painting_train","clipart_test","painting_test","sketch_test")}
        # Duplicate audit in train; duplicate selection in another domain's train.
        train["x"][0]=audit["x"][0]
        dn["painting_train"]["x"][0]=dn["clipart_test"]["x"][0]
        # Conflicting train labels: exclude both examples instead of arbitrarily keeping one.
        train["x"][5]=train["x"][1]
        raw={**{"train_"+k:v for k,v in train.items()},**{"valid_"+k:v for k,v in valid.items()}}
        cfg=dict(d.DEFAULTS,selection_per_class=1)
        with tempfile.TemporaryDirectory() as tmp:
            data,identity=d.prepare_data(raw,dn,audit["x"],audit["y"],cfg,tmp,{"test":"synthetic"})
            tk=set(base.pixel_keys(data["x"]))
            for group_ in list(data["selection"].values())+list(data["audit"].values()):
                self.assertFalse(tk.intersection(base.pixel_keys(group_["x"])))
            self.assertGreaterEqual(identity["conflicting_pixel_keys"],1)
            self.assertEqual(len(data["selection"]["cinic"]["y"]),10)
            self.assertNotIn(4,data["selection"]["painting"]["y"])
    def test_resume_configuration_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"protocol.json"; base.lock_json({"updates":100},p)
            base.lock_json({"updates":100},p)
            with self.assertRaises(RuntimeError): base.lock_json({"updates":101},p)
    def test_winner_requires_all_complete_and_is_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); protocols={a:{"arm":a} for a in d.ARMS}
            for a in d.ARMS:
                p=root/a; p.mkdir()
                score=dict(accuracy=.8,loss=.3,epoch=1)
                base.atomic_json(dict(completed=True,best=score),p/"status.json")
                base.atomic_save(dict(protocol_hash=base.json_hash(protocols[a]),best=score),p/"best.pt")
            chosen=d.choose_completed(root,protocols); self.assertEqual(chosen["winner"],"cinic")
            base.atomic_json(dict(completed=False,best=score),root/d.ARMS[1]/"status.json")
            with self.assertRaises(RuntimeError): d.choose_completed(root,protocols)
    def test_submission_alignment_uses_ids(self):
        self.assertEqual(base.align_submission([7,2],[1,9],[2,7]),[9,1])
        with self.assertRaises(ValueError): base.align_submission([7,7],[1,9],[7,2])

@unittest.skipUnless(os.environ.get("RUN_NETWORK_TESTS")=="1","Explicit public-source smoke test")
class NetworkTests(unittest.TestCase):
    def test_official_sources(self):
        import requests
        for domain,split in d.SPLITS:
            url=d.ROOT+f"domainnet/txt/{domain}_{split}.txt"
            r=requests.get(url,timeout=90); r.raise_for_status()
            self.assertEqual(__import__("hashlib").sha256(r.content).hexdigest(),d.LIST_SHA256[f"{domain}_{split}"])
            rows=d.parse_list(r.text,domain)
            self.assertEqual(set(row[1] for row in rows),set(d.COVERED))
            print("SOURCE",domain,split,len(rows),__import__("hashlib").sha256(r.content).hexdigest(),flush=True)
        for url in d.ARCHIVES.values():
            with requests.get(url,headers={"Range":"bytes=0-15"},stream=True,timeout=90) as r:
                r.raise_for_status()
                self.assertTrue(next(r.iter_content(16)).startswith(b"PK"))
                print("ZIP_OK",url,flush=True)
    def test_adain_actual_weights_and_forward(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            styler=d.AdaIN(tmp,"cpu")
            x=torch.rand(1,3,32,32); result=styler(x,torch.flip(x,[3]),.5)
            self.assertEqual(result.shape,x.shape); self.assertTrue(torch.isfinite(result).all())
            for name,item in styler.manifest.items(): print("ADAIN_SHA256",name,item["sha256"],flush=True)
    def test_s0_actual_checkpoint_forward_and_head_step(self):
        torch.set_num_threads(2); base.seed_all(42)
        with tempfile.TemporaryDirectory() as tmp:
            model,train_tf,eval_tf,manifest=base.load_model(tmp,dict(d.DEFAULTS))
            base.set_stage(model,True)
            x=eval_tf(Image.fromarray(toy_data()["x"][0])).unsqueeze(0)
            before=model.head.weight.detach().clone()
            loss=torch.nn.functional.cross_entropy(model(x),torch.tensor([3]))
            loss.backward(); torch.optim.SGD(model.head.parameters(),lr=.01).step()
            self.assertFalse(torch.equal(before,model.head.weight))
            self.assertEqual(model(x).shape,(1,10))
            self.assertEqual(manifest["visual_parameters"],11406976)

if __name__=="__main__": unittest.main()
