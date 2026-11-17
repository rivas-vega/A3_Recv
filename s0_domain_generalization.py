"""Fixed-budget data ablations for MobileCLIP2-S0; no Warehouse or test selection."""
from pathlib import Path, PurePosixPath
import copy, gc, hashlib, io, json, math, os, random, time, zipfile
from collections import defaultdict
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import mobileclip_s0_cinic as base

CLASSES = base.CLASSES
ARMS = ("cinic", "cinic_domainnet", "cinic_domainnet_adain")
DOMAINS = ("clipart", "painting", "sketch")
# Conservative, explicit mapping. No deer category is invented.
MAPPING = {"airplane":0, "car":1, "bird":2, "cat":3, "dog":5, "frog":6,
           "horse":7, "truck":9, "sailboat":8, "cruise_ship":8,
           "speedboat":8, "aircraft_carrier":8}
COVERED = (0,1,2,3,5,6,7,8,9)
ROOT = "https://csr.bu.edu/ftp/visda/2019/multi-source/"
ARCHIVES = {d: ROOT+("groundtruth/" if d in ("clipart","painting") else "")+d+".zip" for d in DOMAINS}
SPLITS = (("clipart","train"),("clipart","test"),("painting","train"),
          ("painting","test"),("sketch","test"))
DEFAULTS = dict(base.DEFAULTS, max_epochs=15, warmup_epochs=2,
    steps_per_epoch=400, selection_per_class=200, domain_probability=.30,
    style_probability=.30, style_size=128, style_alpha=.5,
    style_pixel_blend=.5, style_batch_size=32, style_chunk=512)
RELEASE = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/"
ASSETS = {"decoder.pth":14023458, "vgg_normalised.pth":80102481}

def download(url, path, expected_size=None):
    """Retry and resume large public files; persist first-download SHA256."""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    receipt=path.with_suffix(path.suffix+".json")
    if path.exists() and receipt.exists():
        meta=json.loads(receipt.read_text())
        if meta["url"]==url and meta["sha256"]==base.digest_file(path):
            if expected_size is None or path.stat().st_size==expected_size: return path
        raise ValueError(f"Cache alterada: {path}; retírala antes de volver a descargar.")
    part=path.with_suffix(path.suffix+".part")
    session=requests.Session()
    session.mount("https://",HTTPAdapter(max_retries=Retry(total=4,backoff_factor=1,
        status_forcelist=[429,500,502,503,504])))
    from tqdm.auto import tqdm
    for attempt in range(4):
        start=part.stat().st_size if part.exists() else 0
        headers={"Accept-Encoding":"identity"}
        if start: headers["Range"]=f"bytes={start}-"
        try:
            with session.get(url,headers=headers,stream=True,timeout=(20,90)) as response:
                if response.status_code==416:
                    part.unlink(missing_ok=True); continue
                response.raise_for_status()
                append=start>0 and response.status_code==206
                if append and not response.headers.get("Content-Range","").startswith(f"bytes {start}-"):
                    raise ValueError("Respuesta Range incompatible")
                if not append: start=0
                length=int(response.headers.get("Content-Length",0))
                with open(part,"ab" if append else "wb") as out, tqdm(
                    total=start+length if length else None,initial=start,unit="B",
                    unit_scale=True,desc=path.name) as bar:
                    for block in response.iter_content(2**20):
                        if block: out.write(block); bar.update(len(block))
                if length and part.stat().st_size!=start+length:
                    raise IOError("Descarga incompleta")
            if expected_size is not None and part.stat().st_size!=expected_size:
                raise ValueError(f"Tamaño inesperado: {path.name}")
            os.replace(part,path)
            base.atomic_json(dict(url=url,bytes=path.stat().st_size,
                sha256=base.digest_file(path)),receipt)
            return path
        except (requests.RequestException, IOError):
            if attempt==3: raise
    raise RuntimeError(f"No se pudo descargar {url}")

def parse_list(text, domain):
    result=[]; seen=set(); mapping_ids={}
    for line in text.splitlines():
        if not line.strip(): continue
        name,label=line.split()
        p=PurePosixPath(name)
        if p.is_absolute() or ".." in p.parts or p.parts[0]!=domain or len(p.parts)<3:
            raise ValueError("Ruta DomainNet no válida")
        category=p.parts[1]; label=int(label)
        if category in mapping_ids and mapping_ids[category]!=label:
            raise ValueError("La taxonomía cambia dentro del listado")
        mapping_ids[category]=label
        if name in seen: raise ValueError("IDs DomainNet duplicados")
        seen.add(name)
        if category in MAPPING: result.append((name,MAPPING[category],label))
    if not result: raise ValueError(f"No hay categorías compatibles en {domain}")
    return sorted(result)

def read_zip(archive, records):
    xs=[]; ys=[]; names=[]
    with zipfile.ZipFile(archive) as z:
        available=set(z.namelist())
        for name,y,_ in records:
            if name not in available: raise ValueError(f"Falta {name} en el archivo oficial")
            # Read only listed members; never extract paths from the archive.
            with Image.open(io.BytesIO(z.read(name))) as im:
                im=ImageOps.exif_transpose(im).convert("RGB")
                xs.append(np.asarray(im.resize((32,32),Image.Resampling.BOX)).copy())
            ys.append(y); names.append(name)
    return dict(x=np.stack(xs),y=np.array(ys,dtype=np.int64),names=np.array(names))

def download_domainnet(cache, keep_archives=False):
    """Only official cleaned lists; selected images cached as uint8 at 32x32."""
    cache=Path(cache)/"domainnet"; cache.mkdir(parents=True,exist_ok=True)
    packs={}; provenance={}
    for domain in DOMAINS:
        needed=[split for d,split in SPLITS if d==domain]
        records={}; sources={}
        for split in needed:
            url=ROOT+f"domainnet/txt/{domain}_{split}.txt"
            file=download(url,cache/f"{domain}_{split}.txt")
            records[split]=parse_list(file.read_text(),domain)
            sources[split]=dict(url=url,sha256=base.digest_file(file))
        packed=cache/f"{domain}_32.npz"; meta_path=cache/f"{domain}_32.json"
        identity=dict(lists=sources,mapping=MAPPING,resize="EXIF RGB PIL BOX 32x32")
        if packed.exists() and meta_path.exists():
            meta=json.loads(meta_path.read_text())
            if meta["identity"]!=identity or meta["sha256"]!=base.digest_file(packed):
                raise ValueError(f"Cache incompatible: {packed}")
        else:
            archive=download(ARCHIVES[domain],cache/f"{domain}.zip")
            arrays={}
            for split in needed:
                for k,v in read_zip(archive,records[split]).items(): arrays[split+"_"+k]=v
            with open(str(packed)+".tmp","wb") as f: np.savez(f,**arrays)
            os.replace(str(packed)+".tmp",packed)
            meta=dict(identity=identity,sha256=base.digest_file(packed),
                archive=json.loads(archive.with_suffix(".zip.json").read_text()))
            base.atomic_json(meta,meta_path)
            if not keep_archives:
                archive.unlink(); archive.with_suffix(".zip.json").unlink()
        with np.load(packed,allow_pickle=False) as z:
            for split in needed: packs[f"{domain}_{split}"]={k:z[split+"_"+k] for k in ("x","y","names")}
        provenance[domain]=meta
    return packs,provenance

def metrics_from_probabilities(p,y):
    y=np.asarray(y); pred=p.argmax(1)
    if len(p)!=len(y) or not len(y): raise ValueError("Evaluación vacía o desalineada")
    recalls={str(c):float((pred[y==c]==c).mean()) for c in np.unique(y)}
    return dict(accuracy=float((pred==y).mean()),macro_accuracy=float(np.mean(list(recalls.values()))),
        loss=float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()),
        count=len(y),class_accuracy=recalls)

def unique_keep(keys,blocked,conflicts):
    kept=[]; seen=set(blocked)
    for i,k in enumerate(keys):
        if k not in seen and k not in conflicts: kept.append(i); seen.add(k)
    return np.asarray(kept,dtype=np.int64)

def prepare_data(raw,dn,audit_x,audit_y,cfg,run_dir,provenance):
    """Priority: audits, selection pools, training. No transformed siblings cross splits."""
    groups={"competition_audit":dict(x=audit_x,y=audit_y,names=np.array([f"audit/{i}" for i in range(len(audit_y))])),
        "sketch_audit":dn["sketch_test"],
        "cinic_selection":dict(x=raw["valid_x"],y=raw["valid_y"],names=raw["valid_names"]),
        "clipart_selection":dn["clipart_test"],"painting_selection":dn["painting_test"],
        "cinic_train":dict(x=raw["train_x"],y=raw["train_y"],names=raw["train_names"]),
        "clipart_train":dn["clipart_train"],"painting_train":dn["painting_train"]}
    keys={}; labels=defaultdict(set)
    for name,g in groups.items():
        base.check_images(g["x"]); keys[name]=base.pixel_keys(g["x"])
        if len(g["y"])!=len(g["x"]) or not np.isin(g["y"],range(10)).all(): raise ValueError("Etiquetas incompatibles")
        for key,y in zip(keys[name],g["y"]): labels[key].add(int(y))
    conflicts={k for k,v in labels.items() if len(v)>1}
    blocked=set(); filtered={}; report={}; rng=np.random.default_rng(cfg["seed"])
    for name,g in groups.items():
        if name=="competition_audit":
            ix=np.arange(len(g["y"]))  # preserve the original labelled audit
        else: ix=unique_keep(keys[name],blocked,conflicts)
        blocked.update(keys[name])  # also reserve unused selection examples
        if name=="cinic_selection":
            selected=[]
            for c in range(10):
                pool=ix[g["y"][ix]==c]
                if len(pool)<cfg["selection_per_class"]: raise ValueError("CINIC selection insuficiente")
                selected.extend(rng.choice(pool,cfg["selection_per_class"],replace=False))
            ix=np.sort(selected)
        filtered[name]={k:v[ix] for k,v in g.items()}
        report[name]=dict(before=len(g["y"]),after=len(ix),
            class_counts=np.bincount(g["y"][ix],minlength=10).tolist())
    train_names=("cinic_train","clipart_train","painting_train")
    x=np.concatenate([filtered[n]["x"] for n in train_names])
    y=np.concatenate([filtered[n]["y"] for n in train_names])
    domains=np.concatenate([np.full(len(filtered[n]["y"]),j,dtype=np.int64) for j,n in enumerate(train_names)])
    ids=np.concatenate([np.array([n+"/"+str(s) for s in filtered[n]["names"]]) for n in train_names])
    for c in range(10):
        if not np.any((y==c)&(domains==0)): raise ValueError(f"Falta clase CINIC {c}")
    for d in (1,2):
        if set(y[domains==d].tolist())!=set(COVERED): raise ValueError("Cobertura DomainNet inesperada")
    data=dict(x=x,y=y,domain=domains,ids=ids,
        selection={n.removesuffix("_selection"):g for n,g in filtered.items() if n.endswith("_selection")},
        audit={n.removesuffix("_audit"):g for n,g in filtered.items() if n.endswith("_audit")})
    identity=dict(provenance=provenance,counts=report,mapping=MAPPING,conflicting_pixel_keys=len(conflicts),
        pixel_dedup="SHA256 RGB32 and exact horizontal mirror; not near-duplicate detection",
        train_hash=base.json_hash(list(zip(ids.tolist(),base.pixel_keys(x),y.tolist()))),
        selection_hash={n:base.json_hash(list(zip(g["names"].tolist(),base.pixel_keys(g["x"]),g["y"].tolist())))
                        for n,g in data["selection"].items()},
        audit_hash={n:base.json_hash(list(zip(g["names"].tolist(),base.pixel_keys(g["x"]),g["y"].tolist())))
                    for n,g in data["audit"].items()})
    Path(run_dir).mkdir(parents=True,exist_ok=True); base.lock_json(identity,Path(run_dir)/"data_manifest.json")
    return data,identity

def epoch_plan(data,arm,cfg,epoch):
    if arm not in ARMS: raise ValueError(arm)
    n=cfg["batch_size"]*cfg["steps_per_epoch"]
    rng=np.random.default_rng(np.random.SeedSequence([cfg["seed"],epoch]))
    target=np.resize(np.arange(10),n); rng.shuffle(target)
    indices=np.empty(n,dtype=np.int64)
    for c in range(10):
        mask=np.flatnonzero(target==c)
        pool=np.flatnonzero((data["y"]==c)&(data["domain"]==0))
        indices[mask]=rng.choice(pool,len(mask),replace=True)
    # Draw the same decisions for all arms; B and C use exactly the same original IDs.
    use_domain=rng.random(n)<cfg["domain_probability"]
    chosen_domain=rng.integers(1,3,n)
    for c in COVERED:
        for d in (1,2):
            positions=np.flatnonzero((target==c)&use_domain&(chosen_domain==d))
            pool=np.flatnonzero((data["y"]==c)&(data["domain"]==d))
            chosen=rng.choice(pool,len(positions),replace=True)
            if arm!="cinic": indices[positions]=chosen
    style_rng=np.random.default_rng(np.random.SeedSequence([cfg["seed"],epoch,991]))
    styled=(style_rng.random(n)<cfg["style_probability"]) if arm==ARMS[2] else np.zeros(n,dtype=bool)
    return indices,styled

class PlannedDataset(Dataset):
    def __init__(self,data,indices,styled,transform,style_x=None):
        self.data,self.indices,self.styled,self.transform,self.style_x=data,indices,styled,transform,style_x
        if styled.any() and style_x is None: raise ValueError("Falta cache AdaIN; no se sustituye silenciosamente")
    def __len__(self): return len(self.indices)
    def __getitem__(self,i):
        j=self.indices[i]
        x=self.style_x[j] if self.styled[i] else self.data["x"][j]
        return self.transform(Image.fromarray(x)),int(self.data["y"][j])

def training_loader(data,arm,cfg,epoch,transform,style_x=None):
    indices,styled=epoch_plan(data,arm,cfg,epoch)
    return DataLoader(PlannedDataset(data,indices,styled,transform,style_x),
        batch_size=cfg["batch_size"],num_workers=cfg["workers"],shuffle=False,
        pin_memory=torch.cuda.is_available(),worker_init_fn=base.worker_seed,
        generator=torch.Generator().manual_seed(cfg["seed"]+epoch),persistent_workers=False)

class AdaIN:
    def __init__(self,cache,device):
        import adain_networks as networks
        self.device=torch.device(device)
        self.encoder=copy.deepcopy(networks.vgg); self.decoder=copy.deepcopy(networks.decoder)
        self.manifest={}
        for filename,model in (("vgg_normalised.pth",self.encoder),("decoder.pth",self.decoder)):
            p=download(RELEASE+filename,Path(cache)/"adain"/filename,ASSETS[filename])
            model.load_state_dict(torch.load(p,map_location="cpu",weights_only=True),strict=True)
            self.manifest[filename]=dict(url=RELEASE+filename,sha256=base.digest_file(p))
        self.encoder=nn.Sequential(*list(self.encoder.children())[:31])
        for model in (self.encoder,self.decoder): model.eval().requires_grad_(False).to(self.device)
    @torch.inference_mode()
    def __call__(self,content,style,alpha=.5):
        c=self.encoder(content.to(self.device)); s=self.encoder(style.to(self.device))
        cm=c.mean((2,3),keepdim=True); cs=(c.var((2,3),keepdim=True)+1e-5).sqrt()
        sm=s.mean((2,3),keepdim=True); ss=(s.var((2,3),keepdim=True)+1e-5).sqrt()
        target=(c-cm)/cs*ss+sm
        return self.decoder(alpha*target+(1-alpha)*c).clamp(0,1)

def image_tensor(x,size):
    a=np.stack([np.asarray(Image.fromarray(im).resize((size,size),Image.Resampling.BILINEAR)) for im in x])
    return torch.from_numpy(a.copy()).permute(0,3,1,2).float()/255

def stylize_batch(styler,content,styles,cfg):
    generated=styler(image_tensor(content,cfg["style_size"]),image_tensor(styles,cfg["style_size"]),cfg["style_alpha"])
    a=(generated.cpu().permute(0,2,3,1).numpy()*255).round().astype(np.uint8)
    small=np.stack([np.asarray(Image.fromarray(im).resize((32,32),Image.Resampling.BOX)) for im in a])
    blend=cfg["style_pixel_blend"]
    return np.rint(blend*small+(1-blend)*content).clip(0,255).astype(np.uint8)

def build_style_cache(data,cfg,cache,identity,session_hours=1.5):
    """Only training painting donors, deterministic per-image pairing; resumable chunks."""
    if not torch.cuda.is_available(): raise RuntimeError("Activa GPU para construir AdaIN.")
    cache=Path(cache); cache.mkdir(parents=True,exist_ok=True)
    styler=AdaIN(cache,"cuda")
    settings={k:v for k,v in cfg.items() if k.startswith("style_")}
    manifest=dict(train_hash=identity["train_hash"],settings=settings,weights=styler.manifest,
        donors="painting_train only; all classes share donor pool",seed=cfg["seed"])
    base.lock_json(manifest,cache/"style_manifest.json")
    donors=np.flatnonzero(data["domain"]==2)
    rng=np.random.default_rng(cfg["seed"]+812)
    pair=rng.choice(donors,len(data["y"]),replace=True)
    deadline=time.monotonic()+session_hours*3600
    chunks=[]; chunk=cfg["style_chunk"]; batch=cfg["style_batch_size"]
    from tqdm.auto import tqdm
    for lo in tqdm(range(0,len(pair),chunk),desc="AdaIN: bloques"):
        hi=min(lo+chunk,len(pair)); p=cache/f"style_{lo:06d}.npy"; receipt=p.with_suffix(".json")
        if p.exists() and receipt.exists():
            meta=json.loads(receipt.read_text())
            if meta!=dict(lo=lo,hi=hi,sha256=base.digest_file(p)): raise ValueError("Bloque AdaIN alterado")
            a=np.load(p,allow_pickle=False); base.check_images(a)
            if len(a)!=hi-lo: raise ValueError("Bloque AdaIN incompleto")
        else:
            if time.monotonic()>deadline:
                del styler; gc.collect(); torch.cuda.empty_cache()
                raise RuntimeError("AdaIN pausado entre bloques. Reejecuta esta celda con el mismo RUN_NAME.")
            result=[]
            for j in range(lo,hi,batch):
                k=min(j+batch,hi)
                result.append(stylize_batch(styler,data["x"][j:k],data["x"][pair[j:k]],cfg))
            a=np.concatenate(result)
            with open(str(p)+".tmp","wb") as f: np.save(f,a)
            os.replace(str(p)+".tmp",p)
            base.atomic_json(dict(lo=lo,hi=hi,sha256=base.digest_file(p)),receipt)
        chunks.append(a)
    del styler; gc.collect(); torch.cuda.empty_cache()
    return np.concatenate(chunks),manifest

@torch.inference_mode()
def evaluate_groups(model,groups,eval_tf,cfg,device,dtype):
    result={}
    for name,g in groups.items():
        _,p=base.evaluate(model,base.loader(g["x"],g["y"],eval_tf,cfg),device,dtype)
        result[name]=metrics_from_probabilities(p,g["y"])
    return result

def aggregate_selection(groups):
    expected={"cinic","clipart","painting"}
    if set(groups)!=expected: raise ValueError("La selección necesita los tres dominios fijados")
    return dict(accuracy=float(np.mean([g["macro_accuracy"] for g in groups.values()])),
        loss=float(np.mean([g["loss"] for g in groups.values()])),domains=groups)

def fit(model,data,train_tf,eval_tf,cfg,run_dir,protocol,arm,style_x=None,session_hours=6):
    """Fixed updates, no early stopping; identical initial weights and schedule across arms."""
    run_dir=Path(run_dir); run_dir.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda")
    if not torch.cuda.is_available(): raise RuntimeError("Entrenamiento completo requiere GPU.")
    model.to(device)
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler=torch.amp.GradScaler("cuda",enabled=dtype==torch.float16)
    base.lock_json(protocol,run_dir/"protocol.json"); ph=base.json_hash(protocol)
    last_path=run_dir/"last.pt"; best_path=run_dir/"best.pt"
    start=0; history=[]; best=None; spent=0.; resume=None
    if last_path.exists():
        resume=torch.load(last_path,map_location="cpu",weights_only=True)
        if resume["protocol_hash"]!=ph: raise RuntimeError("Checkpoint de otro protocolo")
        model.load_state_dict(resume["model"],strict=True)
        start=resume["epoch"]+1; history=resume["history"]; best=resume["best"]; spent=resume["seconds"]
        if resume["terminal"]: return history
    if (run_dir.parent/"chosen_method.json").exists(): raise RuntimeError("El estudio ya está congelado")
    deadline=time.monotonic()+session_hours*3600; optimizer=None; previous=None
    for epoch in range(start,cfg["max_epochs"]):
        expected=history[-1]["epoch_seconds"] if history else 0
        if time.monotonic()+expected+60>deadline:
            print("Pausa por presupuesto; reejecuta con el mismo RUN_NAME."); break
        tic=time.monotonic(); base.seed_all(cfg["seed"]+epoch)
        warmup=epoch<cfg["warmup_epochs"]; base.set_stage(model,warmup,cfg["freeze_bn_stats"])
        if optimizer is None or warmup!=previous:
            optimizer=base.make_optimizer(model,cfg,warmup)
            if resume is not None and resume["warmup"]==warmup:
                optimizer.load_state_dict(resume["optimizer"]); scaler.load_state_dict(resume["scaler"])
            previous=warmup
        if not warmup:
            progress=(epoch-cfg["warmup_epochs"])/max(1,cfg["max_epochs"]-cfg["warmup_epochs"]-1)
            factor=.1+.9*.5*(1+math.cos(math.pi*progress))
            for group,lr in zip(optimizer.param_groups,[cfg["finetune_head_lr"],cfg["encoder_lr"]]): group["lr"]=lr*factor
        dl=training_loader(data,arm,cfg,epoch,train_tf,style_x)
        loss_sum=0.; count=0
        from tqdm.auto import tqdm
        for x,y in tqdm(dl,desc=f"{arm}: {epoch+1}/{cfg['max_epochs']}"):
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=dtype):
                loss=nn.functional.cross_entropy(model(x),y,label_smoothing=cfg["label_smoothing"])
            if not torch.isfinite(loss): raise FloatingPointError("Pérdida no finita")
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            scaler.step(optimizer); scaler.update(); loss_sum+=loss.item()*len(y); count+=len(y)
        metrics=aggregate_selection(evaluate_groups(model,data["selection"],eval_tf,cfg,device,dtype))
        elapsed=time.monotonic()-tic; spent+=elapsed
        row=dict(epoch=epoch+1,stage="head" if warmup else "finetune",train_loss=loss_sum/count,
            selection_accuracy=metrics["accuracy"],selection_loss=metrics["loss"],
            domains=metrics["domains"],epoch_seconds=elapsed,updates=(epoch+1)*cfg["steps_per_epoch"])
        history.append(row)
        if base.choose_improvement(metrics,best):
            best=dict(**metrics,epoch=epoch+1)
            base.atomic_save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                best=best,protocol_hash=ph),best_path)
        terminal=epoch+1==cfg["max_epochs"]
        base.atomic_save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
            optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),epoch=epoch,warmup=warmup,
            history=history,best=best,seconds=spent,terminal=terminal,protocol_hash=ph),last_path)
        base.atomic_json(history,run_dir/"history.json")
        base.atomic_json(dict(best=best,completed=terminal,training_hours=spent/3600,
            stop_reason="fixed_update_budget" if terminal else "running"),run_dir/"status.json")
        print(f"Selección media macro por dominio: {metrics['accuracy']:.3%}; {spent/3600:.2f} h")
        resume=None
    return history

def choose_completed(run_dir,protocols):
    """Lock winner before any reserved audit. Deterministic ties prefer simpler arm."""
    run_dir=Path(run_dir); scores={}; fingerprints={}
    for arm in ARMS:
        path=run_dir/arm
        status=json.loads((path/"status.json").read_text())
        if not status["completed"]: raise RuntimeError("Completa las tres variantes antes de elegir")
        checkpoint=torch.load(path/"best.pt",map_location="cpu",weights_only=True)
        if checkpoint["protocol_hash"]!=base.json_hash(protocols[arm]): raise RuntimeError("Protocolo incompatible")
        scores[arm]=status["best"]
        fingerprints[arm]=dict(checkpoint_sha256=base.digest_file(path/"best.pt"),
            protocol_hash=checkpoint["protocol_hash"])
    winner=min(ARMS,key=lambda a:(-scores[a]["accuracy"],scores[a]["loss"],ARMS.index(a)))
    record=dict(winner=winner,selection="mean per-domain macro accuracy: CINIC/clipart/painting",
        scores=scores,checkpoints=fingerprints)
    base.lock_json(record,run_dir/"chosen_method.json")
    return record

def export_chosen(model,competition_root,eval_tf,cfg,run_dir,protocols):
    run_dir=Path(run_dir)
    if not (run_dir/"chosen_method.json").exists(): raise RuntimeError("Primero congela el método")
    chosen=json.loads((run_dir/"chosen_method.json").read_text())
    if choose_completed(run_dir,protocols)!=chosen: raise RuntimeError("La elección ha cambiado")
    arm=chosen["winner"]
    path=base.export_final(model,competition_root,eval_tf,cfg,run_dir/arm,protocols[arm])
    dest=run_dir/"submission_s0_domain_generalization.csv"
    payload=path.read_bytes()
    if dest.exists() and dest.read_bytes()!=payload: raise RuntimeError("CSV final modificado")
    dest.write_bytes(payload)
    return dest
