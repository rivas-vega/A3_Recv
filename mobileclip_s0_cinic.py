"""Supervised MobileCLIP2-S0 / CINIC-10. No teacher and no test-driven selection."""
from pathlib import Path
import csv, gc, hashlib, io, json, math, os, random, re, time
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

CLASSES = ('airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck')
MODEL_REPO = 'timm/MobileCLIP2-S0-OpenCLIP'
MODEL_REVISION = '095906d28bf54d7584dc411e8ffe448f34289e05'
# Flower mirror: same original train/valid PNGs, pinned and independently verified.
CINIC_REPO = 'flwrlabs/cinic10'
CINIC_REVISION = '44c23bab233087df95aad862c5b6fa0e5e6030cf'
CINIC_FILES = {
    'train': ('data/train-00000-of-00001.parquet', 'dd44e8832411a2f9d80ded629813ad450de9339032bd0e8afbae40029da43d07'),
    'valid': ('data/validation-00000-of-00001.parquet', '3ea8bea7b2cf33fefe9f450dc86307139a2fc89954e079a86e93bc48a18cff5d'),
}
DEFAULTS = dict(seed=42, batch_size=128, workers=2, warmup_epochs=3,
    max_epochs=25, min_finetune_epochs=5, patience=5, head_lr=1e-3,
    finetune_head_lr=1e-4, encoder_lr=1e-5, weight_decay=.01,
    label_smoothing=.05, selection_per_class=1000, augment=True,
    image_size=256, freeze_bn_stats=True)

def digest_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def json_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

def atomic_json(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True)); os.replace(tmp, path)

def atomic_save(obj, path):
    path = Path(path); tmp = path.with_suffix(path.suffix+'.tmp')
    torch.save(obj, tmp); os.replace(tmp, path)

def lock_json(obj, path):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != obj:
            raise RuntimeError(f'La configuración o los datos han cambiado: {path}. No reutilices esta ejecución.')
    else: atomic_json(obj, path)

def stable_repr(obj):
    # torchvision Compose includes function addresses; these change after a restart.
    return re.sub(r" at 0x[0-9a-fA-F]+", "", repr(obj))

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def check_images(x):
    if x.dtype != np.uint8 or x.ndim != 4 or x.shape[1:] != (32,32,3):
        raise ValueError(f'Se requieren imágenes uint8 NHWC 32x32 RGB; recibido {x.shape}, {x.dtype}')

def unique_file(root, name):
    found = list(Path(root).rglob(name))
    if len(found) != 1: raise ValueError(f'Esperado un {name}, encontrados {len(found)}')
    return found[0]

def read_validation(root):
    # No inference about its origin is made from its size or filename.
    path = unique_file(root, 'validation.npz')
    with np.load(path, allow_pickle=False) as z:
        x, y = z['images'], z['labels'].reshape(-1)
    check_images(x)
    if len(x)!=len(y) or not np.issubdtype(y.dtype,np.integer) or not np.isin(y,range(10)).all():
        raise ValueError('Etiquetas de validación incompatibles con CIFAR-10')
    return x, y.astype(np.int64), digest_file(path)

def read_kaggle_secrets(get_secret):
    """Read Colab secrets without prompting, printing, or persisting credentials."""
    def read(name):
        try:
            value = get_secret(name)
        except Exception as exc:
            if isinstance(exc, KeyError) or type(exc).__name__ == 'SecretNotFoundError':
                return None
            raise RuntimeError(f'Colab no permite leer {name}. En Secretos (llave), activa '
                               'el acceso de este notebook y vuelve a ejecutar.') from None
        return str(value).strip() if value is not None else None
    token = read('KAGGLE_API_TOKEN')
    if token:
        return {'KAGGLE_API_TOKEN': token}
    username, key = read('KAGGLE_USERNAME'), read('KAGGLE_KEY')
    if username and key:
        return {'KAGGLE_USERNAME': username, 'KAGGLE_KEY': key}
    raise RuntimeError('Configura una sola vez KAGGLE_API_TOKEN en Secretos de Colab (icono de llave), '
                       'con el valor completo del token y acceso al notebook activado. '
                       'Para una credencial antigua kaggle.json, usa los dos secretos '
                       'KAGGLE_USERNAME y KAGGLE_KEY. No hay login interactivo.')

def configure_kaggle_secrets():
    from google.colab import userdata
    from kagglehub import config as kaggle_config
    credentials = read_kaggle_secrets(userdata.get)
    # Override cached credentials as well: re-running this cell respects edited secrets.
    kaggle_config.clear_kaggle_credentials()
    if 'KAGGLE_API_TOKEN' in credentials:
        os.environ['KAGGLE_API_TOKEN'] = credentials['KAGGLE_API_TOKEN']
        kaggle_config.set_kaggle_api_token(credentials['KAGGLE_API_TOKEN'])
    else:
        os.environ.pop('KAGGLE_API_TOKEN', None)
        os.environ.update(credentials)
        kaggle_config.set_kaggle_credentials(credentials['KAGGLE_USERNAME'], credentials['KAGGLE_KEY'])
    del credentials
    print('Kaggle: credenciales leídas de Secretos; sin entrada interactiva.')

def _read_cinic_parquet(path, split, expected_count=90000):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    if pf.metadata.num_rows != expected_count:
        raise ValueError(f'CINIC {split}: número de filas inesperado')
    metadata = json.loads(pf.schema_arrow.metadata[b'huggingface'])
    if metadata['info']['features']['label']['names'] != list(CLASSES):
        raise ValueError('El orden de clases del mirror no coincide con CIFAR-10')
    records = []
    for batch in pf.iter_batches(batch_size=512, columns=['image', 'label']):
        for row in batch.to_pylist():
            label = row['label']; image = row['image']
            if not isinstance(label, int) or not 0 <= label < 10:
                raise ValueError('Etiqueta CINIC inválida')
            if not image['bytes'] or not image['path']:
                raise ValueError('Falta el PNG o el nombre original; no se inventan IDs')
            name = f"{split}/{CLASSES[label]}/{Path(image['path']).name}"
            with Image.open(io.BytesIO(image['bytes'])) as im:
                x = np.asarray(im.convert('RGB'), dtype=np.uint8).copy()
            if x.shape != (32,32,3): raise ValueError(f'Imagen inesperada: {name}')
            records.append((name, label, x))
    records.sort(key=lambda row: row[0])
    names = np.array([row[0] for row in records])
    labels = np.array([row[1] for row in records], dtype=np.int64)
    if len(set(names)) != expected_count or not (np.bincount(labels,minlength=10)==expected_count//10).all():
        raise ValueError('CINIC: nombres duplicados o balance de clases inesperado')
    return {split+'_names': names, split+'_y': labels,
            split+'_x': np.stack([row[2] for row in records])}

def download_cinic(cache):
    """Verified HF mirror: fetch train/validation only, never contact Edinburgh."""
    from huggingface_hub import hf_hub_download
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    packed = cache/'cinic_hf_train_valid.npz'; info_path = cache/'cinic_hf_pack.json'
    source = dict(repo=CINIC_REPO, revision=CINIC_REVISION,
                  files={split: {'filename': name, 'sha256': sha} for split,(name,sha) in CINIC_FILES.items()})
    if packed.exists() and info_path.exists():
        info = json.loads(info_path.read_text())
        if info.get('source') == source and info.get('pack_sha256') == digest_file(packed):
            with np.load(packed,allow_pickle=False) as z: data = {k:z[k] for k in z.files}
            return data, info
    data = {}
    for split,(filename,expected_sha) in CINIC_FILES.items():
        print(f'CINIC {split}: descarga desde Hugging Face; no requiere token.')
        path = hf_hub_download(CINIC_REPO, filename, repo_type='dataset',
                               revision=CINIC_REVISION, cache_dir=str(cache/'hf'), token=False)
        if digest_file(path) != expected_sha:
            raise ValueError(f'Checksum incorrecto del mirror: {filename}; no se usan estos datos')
        data.update(_read_cinic_parquet(path,split))
    with open(str(packed)+'.tmp','wb') as f: np.savez(f,**data)
    os.replace(str(packed)+'.tmp',packed)
    info = dict(source=source, pack_sha256=digest_file(packed),
        original_dataset='https://doi.org/10.7488/ds/2448',
        original_archive_md5='6ee4d0c996905fe93221de577967a372',
        original_archive_sha256='31b095acf6d75e25a9e028bae82a07a0f94ff6b00671be2802d34ac4efa81a9e',
        equivalence='All 180000 train/valid names, RGB pixels and labels checked against original archive')
    atomic_json(info, info_path)
    return data,info

def pixel_keys(x):
    check_images(x)
    # Treat exact horizontal mirrors as the same example as well.
    return [min(hashlib.sha256(im.tobytes()).hexdigest(),
                hashlib.sha256(im[:,::-1].tobytes()).hexdigest()) for im in x]

def disjoint_indices(train_keys, valid_keys, audit_keys):
    """Priority: competition validation, then CINIC valid, then CINIC train."""
    blocked=set(audit_keys); vi=[]; seen=set()
    for i,k in enumerate(valid_keys):
        if k not in blocked and k not in seen: vi.append(i); seen.add(k)
    blocked.update(valid_keys); ti=[]; seen=set()
    for i,k in enumerate(train_keys):
        if k not in blocked and k not in seen: ti.append(i); seen.add(k)
    return np.array(ti,dtype=np.int64),np.array(vi,dtype=np.int64)

def prepare_data(raw, audit_x, audit_y, cfg, run_dir, provenance):
    tk,vk,ak = pixel_keys(raw['train_x']),pixel_keys(raw['valid_x']),pixel_keys(audit_x)
    ti,vi = disjoint_indices(tk,vk,ak)
    rng=np.random.default_rng(cfg['seed']); selected=[]
    for cls in range(10):
        pool=vi[raw['valid_y'][vi]==cls]
        if len(pool)<cfg['selection_per_class']: raise ValueError('Validación insuficiente tras deduplicar')
        selected.extend(rng.choice(pool,cfg['selection_per_class'],replace=False).tolist())
    vi=np.sort(selected)
    audit_key_set=set(ak)
    report=dict(train_before=len(tk),train_after=len(ti),valid_before=len(vk),
        valid_unique_disjoint=len(disjoint_indices(tk,vk,ak)[1]),selection_size=len(vi),
        train_overlap_competition_validation=sum(k in audit_key_set for k in set(tk)),
        validation_origin='unknown; no assumption',dedup='RGB SHA256 and exact horizontal mirror; not near-duplicate search',
        train_class_counts=np.bincount(raw['train_y'][ti],minlength=10).tolist())
    identity=dict(provenance=provenance, train_ids_hash=json_hash(raw['train_names'][ti].tolist()),
        selection_ids_hash=json_hash(raw['valid_names'][vi].tolist()),counts=report)
    lock_json(identity,Path(run_dir)/'data_manifest.json')
    np.savez(Path(run_dir)/'split_indices.npz',train=ti,selection=vi)
    data=dict(train_x=raw['train_x'][ti],train_y=raw['train_y'][ti],
        selection_x=raw['valid_x'][vi],selection_y=raw['valid_y'][vi],audit_x=audit_x,audit_y=audit_y)
    return data,identity

class ImageDataset(Dataset):
    def __init__(self,x,y,transform): self.x,self.y,self.transform=x,y,transform
    def __len__(self): return len(self.x)
    def __getitem__(self,i): return self.transform(Image.fromarray(self.x[i])),int(self.y[i])

def worker_seed(_):
    seed=torch.initial_seed()%2**32; random.seed(seed); np.random.seed(seed)

def loader(x,y,transform,cfg,shuffle=False,seed=0):
    return DataLoader(ImageDataset(x,y,transform),batch_size=cfg['batch_size'],shuffle=shuffle,
        num_workers=cfg['workers'],pin_memory=torch.cuda.is_available(),worker_init_fn=worker_seed,
        generator=torch.Generator().manual_seed(seed),persistent_workers=False)

class Classifier(nn.Module):
    def __init__(self,visual,dim=512):
        super().__init__(); self.visual=visual; self.head=nn.Linear(dim,10)
    def forward(self,x): return self.head(self.visual(x))

def load_model(run_dir, cfg):
    import open_clip
    from huggingface_hub import hf_hub_download
    from torchvision import transforms as T
    # Read config and weights from the SAME immutable revision.
    config_path=hf_hub_download(MODEL_REPO,'open_clip_config.json',revision=MODEL_REVISION)
    weights=hf_hub_download(MODEL_REPO,'open_clip_model.safetensors',revision=MODEL_REVISION)
    reference=json.loads(Path(config_path).read_text()); pp=reference['preprocess_cfg']
    if reference['model_cfg']['vision_cfg']['image_size']!=cfg['image_size']:
        raise ValueError('La resolución debe coincidir con el checkpoint')
    # 3.2.0 contains the canonical MobileCLIP2-S0 config. Strict load by OpenCLIP.
    full,_,preprocess=open_clip.create_model_and_transforms('MobileCLIP2-S0',pretrained=weights,
        image_mean=pp['mean'],image_std=pp['std'],image_interpolation=pp['interpolation'],
        image_resize_mode=pp['resize_mode'],device='cpu')
    model=Classifier(full.visual,reference['model_cfg']['embed_dim']); del full; gc.collect()
    # No reparameterization before fine-tuning: retain trainable branches.
    weak=T.Compose([T.RandomCrop(32,padding=2,padding_mode='reflect'),T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(.15,.15,.15,.02)],p=.5)])
    train_transform=T.Compose([weak,preprocess]) if cfg['augment'] else preprocess
    manifest=dict(repo=MODEL_REPO,revision=MODEL_REVISION,weights_sha256=digest_file(weights),
        preprocess=pp,transform=stable_repr(preprocess),training_transform=stable_repr(train_transform),
        visual_parameters=sum(p.numel() for p in model.visual.parameters()),
        classifier_parameters=sum(p.numel() for p in model.head.parameters()),
        model_config=reference['model_cfg'])
    lock_json(manifest,Path(run_dir)/'model_manifest.json')
    return model,train_transform,preprocess,manifest

def set_stage(model,warmup,freeze_bn=True):
    model.train()
    model.visual.requires_grad_(not warmup)
    if warmup: model.visual.eval()
    elif freeze_bn:
        for module in model.visual.modules():
            if isinstance(module,nn.modules.batchnorm._BatchNorm): module.eval()
    model.head.train()

def make_optimizer(model,cfg,warmup):
    groups=[dict(params=model.head.parameters(),lr=cfg['head_lr'] if warmup else cfg['finetune_head_lr'])]
    if not warmup: groups.append(dict(params=model.visual.parameters(),lr=cfg['encoder_lr']))
    return torch.optim.AdamW(groups,weight_decay=cfg['weight_decay'])

@torch.inference_mode()
def evaluate(model,dl,device,amp_dtype):
    model.eval(); probs=[]; truth=[]; total_loss=0.
    for x,y in dl:
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
        with torch.autocast(device.type,dtype=amp_dtype,enabled=device.type=='cuda'):
            logits=model(x)
        logits=logits.float()
        total_loss+=nn.functional.cross_entropy(logits,y,reduction='sum').item()
        probs.append(logits.softmax(-1).cpu().numpy()); truth.append(y.cpu().numpy())
    p=np.concatenate(probs); y=np.concatenate(truth)
    return dict(accuracy=float((p.argmax(1)==y).mean()),loss=total_loss/len(y)),p

def choose_improvement(metrics,best):
    return best is None or metrics['accuracy']>best['accuracy'] or (
        metrics['accuracy']==best['accuracy'] and metrics['loss']<best['loss'])

def fit(model,data,train_tf,eval_tf,cfg,run_dir,protocol,session_hours=6.5):
    run_dir=Path(run_dir); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda': raise RuntimeError('Entrenamiento completo: activa GPU en Colab.')
    model.to(device); dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler=torch.amp.GradScaler('cuda',enabled=dtype==torch.float16)
    lock_json(protocol,run_dir/'protocol.json'); protocol_hash=json_hash(protocol)
    last_path=run_dir/'last.pt'; best_path=run_dir/'best.pt'
    start=0; history=[]; best=None; bad=0; spent=0.; resume=None
    if last_path.exists():
        resume=torch.load(last_path,map_location='cpu',weights_only=True)
        if resume['protocol_hash']!=protocol_hash: raise RuntimeError('Checkpoint de otro protocolo')
        model.load_state_dict(resume['model'],strict=True)
        start=resume['epoch']+1; history=resume['history']; best=resume['best']; bad=resume['bad']; spent=resume['seconds']
        if resume['terminal']:
            print('Entrenamiento ya completado; se conserva el checkpoint seleccionado.'); return history
    if (run_dir/'final_export.json').exists(): raise RuntimeError('Método exportado: no se vuelve a entrenar esta ejecución')
    selection_dl=loader(data['selection_x'],data['selection_y'],eval_tf,cfg)
    deadline=time.monotonic()+session_hours*3600; optimizer=None; previous_stage=None
    for epoch in range(start,cfg['max_epochs']):
        # Pause BETWEEN epochs; resume repeats no completed epoch. Time budget is approximate.
        expected=history[-1]['epoch_seconds'] if history else 0
        if time.monotonic()+expected+60>deadline:
            print('Pausa por presupuesto. Reanuda las celdas con el mismo RUN_NAME.'); break
        epoch_start=time.monotonic(); seed_all(cfg['seed']+epoch)
        warmup=epoch<cfg['warmup_epochs']; set_stage(model,warmup,cfg['freeze_bn_stats'])
        if optimizer is None or warmup!=previous_stage:
            optimizer=make_optimizer(model,cfg,warmup)
            if resume is not None and resume['warmup']==warmup:
                optimizer.load_state_dict(resume['optimizer']); scaler.load_state_dict(resume['scaler'])
            previous_stage=warmup
        if not warmup:
            progress=(epoch-cfg['warmup_epochs'])/max(1,cfg['max_epochs']-cfg['warmup_epochs']-1)
            factor=.1+.9*.5*(1+math.cos(math.pi*progress))
            for group,lr in zip(optimizer.param_groups,[cfg['finetune_head_lr'],cfg['encoder_lr']]): group['lr']=lr*factor
        dl=loader(data['train_x'],data['train_y'],train_tf,cfg,True,cfg['seed']+epoch)
        from tqdm.auto import tqdm
        loss_sum=0.; count=0
        for x,y in tqdm(dl,desc=f'Época {epoch+1}/{cfg["max_epochs"]} '+('cabeza' if warmup else 'fine-tuning')):
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=dtype):
                logits=model(x); loss=nn.functional.cross_entropy(logits,y,label_smoothing=cfg['label_smoothing'])
            if not torch.isfinite(loss): raise FloatingPointError('Pérdida no finita; checkpoint anterior conservado')
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            scaler.step(optimizer); scaler.update(); loss_sum+=loss.item()*len(y); count+=len(y)
        metrics,_=evaluate(model,selection_dl,device,dtype)
        improved=choose_improvement(metrics,best)
        elapsed=time.monotonic()-epoch_start; spent+=elapsed
        row=dict(epoch=epoch+1,stage='head' if warmup else 'finetune',train_loss=loss_sum/count,
            selection_accuracy=metrics['accuracy'],selection_loss=metrics['loss'],epoch_seconds=elapsed)
        history.append(row)
        if improved:
            best=dict(**metrics,epoch=epoch+1); bad=0
            atomic_save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                best=best,protocol_hash=protocol_hash),best_path)
        elif not warmup: bad+=1
        early=(not warmup and epoch+1>=cfg['warmup_epochs']+cfg['min_finetune_epochs'] and bad>=cfg['patience'])
        terminal=early or epoch+1==cfg['max_epochs']
        state=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(),epoch=epoch,warmup=warmup,history=history,best=best,bad=bad,
            seconds=spent,terminal=terminal,protocol_hash=protocol_hash,
            stop_reason='early_stopping' if early else ('max_epochs' if terminal else 'running'))
        atomic_save(state,last_path); atomic_json(history,run_dir/'history.json')
        atomic_json(dict(best=best,completed=terminal,stop_reason=state['stop_reason'],training_hours=spent/3600),run_dir/'status.json')
        print(f'Validación CINIC: {metrics["accuracy"]:.3%}; mejor época {best["epoch"]}; tiempo acumulado {spent/3600:.2f} h')
        resume=None
        if terminal: break
    return history

def load_selected(model,run_dir,protocol):
    run_dir=Path(run_dir); status=json.loads((run_dir/'status.json').read_text())
    if not status['completed']: raise RuntimeError('Reanuda el entrenamiento hasta terminar antes del audit/export.')
    best=torch.load(run_dir/'best.pt',map_location='cpu',weights_only=True)
    if best['protocol_hash']!=json_hash(protocol): raise RuntimeError('Protocolo incompatible')
    model.load_state_dict(best['model'],strict=True); model.eval()
    return best['best']

def align_submission(ids,labels,sample_ids):
    ids=[str(x) for x in ids]; sample_ids=[str(x) for x in sample_ids]
    if len(ids)!=len(labels) or len(set(ids))!=len(ids) or len(set(sample_ids))!=len(sample_ids):
        raise ValueError('IDs duplicados o longitud incorrecta')
    if set(ids)!=set(sample_ids) or not np.isin(labels,range(10)).all(): raise ValueError('IDs/clases incompatibles')
    by_id=dict(zip(ids,map(int,labels)))
    return [by_id[i] for i in sample_ids]

def export_final(model,root,eval_tf,cfg,run_dir,protocol):
    # This is the ONLY function that reads competition test pixels/IDs.
    run_dir=Path(run_dir); best=load_selected(model,run_dir,protocol)
    frozen=dict(protocol_hash=json_hash(protocol),best_sha256=digest_file(run_dir/'best.pt'),
        best_epoch=best['epoch'],inference='official preprocess, single full image; no TTA')
    lock_json(frozen,run_dir/'frozen_method.json')
    csv_path=run_dir/'submission_mobileclip2_s0_cinic.csv'
    if (run_dir/'final_export.json').exists():
        record=json.loads((run_dir/'final_export.json').read_text())
        if record['method']!=frozen or digest_file(csv_path)!=record['csv_sha256']:
            raise RuntimeError('Exportación final modificada')
        print('CSV final ya generado; no se cambia.'); return csv_path
    test_path=unique_file(root,'test.npz'); sample_path=unique_file(root,'sample_submission.csv')
    with np.load(test_path,allow_pickle=False) as z:
        x=z['images']; ids=z['ids'].reshape(-1)  # never read test labels
    check_images(x)
    if len(ids)!=len(x): raise ValueError('IDs e imágenes no coinciden')
    device=next(model.parameters()).device
    dtype=torch.bfloat16 if device.type=='cuda' and torch.cuda.is_bf16_supported() else torch.float16
    dl=loader(x,np.zeros(len(x),dtype=np.int64),eval_tf,cfg)
    # No metric for test: the zeros are only an unused Dataset placeholder.
    predictions=[]; model.eval()
    with torch.inference_mode():
        for batch,_ in dl:
            with torch.autocast(device.type,dtype=dtype,enabled=device.type=='cuda'):
                logits=model(batch.to(device,non_blocking=True))
            predictions.extend(logits.argmax(1).cpu().tolist())
    with open(sample_path,newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=['id','label']: raise ValueError(f'Columnas inesperadas: {reader.fieldnames}')
        sample_ids=[r['id'] for r in reader]
    labels=align_submission(ids,predictions,sample_ids)
    tmp=csv_path.with_suffix('.csv.tmp')
    with open(tmp,'w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['id','label']); writer.writerows(zip(sample_ids,labels))
    os.replace(tmp,csv_path)
    atomic_json(dict(method=frozen,csv_sha256=digest_file(csv_path),rows=len(labels),
        test_sha256=digest_file(test_path),sample_sha256=digest_file(sample_path)),run_dir/'final_export.json')
    return csv_path
