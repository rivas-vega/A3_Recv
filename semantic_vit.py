"""Colab experiment: frozen SigLIP 2 Giant + DINOv3 ViT, supervised probes.

No test labels, generated captions, generated prompts, or API predictions.
Prompt templates are imported verbatim from a pinned public OpenCLIP release.
Large models run sequentially. Intermediate image features are resumable.
"""
from __future__ import annotations

import ast
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

VERSION = "semantic_vit_v1"
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
OPENCLIP_REVISION = "4cc208d3a44a2fb6d6435006a46f520d2ccc29b1"
PROMPT_SHA256 = "3bd8adb168bdf80fd681a71258e67f2c4d0f8ed73eae9b504ed5063e1726bbe9"
PROMPT_URL = f"https://raw.githubusercontent.com/mlfoundations/open_clip/{OPENCLIP_REVISION}/src/open_clip/zero_shot_metadata.py"
DINO_MODELS = {
    "7b": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    "hplus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def atomic_torch(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def release_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, hours):
        if hours <= 0:
            raise ValueError("hours must be positive")
        self.deadline = time.monotonic() + hours * 3600

    def require(self, seconds=0, label="operación"):
        remaining = self.deadline - time.monotonic()
        if remaining < seconds:
            raise BudgetExceeded(
                f"Pausa: {label} necesita aproximadamente {seconds/3600:.2f} h; "
                f"quedan {max(0, remaining)/3600:.2f} h. Lo terminado queda guardado. "
                "Reanuda en otra sesión con el mismo RUN_NAME; no se sustituye el modelo automáticamente."
            )


def stratified_parts(labels, seed=42):
    """20% head selection, 20% temperature fitting, 60% reporting only."""
    rng = np.random.default_rng(seed)
    parts = [[], [], []]
    for label in range(10):
        ids = np.flatnonzero(np.asarray(labels) == label)
        if len(ids) < 5:
            raise ValueError("Validation needs at least five images per class")
        ids = rng.permutation(ids)
        n = max(1, len(ids) // 5)
        for dst, values in zip(parts, (ids[:n], ids[n:2*n], ids[2*n:])):
            dst.extend(values.tolist())
    return tuple(np.sort(np.asarray(p, dtype=np.int64)) for p in parts)


def parse_prompt_bank(source):
    """Read only literal f-strings; never execute downloaded Python."""
    tree = ast.parse(source)
    matches = [n.value for n in tree.body if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "OPENAI_IMAGENET_TEMPLATES" for t in n.targets)]
    if len(matches) != 1 or not isinstance(matches[0], (ast.Tuple, ast.List)):
        raise ValueError("Public prompt bank not found")
    templates = []
    for item in matches[0].elts:
        if not isinstance(item, ast.Lambda) or not isinstance(item.body, ast.JoinedStr):
            raise ValueError("Unexpected template syntax")
        if len(item.args.args) != 1 or item.args.args[0].arg != "c":
            raise ValueError("Unexpected template parameter")
        text = ""
        for part in item.body.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                text += part.value
            elif (isinstance(part, ast.FormattedValue) and isinstance(part.value, ast.Name)
                  and part.value.id == "c" and part.conversion == -1 and part.format_spec is None):
                text += "{c}"
            else:
                raise ValueError("Template contains executable expressions")
        if text.count("{c}") != 1:
            raise ValueError("Expected exactly one class placeholder")
        templates.append(text)
    if len(templates) != 80:
        raise ValueError("Expected the original 80 public templates")
    return templates


def load_prompt_bank(run_root):
    import requests
    folder = Path(run_root) / "provenance"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "openclip_zero_shot_metadata.py.txt"
    if not path.exists():
        response = requests.get(PROMPT_URL, timeout=60)
        response.raise_for_status()
        path.write_text(response.text)
    if file_hash(path) != PROMPT_SHA256:
        raise ValueError("Public template source differs from the audited upstream file")
    templates = parse_prompt_bank(path.read_text())
    provenance = dict(url=PROMPT_URL, revision=OPENCLIP_REVISION, sha256=file_hash(path),
                      template_count=len(templates), classes=list(CLASSES),
                      origin="Existing public templates; no LLM-generated text")
    atomic_json(provenance, folder / "prompts.json")
    return templates, provenance


def read_data(root):
    files = {}
    for name in ("train.npz", "validation.npz", "test.npz", "sample_submission.csv"):
        found = list(Path(root).rglob(name))
        if len(found) != 1:
            raise ValueError(f"Expected exactly one {name}; found {len(found)}")
        files[name] = found[0]
    data = {}
    for split, name in (("train", "train.npz"), ("val", "validation.npz"), ("test", "test.npz")):
        with np.load(files[name], allow_pickle=False) as f:
            images = f["images"]
            # Test reads pixels and IDs only, even if extra archive fields exist.
            values = f["ids"] if split == "test" else f["labels"]
        if images.dtype != np.uint8 or images.ndim != 4 or images.shape[1:] != (32, 32, 3):
            raise ValueError(f"Invalid {split} images")
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError("Labels and IDs must be integers")
        values = values.reshape(-1).astype(np.int64)
        if len(values) != len(images):
            raise ValueError("Image / metadata length mismatch")
        if split != "test" and not ((values >= 0) & (values < 10)).all():
            raise ValueError("Training/validation labels must be in 0..9")
        if split == "test" and len(np.unique(values)) != len(values):
            raise ValueError("Duplicate test IDs")
        data[split] = images
        data[split + ("_ids" if split == "test" else "_labels")] = values
    data["files"] = files
    data["hashes"] = {name: file_hash(path) for name, path in files.items()}
    data["parts"] = stratified_parts(data["val_labels"])
    return data


def verify_hf_access(token, model_id):
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError
    if not isinstance(token, str) or not token.strip().startswith("hf_"):
        raise RuntimeError("Escribe tu token completo en HF_TOKEN y ejecuta esa celda. Solo DINOv3 lo requiere.")
    token = token.strip()
    api = HfApi(token=token)
    try:
        name = api.whoami()["name"]
    except HfHubHTTPError:
        raise RuntimeError("Hugging Face no reconoce el token; revisa su valor y vuelve a ejecutar HF_TOKEN.") from None
    try:
        api.auth_check(repo_id=model_id, repo_type="model")
    except HfHubHTTPError:
        raise RuntimeError(f"Token reconocido para {name}, pero sin acceso a {model_id}. Revisa aprobación y permiso Read.") from None
    print("Acceso DINOv3 confirmado para:", name)
    return token


def pinned_revision(model_id, root, token=False):
    from huggingface_hub import HfApi
    path = Path(root) / "models" / (model_id.replace("/", "__") + ".json")
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["model_id"] != model_id:
            raise ValueError("Model identity mismatch")
        return saved["revision"]
    revision = HfApi(token=token).model_info(model_id).sha
    if not revision or len(revision) != 40:
        raise ValueError("Could not resolve an immutable model revision")
    atomic_json(dict(model_id=model_id, revision=revision), path)
    return revision


def strict_model(model_id, revision, token=False):
    from transformers import AutoModel
    model, info = AutoModel.from_pretrained(
        model_id, revision=revision, token=token, trust_remote_code=False,
        use_safetensors=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="sdpa", output_loading_info=True,
    )
    errors = {k: info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if info.get(k)}
    if errors:
        raise RuntimeError(f"Incomplete pretrained weights: {errors}")
    return model.requires_grad_(False).eval().to("cuda")


class GPUPreprocess:
    def __init__(self, spec, device="cuda"):
        self.spec, self.device = spec, torch.device(device)
        self.mean = torch.tensor(spec["mean"], device=self.device)[None, :, None, None]
        self.std = torch.tensor(spec["std"], device=self.device)[None, :, None, None]

    def __call__(self, raw):
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.v2 import functional as VF
        if raw.dtype != torch.uint8 or raw.ndim != 4 or raw.shape[1] != 3:
            raise ValueError("GPU preprocessing requires BCHW uint8")
        raw = raw.to(self.device, non_blocking=True)
        mode = {2: InterpolationMode.BILINEAR, 3: InterpolationMode.BICUBIC}[self.spec["resample"]]
        resized = VF.resize(raw, self.spec["size"], interpolation=mode, antialias=True)
        return (resized.float().mul(self.spec["scale"]) - self.mean) / self.std


def dino_descriptor(output, registers):
    tokens = output.last_hidden_state.float()
    if tokens.ndim != 3 or tokens.shape[1] <= 1 + registers:
        raise ValueError("Invalid DINO token layout")
    cls = F.normalize(tokens[:, 0], dim=-1)
    patch_mean = F.normalize(tokens[:, 1 + registers:].mean(1), dim=-1)
    return torch.cat((cls, patch_mean), dim=-1) / math.sqrt(2)


class FrozenEncoder(nn.Module):
    def __init__(self, model, kind, registers=0):
        super().__init__()
        self.model, self.kind, self.registers = model, kind, registers

    def forward(self, pixels):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.model(pixel_values=pixels)
        if self.kind == "siglip":
            return F.normalize(output.pooler_output.float(), dim=-1)
        return dino_descriptor(output, self.registers)


@torch.inference_mode()
def validate_preprocess(encoder, preprocess, reference, raw, report_path):
    from PIL import Image
    raw = raw[:8]
    cpu = reference(images=[Image.fromarray(x) for x in raw], return_tensors="pt")["pixel_values"]
    gpu = preprocess(torch.from_numpy(raw.copy()).permute(0, 3, 1, 2).contiguous())
    max_diff = float((cpu.to("cuda") - gpu).abs().max())
    a, b = encoder(cpu.to("cuda")), encoder(gpu)
    min_cos = float(F.cosine_similarity(a, b).min())
    report = dict(max_pixel_error=max_diff, min_embedding_cosine=min_cos,
                  threshold_pixel_error=0.025, threshold_embedding_cosine=0.999)
    atomic_json(report, report_path)
    if max_diff > 0.025 or min_cos < 0.999:
        raise RuntimeError("El preprocesado GPU no coincide suficientemente con la referencia. Consulta preprocessing_check.json.")
    print(f"Preprocesado contrastado: error máximo {max_diff:.5f}; coseno mínimo {min_cos:.6f}.")


@torch.inference_mode()
def benchmark(encoder, preprocess, images, candidates, budget):
    records = []
    for size in candidates:
        budget.require(0, "benchmark")
        raw = torch.from_numpy(images[:size].copy()).permute(0, 3, 1, 2).contiguous()
        try:
            torch.cuda.reset_peak_memory_stats()
            for _ in range(2):
                encoder(preprocess(raw))
            torch.cuda.synchronize()
            start = time.monotonic()
            for _ in range(3):
                encoder(preprocess(raw))
            torch.cuda.synchronize()
            records.append(dict(batch=size, images_per_second=3*len(raw)/(time.monotonic()-start),
                                peak_gib=torch.cuda.max_memory_allocated()/2**30))
        except torch.cuda.OutOfMemoryError:
            release_gpu()
            break
    if not records:
        raise RuntimeError("No cabe ni el batch mínimo. Revisa la GPU seleccionada.")
    best = max(records, key=lambda x: x["images_per_second"])
    print("Benchmark medido:", records, "\nBatch elegido:", best["batch"])
    return best, records


def cache_key(spec):
    return digest(spec)


def load_feature_chunk(folder, start, end, dim):
    path = Path(folder) / f"{start:07d}_{end:07d}.npy"
    record = path.with_suffix(".json")
    if not path.exists() or not record.exists():
        return None
    info = json.loads(record.read_text())
    if info != dict(start=start, end=end, dim=dim, sha256=file_hash(path)):
        raise ValueError(f"Cache corrupta: {path}. Elimina solo ese chunk y reanuda.")
    x = np.load(path, allow_pickle=False)
    if x.shape != (end-start, dim) or x.dtype != np.float16 or not np.isfinite(x).all():
        raise ValueError(f"Invalid cache shape / values: {path}")
    return x


def save_feature_chunk(array, folder, scratch, start, end):
    array = np.asarray(array).astype(np.float16)
    if not np.isfinite(array).all():
        raise ValueError("Non-finite features; cache not saved")
    folder, scratch = Path(folder), Path(scratch)
    folder.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    name = f"{start:07d}_{end:07d}.npy"
    local = scratch / name
    np.save(local, array.astype(np.float16), allow_pickle=False)
    target = folder / name
    temporary = target.with_suffix(".tmp")
    shutil.copyfile(local, temporary)
    os.replace(temporary, target)
    atomic_json(dict(start=start, end=end, dim=array.shape[1], sha256=file_hash(target)), target.with_suffix(".json"))
    local.unlink()


def raw_view(images, indices, view, seed):
    raw = torch.from_numpy(images[indices].copy()).permute(0, 3, 1, 2).contiguous()
    if view == "full":
        return raw
    if view == "hflip":
        return raw.flip(-1)
    if view != "augmix":
        raise ValueError(f"Unknown view {view}")
    from torchvision.transforms import AugMix
    transform = AugMix(severity=2, mixture_width=3, all_ops=True)
    result = []
    for image, index in zip(raw, indices):
        # Reproducible per image, independent of batching and resumed chunks.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed + int(index)))
            if bool(torch.rand(()) < 0.5):
                image = image.flip(-1)
            result.append(transform(image))
    return torch.stack(result)


@torch.inference_mode()
def extract_cached(encoder, preprocess, images, spec, cache_root, scratch, batch, dim, budget, rate, chunk_size=2048):
    folder = Path(cache_root) / cache_key(spec)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != spec:
        raise ValueError("Cache manifest mismatch")
    atomic_json(spec, manifest)
    features = np.empty((len(images), dim), dtype=np.float16)
    started, new_rows = time.monotonic(), 0
    for start in range(0, len(images), chunk_size):
        end = min(start+chunk_size, len(images))
        cached = load_feature_chunk(folder, start, end, dim)
        if cached is not None:
            features[start:end] = cached
            continue
        budget.require(1.3*(end-start)/rate + 20, f"chunk {spec['split']}/{spec['view']}")
        block = np.empty((end-start, dim), dtype=np.float16)
        for offset in range(start, end, batch):
            stop = min(offset+batch, end)
            raw = raw_view(images, np.arange(offset, stop), spec["view"], spec["seed"])
            values = encoder(preprocess(raw)).float().cpu().numpy()
            if values.shape != (stop-offset, dim):
                raise ValueError("Encoder feature dimension changed")
            block[offset-start:stop-start] = values
        save_feature_chunk(block, folder, scratch, start, end)
        features[start:end] = block
        new_rows += end-start
        print(f"{spec['kind']} {spec['split']}/{spec['view']}: {end:,}/{len(images):,}; "
              f"{(time.monotonic()-started)/60:.1f} min", flush=True)
    print(f"{spec['kind']} {spec['split']}/{spec['view']} lista; {new_rows:,} filas nuevas.")
    return features


def predict_head(head, features, device="cuda", batch=4096):
    head.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(features), batch):
            x = torch.as_tensor(np.asarray(features[start:start+batch]), dtype=torch.float32, device=device)
            values.append(head(x * math.sqrt(x.shape[1])).cpu())
    return torch.cat(values)


def scores(logits, labels):
    logits = torch.as_tensor(logits, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)
    return dict(accuracy=float((logits.argmax(1) == y).float().mean()),
                nll=float(F.cross_entropy(logits, y)))


def train_probe(train_views, labels, val_clean, val_aug, val_labels, select_ids, root, settings, budget, device="cuda"):
    """Only selection split influences stopping. Calibration/audit labels are not read."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    dim = train_views[0].shape[1]
    spec = dict(version=VERSION, settings=settings, dim=dim, train_rows=len(labels), views=len(train_views))
    path = root / "last.pt"
    torch.manual_seed(settings["seed"])
    head = nn.Linear(dim, 10).to(device)
    nn.init.normal_(head.weight, std=0.01)
    nn.init.zeros_(head.bias)
    optimizer = torch.optim.AdamW(head.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    start_epoch, best_score, bad, best_state, history, done = 1, -1., 0, None, [], False
    if path.exists():
        saved = torch.load(path, weights_only=True, map_location="cpu")
        if saved["spec"] != spec:
            raise ValueError("Head configuration changed; use a new RUN_NAME")
        head.load_state_dict(saved["head"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch, best_score, bad = saved["epoch"]+1, saved["best_score"], saved["bad"]
        best_state, history, done = saved["best_state"], saved["history"], saved["done"]
    if not done:
        x = torch.as_tensor(np.concatenate(train_views).astype(np.float32), device=device)
        x.mul_(math.sqrt(dim))
        y = torch.as_tensor(np.tile(labels, len(train_views)), dtype=torch.long, device=device)
        for epoch in range(start_epoch, settings["epochs"]+1):
            budget.require(30, "entrenamiento de la cabeza")
            started = time.monotonic()
            generator = torch.Generator(device=device).manual_seed(settings["seed"]+epoch)
            order = torch.randperm(len(y), generator=generator, device=device)
            head.train()
            optimizer.param_groups[0]["lr"] = settings["lr"] * (0.1+0.9*(1+math.cos(math.pi*(epoch-1)/max(1,settings["epochs"]-1)))/2)
            for ids in order.split(settings["batch"]):
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(head(x[ids]), y[ids])
                loss.backward()
                optimizer.step()
            clean = scores(predict_head(head, val_clean[select_ids], device), val_labels[select_ids])
            aug = scores(predict_head(head, val_aug[select_ids], device), val_labels[select_ids])
            score = (clean["accuracy"] + aug["accuracy"])/2
            if score > best_score:
                best_score, bad = score, 0
                best_state = {k: v.detach().cpu().clone() for k,v in head.state_dict().items()}
            else:
                bad += 1
            if not all(torch.isfinite(p).all() for p in head.parameters()):
                raise FloatingPointError("Head weights became non-finite")
            history.append(dict(epoch=epoch, clean_accuracy=clean["accuracy"], augmented_accuracy=aug["accuracy"],
                                selection_score=score, seconds=time.monotonic()-started))
            done = epoch == settings["epochs"] or (epoch >= settings["min_epochs"] and bad >= settings["patience"])
            atomic_torch(dict(spec=spec, head={k:v.detach().cpu() for k,v in head.state_dict().items()},
                              optimizer=optimizer.state_dict(), epoch=epoch, best_score=best_score,
                              bad=bad, best_state=best_state, history=history, done=done), path)
            atomic_json(history, root / "history.json")
            print(f"Head {epoch:03d}: clean {clean['accuracy']:.2%}, augmented {aug['accuracy']:.2%}", flush=True)
            if done:
                break
        del x, y
    if best_state is None:
        raise RuntimeError("No valid head checkpoint")
    head.load_state_dict(best_state)
    atomic_torch(dict(spec=spec, head=best_state, history=history), root / "best.pt")
    del optimizer
    return head.eval()


def fit_temperature(logits, labels):
    """A bounded 1-D fit on calibration labels; independent of ensemble weights."""
    x = torch.as_tensor(logits, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)
    temperatures = torch.logspace(-2, 2, 121)
    scaled = x[None] / temperatures[:, None, None]
    losses = (scaled.logsumexp(-1) - scaled[:, torch.arange(len(y)), y]).mean(-1)
    return float(temperatures[losses.argmin()])


def probabilities(logits, temperature):
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    return torch.softmax(torch.as_tensor(logits, dtype=torch.float32)/temperature, dim=-1).numpy()


def blend_probabilities(arrays, weights):
    if len(arrays) != len(weights) or not arrays or min(weights) < 0 or not np.isclose(sum(weights), 1):
        raise ValueError("Ensemble weights must be nonnegative and sum to one")
    shape = arrays[0].shape
    if any(a.shape != shape or not np.isfinite(a).all() or (a < 0).any() or not np.allclose(a.sum(1), 1, atol=2e-5) for a in arrays):
        raise ValueError("Invalid or misaligned probability arrays")
    return sum(w*a for w,a in zip(weights, arrays))


def write_submission(ids, probabilities_array, template, path):
    ids = np.asarray(ids)
    p = np.asarray(probabilities_array)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or len(np.unique(ids)) != len(ids):
        raise ValueError("Invalid IDs")
    if p.shape != (len(ids), 10) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1),1,atol=2e-5):
        raise ValueError("Invalid probabilities")
    with open(template, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["id", "label"]:
            raise ValueError("Expected Kaggle columns id,label")
        ordered = [int(row["id"]) for row in reader]
    if len(ordered) != len(ids) or len(set(ordered)) != len(ordered) or set(ordered) != set(ids.tolist()):
        raise ValueError("Submission/template ID mismatch")
    lookup = dict(zip(ids.tolist(), p.argmax(1).tolist()))
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        writer.writerows((i,lookup[i]) for i in ordered)
    os.replace(tmp,path)
    print(f"CSV listo: {path.name} ({len(ids):,} imágenes).")


class Experiment:
    def __init__(self, data, settings, run_root, cache_root, scratch, budget):
        self.data, self.settings = data, settings
        self.root, self.cache, self.scratch = Path(run_root), Path(cache_root), Path(scratch)
        self.budget = budget
        self.root.mkdir(parents=True, exist_ok=True)
        import torchvision
        import transformers
        self.environment = dict(torch=str(torch.__version__), torchvision=str(torchvision.__version__),
                                transformers=str(transformers.__version__))
        if transformers.__version__ != "4.57.3":
            raise RuntimeError("Reinicia Colab después de instalar transformers==4.57.3")
        self.templates, provenance = load_prompt_bank(self.root)
        manifest = dict(version=VERSION, settings=settings, data_hashes=data["hashes"],
                        classes=list(CLASSES), prompts=provenance, environment=self.environment,
                        validation_parts=[p.tolist() for p in data["parts"]])
        path = self.root / "experiment.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("Configuración/datos/versiones distintos: usa un RUN_NAME nuevo; las cachés compatibles se reutilizan.")
        atomic_json(manifest,path)

    def feature_spec(self, kind, model_id, revision, preprocess, split, view):
        filename = dict(train="train.npz", val="validation.npz", test="test.npz")[split]
        return dict(version=VERSION, kind=kind, model_id=model_id, revision=revision, preprocessing=preprocess,
                    split=split, data_sha256=self.data["hashes"][filename], view=view, seed=self.settings["seed"],
                    augmix=dict(severity=2,width=3,all_ops=True) if view=="augmix" else None,
                    environment=self.environment, precision="bf16_encoder_fp16_cache_fp32_head")

    def extract(self, encoder, pp, kind, model_id, revision, split, view, dim, perf):
        spec = self.feature_spec(kind,model_id,revision,pp.spec,split,view)
        return extract_cached(encoder,pp,self.data[split],spec,self.cache,self.scratch,
                              perf["batch"],dim,self.budget,perf["images_per_second"])

    def export(self, name, val_logits, test_logits, val_aug_logits=None):
        _, calibrate, audit = self.data["parts"]
        temperature = fit_temperature(torch.as_tensor(val_logits)[calibrate], self.data["val_labels"][calibrate])
        val_p = probabilities(val_logits,temperature)
        test_p = probabilities(test_logits,temperature)
        metrics = scores(np.log(np.maximum(val_p[audit],1e-30)),self.data["val_labels"][audit])
        metrics.update(name=name, temperature=temperature, provenance="measured_in_this_run",
                       audit_rows=len(audit), kaggle_public_accuracy=None)
        if val_aug_logits is not None:
            metrics["augmented_audit_accuracy"] = scores(torch.as_tensor(val_aug_logits)[audit],self.data["val_labels"][audit])["accuracy"]
        self.export_probabilities(name,val_p,test_p,metrics)
        return val_p,test_p

    def export_probabilities(self,name,val_p,test_p,metrics=None):
        _,_,audit=self.data["parts"]
        if metrics is None:
            metrics = dict(name=name, **scores(np.log(np.maximum(val_p[audit],1e-30)),self.data["val_labels"][audit]),
                           provenance="measured_in_this_run",audit_rows=len(audit),kaggle_public_accuracy=None)
        np.savez_compressed(self.root/f"{name}_probabilities.npz",test_ids=self.data["test_ids"],test=test_p,val=val_p)
        atomic_json(metrics,self.root/f"{name}_metrics.json")
        write_submission(self.data["test_ids"],test_p,self.data["files"]["sample_submission.csv"],self.root/f"submission_{name}.csv")
        print("Validación reservada (dominio CIFAR):",f"{metrics['accuracy']:.2%}")

    def load_probabilities(self,name):
        with np.load(self.root/f"{name}_probabilities.npz",allow_pickle=False) as f:
            if not np.array_equal(f["test_ids"],self.data["test_ids"]):
                raise ValueError("Probability IDs not aligned")
            return f["val"].copy(),f["test"].copy()

    def run_siglip(self):
        from transformers import AutoImageProcessor, AutoTokenizer
        model_id="google/siglip2-giant-opt-patch16-384"
        revision=pinned_revision(model_id,self.root,token=False)
        model=strict_model(model_id,revision,token=False)
        try:
            reference=AutoImageProcessor.from_pretrained(model_id,revision=revision,token=False,use_fast=False,trust_remote_code=False)
            spec=dict(size=[reference.size["height"],reference.size["width"]],resample=int(reference.resample),
                      mean=list(reference.image_mean),std=list(reference.image_std),scale=float(reference.rescale_factor),antialias=True)
            if spec["size"] != [384,384] or model.config.vision_config.hidden_size != 1536 or model.config.model_type != "siglip":
                raise ValueError("Unexpected SigLIP Giant configuration")
            pp=GPUPreprocess(spec)
            tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision,token=False,trust_remote_code=False)
            prototypes=[]
            with torch.inference_mode():
                for name in CLASSES:
                    text=[t.format(c=name) for t in self.templates]
                    vectors=[]
                    for start in range(0,len(text),32):
                        tokens=tokenizer(text[start:start+32],padding="max_length",max_length=64,truncation=True,return_tensors="pt").to("cuda")
                        with torch.autocast("cuda",dtype=torch.bfloat16):
                            z=model.get_text_features(**tokens)
                        vectors.append(F.normalize(z.float(),dim=-1))
                    prototypes.append(F.normalize(torch.cat(vectors).mean(0),dim=-1))
            prototypes=torch.stack(prototypes).cpu()
            atomic_torch(dict(prototypes=prototypes,model_id=model_id,revision=revision),self.root/"text_prototypes.pt")
            model.text_model.to("cpu")
            encoder=FrozenEncoder(model.vision_model,"siglip").eval()
            validate_preprocess(encoder,pp,reference,self.data["train"],self.root/"siglip_preprocessing_check.json")
            perf,records=benchmark(encoder,pp,self.data["train"],[8,16,32,64],self.budget)
            atomic_json(dict(best=perf,trials=records),self.root/"siglip_benchmark.json")
            print("SigLIP: primero val/test para guardar un CSV sin entrenamiento.")
            vf=self.extract(encoder,pp,"siglip",model_id,revision,"val","full",1536,perf)
            tf=self.extract(encoder,pp,"siglip",model_id,revision,"test","full",1536,perf)
            if self.settings["siglip_flip"]:
                vf=(vf.astype(np.float32)+self.extract(encoder,pp,"siglip",model_id,revision,"val","hflip",1536,perf))/2
                tf=(tf.astype(np.float32)+self.extract(encoder,pp,"siglip",model_id,revision,"test","hflip",1536,perf))/2
            # Average view logits via mean features; do not renormalize the view mean.
            zv=torch.from_numpy(vf.astype(np.float32)) @ prototypes.T
            zt=torch.from_numpy(tf.astype(np.float32)) @ prototypes.T
            self.export("siglip_zero_shot",zv,zt)
            train=[self.extract(encoder,pp,"siglip",model_id,revision,"train","full",1536,perf)]
            if self.settings["siglip_augmix"]:
                train.append(self.extract(encoder,pp,"siglip",model_id,revision,"train","augmix",1536,perf))
            va=self.extract(encoder,pp,"siglip",model_id,revision,"val","augmix",1536,perf)
        finally:
            if "encoder" in locals():
                del encoder
            del model
            release_gpu()
        self.fit_and_export("siglip_probe",train,vf,tf,va,dict(model_id=model_id,revision=revision))
        a,b=self.load_probabilities("siglip_zero_shot"),self.load_probabilities("siglip_probe")
        self.export_probabilities("siglip_blend",blend_probabilities([a[0],b[0]],[.5,.5]),blend_probabilities([a[1],b[1]],[.5,.5]))

    def fit_and_export(self,name,train,vf,tf,va,identity):
        settings=dict(seed=self.settings["seed"],lr=.001,weight_decay=.01,epochs=80,min_epochs=10,patience=10,batch=2048,
                      feature_identity=identity,data_hashes=self.data["hashes"],selection="mean_clean_augmix_accuracy")
        head=train_probe(train,self.data["train_labels"],vf,va,self.data["val_labels"],self.data["parts"][0],
                         self.root/name,settings,self.budget)
        logits_v=predict_head(head,vf)
        logits_t=predict_head(head,tf)
        logits_a=predict_head(head,va)
        self.export(name,logits_v,logits_t,logits_a)
        del head
        release_gpu()

    def run_dino(self,token):
        from torchvision.transforms import v2, InterpolationMode
        for name in ("siglip_zero_shot", "siglip_probe"):
            if not (self.root / f"{name}_probabilities.npz").exists():
                raise RuntimeError("Termina primero la celda SigLIP; DINO se combina con esos resultados.")
        model_id=DINO_MODELS[self.settings["dino_variant"]]
        token=verify_hf_access(token,model_id)
        revision=pinned_revision(model_id,self.root,token)
        model=strict_model(model_id,revision,token)
        try:
            if model.config.model_type != "dinov3_vit":
                raise ValueError("Expected DINOv3 ViT")
            registers=model.config.num_register_tokens
            dim=2*model.config.hidden_size
            spec=dict(size=[256,256],resample=2,mean=[.485,.456,.406],std=[.229,.224,.225],scale=1/255,antialias=True)
            pp=GPUPreprocess(spec)
            encoder=FrozenEncoder(model,"dino",registers).eval()
            # Same documented DINO transform on CPU, without HF's separate reference size.
            reference_transform=v2.Compose([v2.ToImage(),v2.Resize((256,256),interpolation=InterpolationMode.BILINEAR,antialias=True),
                                            v2.ToDtype(torch.float32,scale=True),v2.Normalize(spec["mean"],spec["std"])])
            def reference(images,return_tensors):
                return {"pixel_values":torch.stack([reference_transform(x) for x in images])}
            validate_preprocess(encoder,pp,reference,self.data["train"],self.root/"dino_preprocessing_check.json")
            perf,records=benchmark(encoder,pp,self.data["train"],[2,4,8,16,32],self.budget)
            atomic_json(dict(best=perf,trials=records,model_id=model_id),self.root/"dino_benchmark.json")
            images_needed=len(self.data["train"])+2*len(self.data["val"])+len(self.data["test"])
            print(f"DINO {self.settings['dino_variant']}: pasada sin caché estimada {1.3*images_needed/perf['images_per_second']/3600:.2f} h, más cabeza y escritura.")
            train=[self.extract(encoder,pp,"dino",model_id,revision,"train","full",dim,perf)]
            vf=self.extract(encoder,pp,"dino",model_id,revision,"val","full",dim,perf)
            va=self.extract(encoder,pp,"dino",model_id,revision,"val","augmix",dim,perf)
            tf=self.extract(encoder,pp,"dino",model_id,revision,"test","full",dim,perf)
        finally:
            if "encoder" in locals():
                del encoder
            del model
            release_gpu()
        name="dino_"+self.settings["dino_variant"]+"_probe"
        self.fit_and_export(name,train,vf,tf,va,dict(model_id=model_id,revision=revision))
        branches=[self.load_probabilities(n) for n in ("siglip_zero_shot","siglip_probe",name)]
        self.export_probabilities("vit_ensemble",blend_probabilities([b[0] for b in branches],[.5,.25,.25]),
                                  blend_probabilities([b[1] for b in branches],[.5,.25,.25]))
        atomic_json(dict(branches=["siglip_zero_shot","siglip_probe",name],weights=[.5,.25,.25],
                         chosen="fixed in advance; not optimized on Kaggle"),self.root/"ensemble.json")

    def bundle(self):
        path=self.root/"submissions_vit.zip"
        patterns=("submission_*.csv","*_metrics.json","*_benchmark.json","*_preprocessing_check.json","experiment.json","ensemble.json")
        with zipfile.ZipFile(path,"w",zipfile.ZIP_DEFLATED) as z:
            for pattern in patterns:
                for file in self.root.glob(pattern):
                    z.write(file,file.name)
        return path
