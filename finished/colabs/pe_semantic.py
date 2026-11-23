"""PE-Core-G zero-shot and conservative local views; no target-label training.

Uses pinned official PE code/weights and unchanged public OpenCLIP templates.
The previous SigLIP/DINO experiment and its output files are read-only inputs.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import semantic_vit as sv

VERSION = "pe_semantic_v1"
MODEL_NAME = "PE-Core-G14-448"
MODEL_ID = "facebook/" + MODEL_NAME
PE_REVISION = "3e352cca660658d4b5c90f42a7808b11469e4c66"
PE_FILES = {
    "core/__init__.py": "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",
    "core/vision_encoder/__init__.py": "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",
    "core/vision_encoder/pe.py": "b190fe9660c2da5df5bada28f7dc4e87c627b4ab",
    "core/vision_encoder/config.py": "d814904602f1014b49b07f2208b96fa48c7d8ff5",
    "core/vision_encoder/rope.py": "453da56c96ee2e0d2e83df34f69be84306c22619",
    "core/vision_encoder/tokenizer.py": "e3a5547a8543ba7995b2201d0b2604295f965d8c",
    "core/vision_encoder/transforms.py": "a84f2be4f62f629270863cc8c44c430f34165ad2",
    "core/vision_encoder/bpe_simple_vocab_16e6.txt.gz": "7b5088a527f720063f044eb928eee315f63b2fc0",
    "LICENSE.PE": "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64",
}


def git_blob_hash(payload):
    return hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()


def prepare_pe_source(folder):
    """Fetch only the audited image/text encoder files, not the full PLM stack."""
    import requests
    folder = Path(folder).resolve()
    if "core" in sys.modules:
        existing = getattr(sys.modules["core"], "__file__", None)
        if existing is None or Path(existing).resolve() != folder / "core/__init__.py":
            raise RuntimeError("Ya hay otro paquete 'core' importado. Usa una sesión Colab separada para PE.")
    for relative, expected in PE_FILES.items():
        path = folder / relative
        if path.exists() and git_blob_hash(path.read_bytes()) == expected:
            continue
        url = f"https://raw.githubusercontent.com/facebookresearch/perception_models/{PE_REVISION}/{relative}"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        if git_blob_hash(response.content) != expected:
            raise RuntimeError(f"PE upstream content mismatch: {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    return folder


def canonical_state_dict(state):
    if not isinstance(state, dict):
        raise ValueError("Checkpoint must be a state dictionary")
    for key in ("state_dict", "weights"):
        if key in state:
            state = state[key]
            break
    if not isinstance(state, dict) or not state:
        raise ValueError("Empty / invalid PE checkpoint")
    result = {}
    for key, value in state.items():
        key = key.removeprefix("module.")
        if key in result or not isinstance(value, torch.Tensor):
            raise ValueError("Duplicate or non-tensor checkpoint entry")
        result[key] = value
    return result


def load_pe(root):
    from huggingface_hub import hf_hub_download
    import core.vision_encoder.pe as pe
    revision = sv.pinned_revision(MODEL_ID, root, token=False)
    checkpoint = hf_hub_download(MODEL_ID, MODEL_NAME + ".pt", revision=revision, token=False)
    # Official constructor initializes RoPE's non-registered state correctly.
    # No meta-device shortcuts and no permissive missing-key loading.
    model = pe.CLIP.from_config(MODEL_NAME, pretrained=False)
    try:
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    except RuntimeError as exc:
        if "mmap can only be used" not in str(exc):
            raise
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = canonical_state_dict(loaded)
    model.load_state_dict(state, strict=True, assign=True)
    del state,loaded
    if model.image_size != 448 or model.context_length != 72:
        raise ValueError("Unexpected PE architecture")
    counts = dict(total=sum(p.numel() for p in model.parameters()),
                  visual=sum(p.numel() for p in model.visual.parameters()))
    sv.atomic_json(dict(model_id=MODEL_ID, revision=revision, source_revision=PE_REVISION,
                        parameters=counts, weights="fp32", compute="bf16_autocast"), Path(root)/"pe_model.json")
    print("PE parámetros:", counts)
    # FP32 weights preserve the official numerical path; matrix operations use BF16.
    return model.requires_grad_(False).eval().to(device="cuda",dtype=torch.float32), revision


class PEEncoder(nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    def forward(self, pixels):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            features = self.visual(pixels)
        return F.normalize(features.float(), dim=-1)


@torch.inference_mode()
def text_prototypes(model, templates):
    from core.vision_encoder.transforms import get_text_tokenizer
    tokenizer = get_text_tokenizer(model.context_length)
    prototypes = []
    for name in sv.CLASSES:
        texts = [t.format(c=name) for t in templates]
        tokens = tokenizer(texts).to("cuda")
        vectors = []
        for batch in tokens.split(32):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                features = model.encode_text(batch)
            vectors.append(F.normalize(features.float(), dim=-1))
        prototypes.append(F.normalize(torch.cat(vectors).mean(0), dim=-1))
    return torch.stack(prototypes).cpu()


def uncertainty_indices(logits, fraction):
    """Unlabeled selection: smallest top-1/top-2 margins, fixed compute cap."""
    logits = np.asarray(logits)
    if logits.ndim != 2 or logits.shape[1] != 10 or not np.isfinite(logits).all():
        raise ValueError("Expected finite N x 10 logits")
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be in [0,1]")
    top = np.partition(logits, -2, axis=1)[:, -2:]
    margin = top.max(1) - top.min(1)
    count = int(math.ceil(len(logits)*fraction))
    return np.sort(np.argsort(margin, kind="stable")[:count])


def crop_boxes(size=28, width=32):
    if not isinstance(size, int) or not 1 <= size <= width:
        raise ValueError("Invalid crop size")
    edge, center = width-size, (width-size)//2
    return [(0,0,size,size), (0,edge,size,size), (edge,0,size,size),
            (edge,edge,size,size), (center,center,size,size)]


def crop_images(images, box):
    top,left,height,width = box
    if (images.ndim != 4 or images.dtype != np.uint8 or images.shape[-1] != 3
            or min(top,left) < 0 or min(height,width) < 1
            or top+height > images.shape[1] or left+width > images.shape[2]):
        raise ValueError("Invalid crop / image dimensions")
    return np.ascontiguousarray(images[:,top:top+height,left:left+width])


def merge_local_logits(global_logits, indices, local_logits, weight=.25):
    base = np.asarray(global_logits, dtype=np.float32)
    local = np.asarray(local_logits, dtype=np.float32)
    indices = np.asarray(indices)
    if (not 0 <= weight <= 1 or indices.ndim != 1 or not np.issubdtype(indices.dtype,np.integer)
            or len(np.unique(indices)) != len(indices) or (indices < 0).any() or (indices >= len(base)).any()
            or local.shape != (len(indices),10) or base.shape != (len(base),10)
            or not np.isfinite(base).all() or not np.isfinite(local).all()):
        raise ValueError("Invalid local/global predictions")
    result = base.copy()
    result[indices] = (1-weight)*base[indices] + weight*local
    return result


def stress_images(images, mode):
    """Fixed, non-generative diagnostics; not a replacement for target validation."""
    from PIL import Image, ImageFilter, ImageOps
    result = []
    for raw in images:
        image = Image.fromarray(raw)
        if mode == "grayscale":
            image = ImageOps.grayscale(image).convert("RGB")
        elif mode == "posterize4":
            image = ImageOps.posterize(image, 4)
        elif mode == "blur05":
            image = image.filter(ImageFilter.GaussianBlur(.5))
        else:
            raise ValueError("Unknown diagnostic")
        result.append(np.array(image))
    return np.stack(result)


def load_previous_siglip(folder, data):
    folder = Path(folder)
    manifest = json.loads((folder/"experiment.json").read_text())
    if (manifest["data_hashes"] != data["hashes"] or manifest["classes"] != list(sv.CLASSES)
            or manifest["validation_parts"] != [p.tolist() for p in data["parts"]]):
        raise ValueError("Previous SigLIP run uses different data/classes/validation split")
    with np.load(folder/"siglip_zero_shot_probabilities.npz", allow_pickle=False) as f:
        ids, val, test = f["test_ids"], f["val"].copy(), f["test"].copy()
    if not np.array_equal(ids,data["test_ids"]) or val.shape != (len(data["val"]),10) or test.shape != (len(ids),10):
        raise ValueError("Previous SigLIP predictions are not aligned")
    sv.blend_probabilities([val],[1.])
    sv.blend_probabilities([test],[1.])
    provenance = dict(folder=str(folder), probability_sha256=sv.file_hash(folder/"siglip_zero_shot_probabilities.npz"),
                      manifest_sha256=sv.file_hash(folder/"experiment.json"))
    return (val,test),provenance


class PEExperiment(sv.Experiment):
    def __init__(self, data, settings, run_root, cache_root, scratch, budget):
        settings = dict(settings, pe_pipeline=VERSION, pe_source_revision=PE_REVISION,
                        pe_dependencies={p:importlib.metadata.version(p) for p in ("timm","einops","ftfy","regex","Pillow")})
        if Path(run_root).resolve() == Path(settings["previous_run"]).resolve():
            raise ValueError("El nuevo RUN_NAME debe ser distinto al del experimento anterior")
        if not 0 < settings["uncertain_fraction"] <= 1 or not 0 <= settings["crop_weight"] <= 1:
            raise ValueError("Invalid local-view settings")
        crop_boxes(settings["crop_size"])
        super().__init__(data,settings,run_root,cache_root,scratch,budget)

    def view_logits(self, encoder, pp, prototypes, revision, perf, split, view="full", indices=None, crop=None, stress=None):
        images = self.data[split] if indices is None else self.data[split][indices]
        if stress is not None:
            images = stress_images(images,stress)
        if crop is not None:
            images = crop_images(images,crop)
        spec = self.feature_spec("pe",MODEL_ID,revision,pp.spec,split,view)
        spec.update(pe_version=VERSION, pe_code=PE_REVISION, dependencies=self.settings["pe_dependencies"],
                    selected_rows_sha256=None if indices is None else sv.digest(np.asarray(indices).tolist()),
                    crop=None if crop is None else list(crop), stress=stress, precision="fp32_weights_bf16_autocast_fp16_cache")
        features = sv.extract_cached(encoder,pp,images,spec,self.cache,self.scratch,
                                     perf["batch"],1280,self.budget,perf["images_per_second"])
        return (torch.from_numpy(features.astype(np.float32)) @ prototypes.T).numpy()

    def local_logits(self, encoder, pp, prototypes, revision, perf, split, baseline):
        indices = uncertainty_indices(baseline,self.settings["uncertain_fraction"])
        sv.atomic_json(dict(indices=indices.tolist(), selection="smallest top1-top2 cosine margin",
                            uses_labels=False, fraction=self.settings["uncertain_fraction"]),self.root/f"pe_{split}_local_rows.json")
        views = []
        for number,box in enumerate(crop_boxes(self.settings["crop_size"]),1):
            print(f"PE {split}: recorte {number}/5, {box}, {len(indices):,} imágenes seleccionadas.")
            views.append(self.view_logits(encoder,pp,prototypes,revision,perf,split,indices=indices,crop=box))
        return merge_local_logits(baseline,indices,np.mean(views,axis=0),self.settings["crop_weight"])

    def diagnostic(self, encoder, pp, prototypes, revision, perf, baseline_val):
        audit = self.data["parts"][2]
        rng = np.random.default_rng(self.settings["seed"])
        chosen = np.sort(np.concatenate([rng.permutation(audit[self.data["val_labels"][audit] == label])[:100]
                                         for label in range(10)]))
        y = self.data["val_labels"][chosen]
        rows = [dict(mode="original_full_and_flip",rows=len(chosen),**sv.scores(baseline_val[chosen],y))]
        for mode in ("grayscale","posterize4","blur05"):
            logits = np.mean([self.view_logits(encoder,pp,prototypes,revision,perf,"val",view,chosen,stress=mode)
                              for view in ("full","hflip")],axis=0)
            rows.append(dict(mode=mode,rows=len(chosen),**sv.scores(logits,y)))
            sv.atomic_json(dict(rows=rows,labels="original validation labels",used_for_selection=False,
                                limitation="Low-level corruptions do not reproduce toys, drawings or tattoos"),
                           self.root/"pe_stress_audit.json")
        print("Diagnóstico de robustez:", rows)

    def blend_previous(self):
        previous = Path(self.settings["previous_run"])
        if not (previous/"siglip_zero_shot_probabilities.npz").exists():
            print("PE listo. Falta el archivo de probabilidades SigLIP en",previous,
                  "\nCorrige previous_run o usa solo los CSV PE; no se inventan resultados anteriores.")
            return
        a,provenance = load_previous_siglip(previous,self.data)
        record=self.root/"previous_siglip_provenance.json"
        if record.exists() and json.loads(record.read_text()) != provenance:
            raise ValueError("Las predicciones SigLIP anteriores han cambiado. Usa otro RUN_NAME para esta nueva combinación.")
        sv.atomic_json(provenance,record)
        b = self.load_probabilities("pe_zero_shot")
        self.export_probabilities("siglip_pe_zero_shot",sv.blend_probabilities([a[0],b[0]],[.5,.5]),
                                  sv.blend_probabilities([a[1],b[1]],[.5,.5]))
        sv.atomic_json(dict(branches=["previous_siglip_zero_shot","pe_zero_shot"],weights=[.5,.5],
                            chosen="fixed before PE Kaggle results; not claimed optimal",previous=provenance),self.root/"ensemble.json")

    def run(self, source_folder="/content/recvis_pe_source"):
        prepare_pe_source(source_folder)
        from core.vision_encoder.transforms import get_image_transform
        self.budget.require(0,"carga PE")
        model,revision = load_pe(self.root)
        encoder = None
        try:
            prototypes = text_prototypes(model,self.templates)
            sv.atomic_torch(dict(prototypes=prototypes,model_id=MODEL_ID,revision=revision,
                                 source_revision=PE_REVISION),self.root/"pe_text_prototypes.pt")
            encoder = PEEncoder(model.visual).eval()
            del model
            model = None
            sv.release_gpu()
            reference_transform = get_image_transform(448,center_crop=False)
            def reference(images,return_tensors):
                return {"pixel_values":torch.stack([reference_transform(x) for x in images])}
            # Exact official PIL transform from the start: bilinear, no center crop, mean/std 0.5.
            pp = sv.ReferencePreprocess(reference,dict(size=[448,448],resample=2,mean=[.5]*3,std=[.5]*3,
                                                       scale=1/255,antialias=True),device="cuda")
            perf,trials = sv.benchmark(encoder,pp,self.data["train"],[2,4,8,16,32],self.budget)
            sv.atomic_json(dict(best=perf,trials=trials,source_revision=PE_REVISION),self.root/"pe_benchmark.json")
            n = len(self.data["val"])+len(self.data["test"])
            planned = 2*n
            if self.settings["run_local_views"]: planned += 5*n*self.settings["uncertain_fraction"]
            if self.settings["run_stress_audit"]: planned += 6000
            print(f"Estimación orientativa sin caché: {1.3*planned/perf['images_per_second']/3600:.2f} h, más descarga/escritura. "
                  "El presupuesto se comprueba por bloques.")
            val = np.mean([self.view_logits(encoder,pp,prototypes,revision,perf,"val",view)
                           for view in ("full","hflip")],axis=0)
            test = np.mean([self.view_logits(encoder,pp,prototypes,revision,perf,"test",view)
                            for view in ("full","hflip")],axis=0)
            self.export("pe_zero_shot",val,test)
            self.blend_previous()
            if self.settings["run_local_views"]:
                local_v = self.local_logits(encoder,pp,prototypes,revision,perf,"val",val)
                local_t = self.local_logits(encoder,pp,prototypes,revision,perf,"test",test)
                self.export("pe_local_views",local_v,local_t)
            if self.settings["run_stress_audit"]:
                self.diagnostic(encoder,pp,prototypes,revision,perf,val)
        finally:
            del encoder,model
            sv.release_gpu()

    def bundle(self):
        path = super().bundle()
        path = path.replace(self.root/"submissions_pe.zip")
        import zipfile
        with zipfile.ZipFile(path,"a",zipfile.ZIP_DEFLATED) as archive:
            for name in ("pe_stress_audit.json","pe_model.json","code_revision.json"):
                file=self.root/name
                if file.exists(): archive.write(file,file.name)
        return path
