import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download

from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration


def _resolve_model_dir(model_or_dir: str) -> Path:
    p = Path(model_or_dir)
    if p.exists() and p.is_dir():
        return p
    return Path(snapshot_download(model_or_dir))


def _sanitize_nested_model_type(cfg: dict) -> dict:
    cfg = dict(cfg)
    for k in ("talker_config", "speaker_encoder_config"):
        if isinstance(cfg.get(k), dict):
            cfg[k].pop("model_type", None)
    tc = cfg.get("talker_config")
    if isinstance(tc, dict) and isinstance(tc.get("code_predictor_config"), dict):
        tc["code_predictor_config"].pop("model_type", None)
    return cfg


def _build_uniform_layer_map(src_layers: int, tgt_layers: int) -> list[int]:
    if tgt_layers <= 1:
        return [src_layers - 1]
    idx = np.linspace(0, src_layers - 1, tgt_layers)
    mapped = np.round(idx).astype(int).tolist()
    # Ensure non-decreasing and in-range.
    mapped = [max(0, min(src_layers - 1, x)) for x in mapped]
    for i in range(1, len(mapped)):
        if mapped[i] < mapped[i - 1]:
            mapped[i] = mapped[i - 1]
    return mapped


def _copy_runtime_files(src_dir: Path, out_dir: Path) -> None:
    # Keep student model weights/config from save_pretrained(), copy only aux files.
    skip = {
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "model.safetensors.index.json",
    }
    for p in src_dir.iterdir():
        name = p.name
        if name in skip or name.startswith("model-"):
            continue
        dst = out_dir / name
        if dst.exists():
            continue
        if p.is_dir():
            shutil.copytree(p, dst)
        else:
            shutil.copy2(p, dst)


def main() -> None:
    p = argparse.ArgumentParser(description="Initialize student weights from 0.6B with layer thinning map.")
    p.add_argument("--source-model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base", help="HF id or local dir")
    p.add_argument("--student-config", required=True, help="Path to student config json (e.g. 20-layer config)")
    p.add_argument("--output-dir", required=True, help="Output model dir")
    p.add_argument(
        "--map-output-json",
        default=None,
        help="Optional path to save layer map details (default: <output-dir>/layer_map.json)",
    )
    args = p.parse_args()

    src_dir = _resolve_model_dir(args.source_model)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_dict = json.loads(Path(args.student_config).read_text(encoding="utf-8"))
    cfg_dict = _sanitize_nested_model_type(cfg_dict)
    student_cfg = Qwen3TTSConfig(**cfg_dict)
    student = Qwen3TTSForConditionalGeneration(student_cfg).cpu()

    source = Qwen3TTSForConditionalGeneration.from_pretrained(str(src_dir), torch_dtype=torch.float32).cpu()
    src_sd = source.state_dict()
    tgt_sd = student.state_dict()

    src_layers = int(source.config.talker_config.num_hidden_layers)
    tgt_layers = int(student.config.talker_config.num_hidden_layers)
    layer_map = _build_uniform_layer_map(src_layers, tgt_layers)

    layer_pat = re.compile(r"^talker\.model\.layers\.(\d+)(\..+)$")
    copied = 0
    random_init = 0
    shape_mismatch = 0

    with torch.no_grad():
        for tkey, tval in tgt_sd.items():
            skey = tkey
            m = layer_pat.match(tkey)
            if m:
                tidx = int(m.group(1))
                rest = m.group(2)
                if tidx >= len(layer_map):
                    random_init += 1
                    continue
                sidx = layer_map[tidx]
                skey = f"talker.model.layers.{sidx}{rest}"

            sval = src_sd.get(skey)
            if sval is None:
                random_init += 1
                continue
            if tuple(sval.shape) != tuple(tval.shape):
                shape_mismatch += 1
                random_init += 1
                continue
            tgt_sd[tkey] = sval.to(dtype=tval.dtype)
            copied += 1

    student.load_state_dict(tgt_sd, strict=True)
    student.save_pretrained(str(out_dir), safe_serialization=True)

    # Overwrite config.json with sanitized one to avoid nested model_type issues.
    (out_dir / "config.json").write_text(json.dumps(cfg_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    _copy_runtime_files(src_dir, out_dir)

    map_json = Path(args.map_output_json) if args.map_output_json else (out_dir / "layer_map.json")
    map_json.write_text(
        json.dumps(
            {
                "source_model": str(src_dir),
                "student_config": str(Path(args.student_config)),
                "source_layers": src_layers,
                "target_layers": tgt_layers,
                "talker_layer_map": layer_map,
                "copied_tensors": copied,
                "shape_mismatch_tensors": shape_mismatch,
                "random_init_tensors": random_init,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[OK] student initialized from source model")
    print(f"  source_dir       : {src_dir}")
    print(f"  output_dir       : {out_dir}")
    print(f"  source_layers    : {src_layers}")
    print(f"  target_layers    : {tgt_layers}")
    print(f"  talker_layer_map : {layer_map}")
    print(f"  copied_tensors   : {copied}")
    print(f"  random_init      : {random_init}")
    print(f"  shape_mismatch   : {shape_mismatch}")
    print(f"  map_json         : {map_json}")


if __name__ == "__main__":
    main()
