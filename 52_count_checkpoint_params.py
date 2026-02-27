import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from huggingface_hub import snapshot_download
from safetensors import safe_open


def _resolve_model_dir(model_or_dir: str) -> Path:
    p = Path(model_or_dir)
    if p.exists() and p.is_dir():
        return p
    local_dir = snapshot_download(repo_id=model_or_dir, ignore_patterns=["*.bin"])
    return Path(local_dir)


def _iter_safetensors_files(model_dir: Path) -> Iterable[Path]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        idx = json.loads(index_path.read_text(encoding="utf-8"))
        files = sorted(set(idx["weight_map"].values()))
        for fn in files:
            fp = model_dir / fn
            if fp.exists():
                yield fp
        return
    single = model_dir / "model.safetensors"
    if single.exists():
        yield single
        return
    for fp in sorted(model_dir.glob("*.safetensors")):
        yield fp


def _bucket(name: str) -> str:
    if name.startswith("talker.model.layers."):
        return "talker.layers"
    if name.startswith("talker.model.codec_embedding"):
        return "talker.codec_embedding"
    if name.startswith("talker.model.text_embedding"):
        return "talker.text_embedding"
    if name.startswith("talker.model.norm"):
        return "talker.final_norm"
    if name.startswith("talker.text_projection"):
        return "talker.text_projection"
    if name.startswith("talker.codec_head"):
        return "talker.codec_head"

    if name.startswith("talker.code_predictor.model.layers."):
        return "code_predictor.layers"
    if name.startswith("talker.code_predictor.model.codec_embedding"):
        return "code_predictor.input_embeddings"
    if name.startswith("talker.code_predictor.lm_heads"):
        return "code_predictor.output_heads"
    if name.startswith("talker.code_predictor.model.norm"):
        return "code_predictor.final_norm"
    if name.startswith("talker.code_predictor.small_to_mtp_projection"):
        return "code_predictor.small_to_mtp_projection"

    if name.startswith("speaker_encoder."):
        return "speaker_encoder"
    if name.startswith("speech_tokenizer."):
        return "speech_tokenizer"
    return "other"


def _count_params(model_dir: Path) -> dict[str, int]:
    out = defaultdict(int)
    files = list(_iter_safetensors_files(model_dir))
    if not files:
        raise FileNotFoundError(f"No safetensors found under: {model_dir}")
    for fp in files:
        with safe_open(str(fp), framework="pt", device="cpu") as f:
            for k in f.keys():
                shape = f.get_slice(k).get_shape()
                numel = 1
                for d in shape:
                    numel *= int(d)
                out[_bucket(k)] += numel
                out["__total__"] += numel
    return dict(out)


def _ratio(a: int, b: int) -> str:
    if a == 0:
        return "-"
    return f"{(b / a):.3f}x"


def main() -> None:
    p = argparse.ArgumentParser(description="Count parameters from Qwen3-TTS safetensors by module bucket.")
    p.add_argument("--model-17b", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    p.add_argument("--model-06b", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    p.add_argument("--output-json", default=None)
    args = p.parse_args()

    dir17 = _resolve_model_dir(args.model_17b)
    dir06 = _resolve_model_dir(args.model_06b)
    c17 = _count_params(dir17)
    c06 = _count_params(dir06)

    print("[Resolved]")
    print(f"1.7B dir: {dir17}")
    print(f"0.6B dir: {dir06}")

    keys = sorted(set(c17) | set(c06))
    print("\n[Param Count by Bucket]")
    print("bucket | 1.7B -> 0.6B | ratio(0.6/1.7)")
    print("-" * 72)
    for k in keys:
        a = int(c17.get(k, 0))
        b = int(c06.get(k, 0))
        print(f"{k} | {a} -> {b} | {_ratio(a, b)}")

    if args.output_json:
        out = {
            "model_17b": args.model_17b,
            "model_06b": args.model_06b,
            "resolved_17b": str(dir17),
            "resolved_06b": str(dir06),
            "counts_17b": c17,
            "counts_06b": c06,
        }
        op = Path(args.output_json)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[OK] wrote -> {op}")


if __name__ == "__main__":
    main()
