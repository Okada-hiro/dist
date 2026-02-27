import argparse
import json
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download


KEYS_OF_INTEREST = [
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "num_code_groups",
    "vocab_size",
]

CODE_PREDICTOR_KEYS = [
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "vocab_size",
    "num_code_groups",
]


def _load_config(model_or_dir: str) -> dict[str, Any]:
    p = Path(model_or_dir)
    if p.exists() and p.is_dir() and (p / "config.json").exists():
        return json.loads((p / "config.json").read_text(encoding="utf-8"))
    config_path = hf_hub_download(repo_id=model_or_dir, filename="config.json")
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def _pick(d: dict[str, Any], key: str, default: Any = None) -> Any:
    return d.get(key, default) if isinstance(d, dict) else default


def _ratio_str(a: Any, b: Any) -> str:
    try:
        a = float(a)
        b = float(b)
        if a == 0:
            return "-"
        return f"{(b / a):.3f}x"
    except Exception:
        return "-"


def _emit_section(title: str) -> None:
    print(f"\n[{title}]")


def _emit_table(rows: list[tuple[str, Any, Any]]) -> None:
    maxk = max(len(k) for k, _, _ in rows) if rows else 10
    print(f"{'key'.ljust(maxk)} | 1.7B -> 0.6B | ratio(0.6/1.7)")
    print("-" * (maxk + 36))
    for k, v17, v06 in rows:
        print(f"{k.ljust(maxk)} | {str(v17)} -> {str(v06)} | {_ratio_str(v17, v06)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Compare Qwen3-TTS model configs (1.7B vs 0.6B).")
    p.add_argument("--model-17b", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    p.add_argument("--model-06b", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    p.add_argument("--output-json", default=None, help="Optional path to save full comparison JSON")
    args = p.parse_args()

    cfg17 = _load_config(args.model_17b)
    cfg06 = _load_config(args.model_06b)

    t17 = _pick(cfg17, "talker_config", {})
    t06 = _pick(cfg06, "talker_config", {})
    cp17 = _pick(t17, "code_predictor_config", {})
    cp06 = _pick(t06, "code_predictor_config", {})

    _emit_section("Models")
    print(f"1.7B: {args.model_17b}")
    print(f"0.6B: {args.model_06b}")

    _emit_section("Top-level")
    top_rows = [
        ("tokenizer_type", _pick(cfg17, "tokenizer_type"), _pick(cfg06, "tokenizer_type")),
        ("tts_model_size", _pick(cfg17, "tts_model_size"), _pick(cfg06, "tts_model_size")),
        ("tts_model_type", _pick(cfg17, "tts_model_type"), _pick(cfg06, "tts_model_type")),
    ]
    _emit_table(top_rows)

    _emit_section("talker_config")
    talker_rows = [(k, _pick(t17, k), _pick(t06, k)) for k in KEYS_OF_INTEREST]
    _emit_table(talker_rows)

    _emit_section("talker_config.code_predictor_config")
    cp_rows = [(k, _pick(cp17, k), _pick(cp06, k)) for k in CODE_PREDICTOR_KEYS]
    _emit_table(cp_rows)

    _emit_section("Recommended First Student Cut")
    l17 = _pick(t17, "num_hidden_layers")
    h17 = _pick(t17, "hidden_size")
    f17 = _pick(t17, "intermediate_size")
    print(f"from 1.7B talker: layers={l17}, hidden={h17}, ffn={f17}")
    try:
        print(
            "candidate Student-S: "
            f"layers={max(1, round(float(l17) * 0.5))}, "
            f"hidden={round(float(h17) * 0.8)}, "
            f"ffn={round(float(f17) * 0.8)}"
        )
    except Exception:
        print("candidate Student-S: unable to compute (missing numeric keys)")

    if args.output_json:
        out = {
            "model_17b": args.model_17b,
            "model_06b": args.model_06b,
            "config_17b": cfg17,
            "config_06b": cfg06,
            "talker_compare": {k: {"v17": _pick(t17, k), "v06": _pick(t06, k)} for k in KEYS_OF_INTEREST},
            "code_predictor_compare": {
                k: {"v17": _pick(cp17, k), "v06": _pick(cp06, k)} for k in CODE_PREDICTOR_KEYS
            },
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[OK] wrote comparison json -> {out_path}")


if __name__ == "__main__":
    main()

