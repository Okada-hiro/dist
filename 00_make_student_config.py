import argparse
import copy
import json
from pathlib import Path


def _round_to_multiple(v: int, m: int) -> int:
    return max(m, int(round(v / m) * m))


def _pick_key(d: dict, candidates: list[str]) -> str | None:
    for k in candidates:
        if k in d:
            return k
    return None


def _find_model_cfg_path(cfg: dict) -> tuple[list[str], dict]:
    # Common places for architecture fields in TTS configs.
    candidates: list[tuple[list[str], dict]] = [
        ([], cfg),
        (["talker_config"], cfg.get("talker_config", {})),
        (["model"], cfg.get("model", {})),
        (["text_config"], cfg.get("text_config", {})),
    ]
    for path, d in candidates:
        if not isinstance(d, dict):
            continue
        has_layers = _pick_key(d, ["num_hidden_layers", "n_layer"]) is not None
        has_hidden = _pick_key(d, ["hidden_size", "n_embd"]) is not None
        has_heads = _pick_key(d, ["num_attention_heads", "n_head"]) is not None
        if has_layers and has_hidden and has_heads:
            return path, d
    raise ValueError(
        "Teacher config does not contain expected layers/hidden/heads fields at top-level or talker_config/model/text_config."
    )


def build_student_config(
    teacher_cfg: dict,
    layer_ratio: float = 0.5,
    hidden_ratio: float = 0.8,
    ffn_ratio: float = 0.8,
    target_layers: int | None = None,
    layers_only: bool = False,
) -> tuple[dict, list[str]]:
    cfg = copy.deepcopy(teacher_cfg)
    path, model_cfg = _find_model_cfg_path(cfg)

    num_layers_key = _pick_key(model_cfg, ["num_hidden_layers", "n_layer"])
    hidden_key = _pick_key(model_cfg, ["hidden_size", "n_embd"])
    heads_key = _pick_key(model_cfg, ["num_attention_heads", "n_head"])
    ffn_key = _pick_key(model_cfg, ["intermediate_size", "ffn_hidden_size", "n_inner"])

    if num_layers_key is None or hidden_key is None or heads_key is None:
        raise ValueError("Teacher config must include layers/hidden/heads fields.")

    teacher_layers = int(model_cfg[num_layers_key])
    teacher_hidden = int(model_cfg[hidden_key])
    teacher_heads = int(model_cfg[heads_key])

    student_layers = int(target_layers) if target_layers is not None else max(1, int(round(teacher_layers * layer_ratio)))
    model_cfg[num_layers_key] = max(1, student_layers)

    if not layers_only:
        # Keep head_dim close to teacher by shrinking heads with hidden.
        teacher_head_dim = teacher_hidden // teacher_heads
        raw_hidden = max(teacher_heads, int(round(teacher_hidden * hidden_ratio)))
        student_heads = max(1, int(round(raw_hidden / teacher_head_dim)))
        student_hidden = _round_to_multiple(raw_hidden, student_heads)

        model_cfg[hidden_key] = student_hidden
        model_cfg[heads_key] = student_heads

        # Keep kv heads <= attention heads when present.
        if "num_key_value_heads" in model_cfg:
            kv = int(model_cfg["num_key_value_heads"])
            model_cfg["num_key_value_heads"] = min(kv, student_heads)

        if ffn_key is not None and ffn_key in model_cfg:
            teacher_ffn = int(model_cfg[ffn_key])
            # Many models prefer FFN multiple of 256.
            model_cfg[ffn_key] = _round_to_multiple(max(256, int(round(teacher_ffn * ffn_ratio))), 256)

        # Optional rope/head consistency fields.
        if "head_dim" in model_cfg:
            model_cfg["head_dim"] = model_cfg[hidden_key] // model_cfg[heads_key]

    return cfg, path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-config", required=True, help="Path to teacher config.json")
    p.add_argument("--output", required=True, help="Output student config.json")
    p.add_argument("--layer-ratio", type=float, default=0.5)
    p.add_argument("--hidden-ratio", type=float, default=0.8)
    p.add_argument("--ffn-ratio", type=float, default=0.8)
    p.add_argument("--target-layers", type=int, default=None, help="Set absolute layer count (overrides --layer-ratio).")
    p.add_argument("--layers-only", action="store_true", help="Only modify num_hidden_layers and keep all other fields unchanged.")
    args = p.parse_args()

    teacher_cfg = json.loads(Path(args.teacher_config).read_text(encoding="utf-8"))
    student_cfg, cfg_path = build_student_config(
        teacher_cfg,
        layer_ratio=args.layer_ratio,
        hidden_ratio=args.hidden_ratio,
        ffn_ratio=args.ffn_ratio,
        target_layers=args.target_layers,
        layers_only=args.layers_only,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(student_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    src_cfg = teacher_cfg
    dst_cfg = student_cfg
    for k in cfg_path:
        src_cfg = src_cfg[k]
        dst_cfg = dst_cfg[k]

    layer_key = _pick_key(src_cfg, ["num_hidden_layers", "n_layer"])
    hidden_key = _pick_key(src_cfg, ["hidden_size", "n_embd"])
    ffn_key = _pick_key(src_cfg, ["intermediate_size", "ffn_hidden_size", "n_inner"])

    print("[OK] student config generated")
    print(f"config_scope: {'/'.join(cfg_path) if cfg_path else 'root'}")
    if args.layers_only:
        print("mode: layers_only")
    if layer_key and hidden_key:
        print(f"layers: {src_cfg[layer_key]} -> {dst_cfg[layer_key]}")
        print(f"hidden: {src_cfg[hidden_key]} -> {dst_cfg[hidden_key]}")
    if ffn_key and ffn_key in src_cfg:
        print(f"ffn: {src_cfg[ffn_key]} -> {dst_cfg[ffn_key]}")


if __name__ == "__main__":
    main()
