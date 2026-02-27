import argparse
import json
import math
from pathlib import Path


def _round_to_multiple(v: int, m: int) -> int:
    return max(m, int(round(v / m) * m))


def build_student_config(
    teacher_cfg: dict,
    layer_ratio: float = 0.5,
    hidden_ratio: float = 0.8,
    ffn_ratio: float = 0.8,
) -> dict:
    cfg = dict(teacher_cfg)

    num_layers_key = "num_hidden_layers" if "num_hidden_layers" in cfg else "n_layer"
    hidden_key = "hidden_size" if "hidden_size" in cfg else "n_embd"
    ffn_key = "intermediate_size" if "intermediate_size" in cfg else None
    heads_key = "num_attention_heads" if "num_attention_heads" in cfg else "n_head"

    if num_layers_key not in cfg or hidden_key not in cfg or heads_key not in cfg:
        raise ValueError("Teacher config must include layers/hidden/heads fields.")

    teacher_layers = int(cfg[num_layers_key])
    teacher_hidden = int(cfg[hidden_key])
    teacher_heads = int(cfg[heads_key])

    student_layers = max(1, int(round(teacher_layers * layer_ratio)))
    # Keep head_dim close to teacher by shrinking heads with hidden.
    teacher_head_dim = teacher_hidden // teacher_heads
    raw_hidden = max(teacher_heads, int(round(teacher_hidden * hidden_ratio)))
    student_heads = max(1, int(round(raw_hidden / teacher_head_dim)))
    student_hidden = _round_to_multiple(raw_hidden, student_heads)

    cfg[num_layers_key] = student_layers
    cfg[hidden_key] = student_hidden
    cfg[heads_key] = student_heads

    # Keep kv heads <= attention heads when present.
    if "num_key_value_heads" in cfg:
        kv = int(cfg["num_key_value_heads"])
        cfg["num_key_value_heads"] = min(kv, student_heads)

    if ffn_key is not None and ffn_key in cfg:
        teacher_ffn = int(cfg[ffn_key])
        # Many models prefer FFN multiple of 256.
        cfg[ffn_key] = _round_to_multiple(max(256, int(round(teacher_ffn * ffn_ratio))), 256)

    # Optional rope/head consistency fields.
    if "head_dim" in cfg:
        cfg["head_dim"] = cfg[hidden_key] // cfg[heads_key]

    return cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-config", required=True, help="Path to teacher config.json")
    p.add_argument("--output", required=True, help="Output student config.json")
    p.add_argument("--layer-ratio", type=float, default=0.5)
    p.add_argument("--hidden-ratio", type=float, default=0.8)
    p.add_argument("--ffn-ratio", type=float, default=0.8)
    args = p.parse_args()

    teacher_cfg = json.loads(Path(args.teacher_config).read_text(encoding="utf-8"))
    student_cfg = build_student_config(
        teacher_cfg,
        layer_ratio=args.layer_ratio,
        hidden_ratio=args.hidden_ratio,
        ffn_ratio=args.ffn_ratio,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(student_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[OK] student config generated")
    print(f"layers: {teacher_cfg.get('num_hidden_layers', teacher_cfg.get('n_layer'))} -> {student_cfg.get('num_hidden_layers', student_cfg.get('n_layer'))}")
    print(f"hidden: {teacher_cfg.get('hidden_size', teacher_cfg.get('n_embd'))} -> {student_cfg.get('hidden_size', student_cfg.get('n_embd'))}")
    if "intermediate_size" in teacher_cfg:
        print(f"ffn: {teacher_cfg['intermediate_size']} -> {student_cfg['intermediate_size']}")


if __name__ == "__main__":
    main()
