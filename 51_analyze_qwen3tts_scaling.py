import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download


def _load_config(model_or_dir: str) -> dict[str, Any]:
    p = Path(model_or_dir)
    if p.exists() and p.is_dir() and (p / "config.json").exists():
        return json.loads((p / "config.json").read_text(encoding="utf-8"))
    config_path = hf_hub_download(repo_id=model_or_dir, filename="config.json")
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def _pick(d: dict[str, Any], key: str, default: Any = None) -> Any:
    return d.get(key, default) if isinstance(d, dict) else default


def _head_dim(cfg: dict[str, Any]) -> int:
    hd = _pick(cfg, "head_dim")
    if hd is not None:
        return int(hd)
    h = int(_pick(cfg, "hidden_size"))
    nh = int(_pick(cfg, "num_attention_heads"))
    return h // nh


@dataclass
class BlockBreakdown:
    attn_q: int
    attn_k: int
    attn_v: int
    attn_o: int
    mlp_gate: int
    mlp_up: int
    mlp_down: int
    norms: int

    @property
    def total(self) -> int:
        return (
            self.attn_q
            + self.attn_k
            + self.attn_v
            + self.attn_o
            + self.mlp_gate
            + self.mlp_up
            + self.mlp_down
            + self.norms
        )


def _decoder_block_params(cfg: dict[str, Any]) -> BlockBreakdown:
    h = int(_pick(cfg, "hidden_size"))
    i = int(_pick(cfg, "intermediate_size"))
    n_heads = int(_pick(cfg, "num_attention_heads"))
    n_kv = int(_pick(cfg, "num_key_value_heads"))
    hd = _head_dim(cfg)
    q_out = n_heads * hd
    kv_out = n_kv * hd
    return BlockBreakdown(
        attn_q=h * q_out,
        attn_k=h * kv_out,
        attn_v=h * kv_out,
        attn_o=q_out * h,
        mlp_gate=h * i,
        mlp_up=h * i,
        mlp_down=i * h,
        norms=2 * h,
    )


def _talker_estimate(tcfg: dict[str, Any]) -> dict[str, int]:
    block = _decoder_block_params(tcfg)
    n_layers = int(_pick(tcfg, "num_hidden_layers"))
    h = int(_pick(tcfg, "hidden_size"))
    v = int(_pick(tcfg, "vocab_size"))
    text_h = int(_pick(tcfg, "text_hidden_size", h))
    text_v = int(_pick(tcfg, "text_vocab_size", 0))

    # talker model + heads
    out: dict[str, int] = {
        "talker.layers.total": block.total * n_layers,
        "talker.layers.attn_q": block.attn_q * n_layers,
        "talker.layers.attn_kv": (block.attn_k + block.attn_v) * n_layers,
        "talker.layers.attn_o": block.attn_o * n_layers,
        "talker.layers.mlp": (block.mlp_gate + block.mlp_up + block.mlp_down) * n_layers,
        "talker.layers.norms": block.norms * n_layers,
        "talker.codec_embedding": v * h,
        "talker.codec_head": h * v,
        "talker.final_norm": h,
        # text_projection = Linear(text_h, text_h, bias=True) + Linear(text_h, h, bias=True)
        "talker.text_projection": (text_h * text_h + text_h) + (text_h * h + h),
    }
    if text_v > 0:
        out["talker.text_embedding"] = text_v * text_h
    return out


def _code_predictor_estimate(tcfg: dict[str, Any], cpcfg: dict[str, Any]) -> dict[str, int]:
    block = _decoder_block_params(cpcfg)
    n_layers = int(_pick(cpcfg, "num_hidden_layers"))
    cp_h = int(_pick(cpcfg, "hidden_size"))
    cp_v = int(_pick(cpcfg, "vocab_size"))
    groups = int(_pick(cpcfg, "num_code_groups", _pick(tcfg, "num_code_groups", 16)))
    talker_h = int(_pick(tcfg, "hidden_size"))
    n_pred = max(groups - 1, 0)

    out: dict[str, int] = {
        "code_predictor.layers.total": block.total * n_layers,
        "code_predictor.layers.attn_q": block.attn_q * n_layers,
        "code_predictor.layers.attn_kv": (block.attn_k + block.attn_v) * n_layers,
        "code_predictor.layers.attn_o": block.attn_o * n_layers,
        "code_predictor.layers.mlp": (block.mlp_gate + block.mlp_up + block.mlp_down) * n_layers,
        "code_predictor.layers.norms": block.norms * n_layers,
        "code_predictor.input_embeddings": n_pred * cp_v * talker_h,
        "code_predictor.output_heads": n_pred * cp_h * cp_v,
        "code_predictor.final_norm": cp_h,
    }
    if cp_h != talker_h:
        out["code_predictor.small_to_mtp_projection"] = talker_h * cp_h + cp_h
    return out


def _speaker_encoder_hint(cfg: dict[str, Any]) -> dict[str, int]:
    secfg = _pick(cfg, "speaker_encoder_config", {})
    enc_channels = _pick(secfg, "enc_channels", [])
    hint = {}
    if isinstance(enc_channels, list) and enc_channels:
        hint["speaker_encoder.channel_sum"] = int(sum(int(x) for x in enc_channels))
    return hint


def _sum(d: dict[str, int]) -> int:
    return int(sum(d.values()))


def _fmt_ratio(a: int, b: int) -> str:
    if a == 0:
        return "-"
    return f"{(b / a):.3f}x"


def main() -> None:
    p = argparse.ArgumentParser(description="Deep scaling analysis for Qwen3-TTS configs.")
    p.add_argument("--model-17b", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    p.add_argument("--model-06b", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    p.add_argument("--output-json", default=None)
    args = p.parse_args()

    cfg17 = _load_config(args.model_17b)
    cfg06 = _load_config(args.model_06b)

    t17 = _pick(cfg17, "talker_config", {})
    t06 = _pick(cfg06, "talker_config", {})
    cp17 = _pick(t17, "code_predictor_config", {})
    cp06 = _pick(t06, "code_predictor_config", {})

    t_est_17 = _talker_estimate(t17)
    t_est_06 = _talker_estimate(t06)
    cp_est_17 = _code_predictor_estimate(t17, cp17)
    cp_est_06 = _code_predictor_estimate(t06, cp06)
    sp_hint_17 = _speaker_encoder_hint(cfg17)
    sp_hint_06 = _speaker_encoder_hint(cfg06)

    total_17 = _sum(t_est_17) + _sum(cp_est_17)
    total_06 = _sum(t_est_06) + _sum(cp_est_06)

    print("[Models]")
    print(f"1.7B: {args.model_17b}")
    print(f"0.6B: {args.model_06b}")

    print("\n[Definition Check]")
    print(
        "hidden_size = Transformer hidden width (embedding/activation width), not parameter count itself.\n"
        "Parameters in linear layers mostly scale with in_dim*out_dim, i.e. often O(hidden_size^2)."
    )

    print("\n[Raw Config Ratios]")
    for k in ("num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads"):
        a = int(_pick(t17, k, 0))
        b = int(_pick(t06, k, 0))
        print(f"talker.{k}: {a} -> {b} ({_fmt_ratio(a, b)})")

    print("\n[Estimated Param Buckets: Talker]")
    keys = sorted(set(t_est_17) | set(t_est_06))
    for k in keys:
        a = int(t_est_17.get(k, 0))
        b = int(t_est_06.get(k, 0))
        print(f"{k}: {a} -> {b} ({_fmt_ratio(a, b)})")

    print("\n[Estimated Param Buckets: Code Predictor]")
    keys = sorted(set(cp_est_17) | set(cp_est_06))
    for k in keys:
        a = int(cp_est_17.get(k, 0))
        b = int(cp_est_06.get(k, 0))
        print(f"{k}: {a} -> {b} ({_fmt_ratio(a, b)})")

    print("\n[Estimated Total (talker + code_predictor only)]")
    print(f"total_estimated: {total_17} -> {total_06} ({_fmt_ratio(total_17, total_06)})")
    print(
        "note: this excludes tokenizer/vocoder weights and uses formula-level estimates, but captures main scaling trend."
    )

    if sp_hint_17 or sp_hint_06:
        print("\n[Speaker Encoder Hint]")
        for k in sorted(set(sp_hint_17) | set(sp_hint_06)):
            a = int(sp_hint_17.get(k, 0))
            b = int(sp_hint_06.get(k, 0))
            print(f"{k}: {a} -> {b} ({_fmt_ratio(a, b)})")

    if args.output_json:
        out = {
            "model_17b": args.model_17b,
            "model_06b": args.model_06b,
            "talker_estimate_17b": t_est_17,
            "talker_estimate_06b": t_est_06,
            "code_predictor_estimate_17b": cp_est_17,
            "code_predictor_estimate_06b": cp_est_06,
            "total_estimated_17b": total_17,
            "total_estimated_06b": total_06,
            "speaker_encoder_hint_17b": sp_hint_17,
            "speaker_encoder_hint_06b": sp_hint_06,
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[OK] wrote analysis json -> {out_path}")


if __name__ == "__main__":
    main()
