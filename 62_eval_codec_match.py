import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _normalize_language(raw: str | None, supported: set[str]) -> str:
    if not raw:
        return "Japanese" if "japanese" in {x.lower() for x in supported} else "Auto"
    s = str(raw).strip()
    if s in supported:
        return s
    low = s.lower()
    alias = {"ja": "Japanese", "japanese": "Japanese", "en": "English"}
    cand = alias.get(low, s)
    for k in supported:
        if k.lower() == cand.lower():
            return k
    return "Japanese" if "japanese" in {x.lower() for x in supported} else "Auto"


def _sanitize_config_for_qwen3(cfg: dict[str, Any]) -> dict[str, Any]:
    clean = json.loads(json.dumps(cfg))
    drop_keys = {"model_type", "dtype", "torch_dtype"}

    def _walk(x: Any) -> Any:
        if isinstance(x, dict):
            out = {}
            for k, v in x.items():
                if k in drop_keys:
                    continue
                out[k] = _walk(v)
            return out
        if isinstance(x, list):
            return [_walk(v) for v in x]
        return x

    return _walk(clean)


def _load_qwen3_model(
    model_path_or_id: str,
    device: str,
    torch_dtype: torch.dtype,
    attn_impl: str,
    processor_model: str | None = None,
):
    try:
        return Qwen3TTSModel.from_pretrained(
            model_path_or_id,
            device_map=device,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
        )
    except Exception as e:
        model_dir = Path(model_path_or_id)
        if not model_dir.exists():
            raise
        print(f"[WARN] wrapper load failed for {model_path_or_id}: {e}")
        print("[INFO] fallback: sanitize config and load core model directly.")

        cfg_path = model_dir / "config.json"
        raw_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        clean_cfg = _sanitize_config_for_qwen3(raw_cfg)
        with tempfile.TemporaryDirectory(prefix="qwen3tts_sanitized_") as td:
            td_path = Path(td)
            for item in model_dir.iterdir():
                if item.name == "config.json":
                    continue
                dst = td_path / item.name
                try:
                    os.symlink(item, dst, target_is_directory=item.is_dir())
                except OSError:
                    import shutil

                    if item.is_dir():
                        shutil.copytree(item, dst)
                    else:
                        shutil.copy2(item, dst)
            (td_path / "config.json").write_text(
                json.dumps(clean_cfg, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            cfg_obj = Qwen3TTSConfig(**clean_cfg)
            core_model = Qwen3TTSForConditionalGeneration.from_pretrained(
                str(td_path),
                config=cfg_obj,
                device_map=device,
                torch_dtype=torch_dtype,
                attn_implementation=attn_impl,
            )

        # Processor is usually available in local model dir when copied from base model.
        from transformers import AutoProcessor

        proc_src = processor_model or model_path_or_id
        processor = AutoProcessor.from_pretrained(proc_src, fix_mistral_regex=True)
        return Qwen3TTSModel(model=core_model, processor=processor, generate_defaults={})


def _gen_codec_ids_2d(
    model: Qwen3TTSModel,
    text: str,
    language: str,
    ref_audio: str | None,
    ref_text: str | None,
    non_streaming_mode: bool,
    max_new_tokens: int,
) -> list[list[int]]:
    prompt = None
    if ref_audio:
        prompt = model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=False,
        )

    input_ids = model._tokenize_texts([model._build_assistant_text(text)])
    ref_ids = None
    voice_clone_prompt_dict = None
    if prompt is not None:
        prompt_items = prompt if isinstance(prompt, list) else [prompt]
        voice_clone_prompt_dict = model._prompt_items_to_voice_clone_prompt(prompt_items)
        ref_ids = [
            model._tokenize_texts([model._build_ref_text(it.ref_text)])[0]
            if getattr(it, "ref_text", None)
            else None
            for it in prompt_items
        ]

    codes_list, _ = model.model.generate(
        input_ids=input_ids,
        ref_ids=ref_ids,
        voice_clone_prompt=voice_clone_prompt_dict,
        languages=[language],
        non_streaming_mode=non_streaming_mode,
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )
    t = codes_list[0].detach().cpu().to(torch.long)
    if t.ndim == 1:
        return [[int(x)] for x in t.tolist()]
    return t.tolist()


def _flat(x: list[list[int]]) -> list[int]:
    out = []
    for r in x:
        out.extend(r)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate train-set codec match rate (deterministic).")
    p.add_argument("--student-model", required=True)
    p.add_argument("--teacher-codes-jsonl", required=True)
    p.add_argument("--ref-audio", default=None)
    p.add_argument("--ref-text-file", default=None)
    p.add_argument(
        "--processor-model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="Processor source used in fallback loading.",
    )
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--non-streaming-mode", action="store_true")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    attn_impl = "flash_attention_2" if str(device).startswith("cuda") else "sdpa"

    ref_text = None
    if args.ref_text_file:
        ref_text = Path(args.ref_text_file).read_text(encoding="utf-8").strip()

    model = _load_qwen3_model(
        args.student_model,
        device=device,
        torch_dtype=dtype,
        attn_impl=attn_impl,
        processor_model=args.processor_model,
    )

    rows = _load_jsonl(Path(args.teacher_codes_jsonl))
    cfg_lang = getattr(getattr(model.model.config, "talker_config", None), "codec_language_id", None)
    supported = set((cfg_lang or {}).keys())

    total_tok = 0
    total_match = 0
    for i, r in enumerate(rows):
        sid = str(r.get("id", f"{i:06d}"))
        text = str(r["text"])
        language = _normalize_language(r.get("language"), supported)

        gt_2d = r.get("codec_ids_2d")
        if not gt_2d:
            gt_flat = r.get("codec_ids_flat", r.get("codec_ids", []))
            n_groups = int(r.get("num_code_groups", 16))
            n = (len(gt_flat) // n_groups) * n_groups
            gt_2d = [gt_flat[j : j + n_groups] for j in range(0, n, n_groups)]

        pred_2d = _gen_codec_ids_2d(
            model=model,
            text=text,
            language=language,
            ref_audio=args.ref_audio,
            ref_text=ref_text,
            non_streaming_mode=args.non_streaming_mode,
            max_new_tokens=args.max_new_tokens,
        )

        gt = _flat(gt_2d)
        pred = _flat(pred_2d)
        n = min(len(gt), len(pred))
        if n == 0:
            print(f"[WARN] {sid}: empty prediction or ground truth")
            continue
        m = sum(1 for a, b in zip(gt[:n], pred[:n]) if int(a) == int(b))
        total_tok += n
        total_match += m
        print(f"[ROW] {sid}: token_match={m/n:.4f} compared={n} gt_len={len(gt)} pred_len={len(pred)}")

    if total_tok == 0:
        print("[DONE] no comparable tokens")
        return
    print(f"[DONE] global_token_match={total_match/total_tok:.4f} compared={total_tok}")


if __name__ == "__main__":
    main()
