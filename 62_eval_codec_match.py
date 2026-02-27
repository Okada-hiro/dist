import argparse
import json
from pathlib import Path
from typing import Any

import torch

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

    model = Qwen3TTSModel.from_pretrained(
        args.student_model,
        device_map=device,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
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
