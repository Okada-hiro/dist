import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForCausalLM

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel


def _get_codec_eos_id(teacher_tts: Qwen3TTSModel) -> int | None:
    try:
        return int(teacher_tts.model.config.talker_config.codec_eos_token_id)
    except Exception:
        return None


def _trim_codec_ids(codec_ids: torch.Tensor, codec_eos_id: int | None) -> torch.Tensor:
    if codec_ids.numel() == 0:
        return codec_ids
    if codec_eos_id is None:
        return codec_ids
    eos_pos = (codec_ids == codec_eos_id).nonzero(as_tuple=False)
    if eos_pos.numel() == 0:
        return codec_ids
    return codec_ids[: int(eos_pos[0].item())]


@torch.inference_mode()
def _generate_codec_ids(
    student_model,
    teacher_tts: Qwen3TTSModel,
    text: str,
    language: str,
    bridge_token_id: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
) -> torch.Tensor:
    # Keep prompt formatting/tokenization identical to teacher pipeline.
    prompt_text = teacher_tts._build_assistant_text(text)
    text_ids = teacher_tts._tokenize_texts([prompt_text])[0].to(student_model.device)

    input_ids = torch.cat(
        [
            text_ids,
            torch.tensor([bridge_token_id], dtype=torch.long, device=student_model.device),
        ],
        dim=0,
    ).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    outputs = student_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        top_k=top_k if do_sample else None,
        pad_token_id=bridge_token_id,
    )

    # Generated part after prompt.
    gen = outputs[0, input_ids.shape[1] :]
    codec_eos_id = _get_codec_eos_id(teacher_tts)
    gen = _trim_codec_ids(gen, codec_eos_id)
    return gen.detach().cpu().to(torch.long)


def _decode_with_teacher(teacher_tts: Qwen3TTSModel, codec_ids: torch.Tensor) -> tuple[np.ndarray, int]:
    if codec_ids.numel() == 0:
        return np.zeros((1,), dtype=np.float32), 24000
    wavs, sr = teacher_tts.model.speech_tokenizer.decode([{"audio_codes": codec_ids}])
    return wavs[0], sr


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-model", required=True, help="Teacher Qwen3-TTS model id/path")
    p.add_argument("--student-model", required=True, help="Student checkpoint path")
    p.add_argument("--text", default=None, help="Single text input")
    p.add_argument("--input-jsonl", default=None, help="Batch input jsonl with {id,text,language?}")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--language", default="ja")
    p.add_argument("--bridge-token-id", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=50)
    args = p.parse_args()

    if not args.text and not args.input_jsonl:
        raise ValueError("Specify either --text or --input-jsonl")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    teacher_tts = Qwen3TTSModel.from_pretrained(args.teacher_model, device_map=device)
    student = AutoModelForCausalLM.from_pretrained(args.student_model, trust_remote_code=True)
    student.to(device)
    student.eval()

    bridge_token_id = args.bridge_token_id
    if bridge_token_id is None:
        # Training script uses eos as default bridge token.
        bridge_token_id = int(teacher_tts.tokenizer.eos_token_id)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    if args.text:
        rows.append({"id": "single", "text": args.text, "language": args.language})
    else:
        with Path(args.input_jsonl).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    meta_rows = []
    for i, row in enumerate(rows):
        sid = str(row.get("id", i))
        text = row["text"]
        language = row.get("language", args.language)

        codec_ids = _generate_codec_ids(
            student_model=student,
            teacher_tts=teacher_tts,
            text=text,
            language=language,
            bridge_token_id=bridge_token_id,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )

        wav, sr = _decode_with_teacher(teacher_tts, codec_ids)

        wav_path = out_dir / f"{sid}.wav"
        codec_path = out_dir / f"{sid}.codec_ids.json"
        sf.write(wav_path, wav, sr)
        codec_path.write_text(json.dumps(codec_ids.tolist(), ensure_ascii=False), encoding="utf-8")

        meta_rows.append(
            {
                "id": sid,
                "text": text,
                "language": language,
                "wav_path": str(wav_path),
                "codec_path": str(codec_path),
                "codec_len": int(codec_ids.numel()),
                "sample_rate": int(sr),
            }
        )

    (out_dir / "meta.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in meta_rows) + "\n",
        encoding="utf-8",
    )

    print(f"[OK] wrote {len(meta_rows)} files to {out_dir}")


if __name__ == "__main__":
    main()
