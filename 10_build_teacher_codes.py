import argparse
import json
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from tqdm import tqdm

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-model", required=True, help="HF model id/path for Qwen3-TTS teacher")
    p.add_argument("--input-jsonl", required=True, help="Input jsonl with at least {text, language}")
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--ref-audio", default=None)
    p.add_argument("--ref-text", default=None)
    p.add_argument("--x-vector-only", action="store_true")
    p.add_argument("--non-streaming-mode", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument(
        "--save-artifacts-dir",
        default=None,
        help="Optional dir to save per-sample teacher artifacts (*.wav, *.codec_ids.json, metadata.jsonl)",
    )
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher = Qwen3TTSModel.from_pretrained(args.teacher_model, device_map=device)

    prompt = None
    if args.ref_audio is not None:
        prompt = teacher.create_voice_clone_prompt(
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            x_vector_only_mode=args.x_vector_only,
        )

    rows = _load_jsonl(Path(args.input_jsonl))
    out_rows: list[dict[str, Any]] = []
    artifacts_dir = Path(args.save_artifacts_dir) if args.save_artifacts_dir else None
    artifact_meta_rows: list[dict[str, Any]] = []
    if artifacts_dir is not None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in enumerate(tqdm(rows, desc="build_teacher_codes")):
        text = row["text"]
        language = row.get("language", "ja")
        sid = str(row.get("id", f"{idx:06d}"))

        # Keep this aligned with inference implementation.
        input_ids = teacher._tokenize_texts([teacher._build_assistant_text(text)])
        ref_ids = None
        voice_clone_prompt_dict = None

        if prompt is not None:
            prompt_items = prompt if isinstance(prompt, list) else [prompt]
            voice_clone_prompt_dict = teacher._prompt_items_to_voice_clone_prompt(prompt_items)
            ref_ids = [
                teacher._tokenize_texts([teacher._build_ref_text(it.ref_text)])[0]
                if getattr(it, "ref_text", None)
                else None
                for it in prompt_items
            ]

        talker_codes_list, _ = teacher.model.generate(
            input_ids=input_ids,
            ref_ids=ref_ids,
            voice_clone_prompt=voice_clone_prompt_dict,
            languages=[language],
            non_streaming_mode=args.non_streaming_mode,
            do_sample=True,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )

        codec_ids = talker_codes_list[0].detach().cpu().to(torch.long).tolist()
        text_input_ids = input_ids[0].detach().cpu().to(torch.long).tolist()

        out_rows.append(
            {
                "id": sid,
                "text": text,
                "language": language,
                "text_input_ids": text_input_ids,
                "codec_ids": codec_ids,
            }
        )

        if artifacts_dir is not None:
            codec_tensor = talker_codes_list[0].detach().cpu().to(torch.long)
            wavs, sr = teacher.model.speech_tokenizer.decode([{"audio_codes": codec_tensor}])
            wav = wavs[0]

            wav_path = artifacts_dir / f"{sid}.teacher.wav"
            codec_path = artifacts_dir / f"{sid}.codec_ids.json"
            sf.write(str(wav_path), wav, sr)
            codec_path.write_text(json.dumps(codec_ids, ensure_ascii=False), encoding="utf-8")

            artifact_meta_rows.append(
                {
                    "id": sid,
                    "text": text,
                    "language": language,
                    "wav_path": str(wav_path),
                    "codec_path": str(codec_path),
                    "codec_len": int(len(codec_ids)),
                    "sample_rate": int(sr),
                }
            )

    _write_jsonl(Path(args.output_jsonl), out_rows)
    if artifacts_dir is not None:
        meta_path = artifacts_dir / "metadata.jsonl"
        _write_jsonl(meta_path, artifact_meta_rows)
        print(f"[OK] artifacts saved -> {artifacts_dir}")
    print(f"[OK] wrote {len(out_rows)} samples -> {args.output_jsonl}")


if __name__ == "__main__":
    main()
