import argparse
import json
import time
from pathlib import Path
from typing import Any

import soundfile as sf
import torch

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


def _norm_language(raw: str | None, fallback: str) -> str:
    if raw is None:
        return fallback
    s = str(raw).strip()
    if not s:
        return fallback
    alias = {
        "ja": "Japanese",
        "japanese": "Japanese",
        "en": "English",
        "english": "English",
        "zh": "Chinese",
        "chinese": "Chinese",
    }
    return alias.get(s.lower(), s)


def _gen_one(
    model: Qwen3TTSModel,
    text: str,
    language: str,
    ref_audio: str,
    ref_text: str,
    max_new_tokens: int,
) -> tuple[list[Any], int]:
    wavs, sr = model.generate_voice_clone(
        text=text,
        language=language,
        ref_audio=ref_audio,
        ref_text=ref_text,
        x_vector_only_mode=False,
        non_streaming_mode=True,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        max_new_tokens=max_new_tokens,
    )
    return wavs, sr


def main() -> None:
    p = argparse.ArgumentParser(description="Generate teacher/student A-B wav pairs for eval.")
    p.add_argument("--teacher-model", required=True, help="Teacher model repo/local dir")
    p.add_argument("--student-model", required=True, help="Student model local dir")
    p.add_argument("--input-jsonl", required=True, help="Input texts jsonl (id,text,language optional)")
    p.add_argument("--ref-audio", required=True)
    p.add_argument("--ref-text-file", required=True)
    p.add_argument("--out-dir", default="dist/eval_ab")
    p.add_argument("--default-language", default="Japanese")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--device", default=None, help="cuda:0 / cpu. default: auto")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_text = Path(args.ref_text_file).read_text(encoding="utf-8").strip()
    rows = _load_jsonl(Path(args.input_jsonl))

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32

    print(f"[INFO] loading teacher: {args.teacher_model}")
    teacher = Qwen3TTSModel.from_pretrained(
        args.teacher_model,
        device_map=device,
        dtype=dtype,
        attn_implementation="flash_attention_2" if str(device).startswith("cuda") else "sdpa",
    )
    print(f"[INFO] loading student: {args.student_model}")
    student = Qwen3TTSModel.from_pretrained(
        args.student_model,
        device_map=device,
        dtype=dtype,
        attn_implementation="flash_attention_2" if str(device).startswith("cuda") else "sdpa",
    )

    meta_rows: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        sid = str(r.get("id", f"{i:04d}"))
        text = str(r["text"])
        language = _norm_language(r.get("language"), args.default_language)

        t0 = time.perf_counter()
        twavs, tsr = _gen_one(
            teacher, text, language, args.ref_audio, ref_text, args.max_new_tokens
        )
        teacher_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        swavs, ssr = _gen_one(
            student, text, language, args.ref_audio, ref_text, args.max_new_tokens
        )
        student_ms = (time.perf_counter() - t1) * 1000.0

        teacher_wav = out_dir / f"{sid}.teacher.wav"
        student_wav = out_dir / f"{sid}.student.wav"
        sf.write(str(teacher_wav), twavs[0], tsr)
        sf.write(str(student_wav), swavs[0], ssr)

        meta_rows.append(
            {
                "id": sid,
                "text": text,
                "language": language,
                "teacher_wav": str(teacher_wav),
                "student_wav": str(student_wav),
                "teacher_sr": int(tsr),
                "student_sr": int(ssr),
                "teacher_ms": round(teacher_ms, 1),
                "student_ms": round(student_ms, 1),
            }
        )
        print(
            f"[OK] {sid}: teacher={teacher_ms:.1f}ms student={student_ms:.1f}ms "
            f"-> {teacher_wav.name}, {student_wav.name}"
        )

    _write_jsonl(out_dir / "ab_metadata.jsonl", meta_rows)
    print(f"[DONE] wrote {len(meta_rows)} pairs -> {out_dir}")


if __name__ == "__main__":
    main()
