import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import soundfile as sf
import torch

from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
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
    x_vector_only_mode: bool,
    max_new_tokens: int,
    do_sample: bool,
    top_k: int,
    top_p: float,
    temperature: float,
) -> tuple[list[Any], int]:
    wavs, sr = model.generate_voice_clone(
        text=text,
        language=language,
        ref_audio=ref_audio,
        ref_text=ref_text,
        x_vector_only_mode=x_vector_only_mode,
        non_streaming_mode=True,
        do_sample=do_sample,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    return wavs, sr


def _sanitize_config_for_qwen3(cfg: dict[str, Any]) -> dict[str, Any]:
    clean = json.loads(json.dumps(cfg))
    root_model_type = clean.get("model_type", None)
    drop_keys = {"dtype", "torch_dtype"}

    def _walk(x: Any, depth: int = 0) -> Any:
        if isinstance(x, dict):
            out = {}
            for k, v in x.items():
                if k in drop_keys:
                    continue
                if k == "model_type" and depth > 0:
                    continue
                out[k] = _walk(v, depth + 1)
            return out
        if isinstance(x, list):
            return [_walk(v, depth + 1) for v in x]
        return x

    sanitized = _walk(clean, 0)
    if root_model_type and "model_type" not in sanitized:
        sanitized["model_type"] = root_model_type
    return sanitized


def _load_qwen3_model(
    model_path_or_id: str,
    device: str,
    torch_dtype: torch.dtype,
    attn_impl: str,
    fallback_processor=None,
    fallback_generate_defaults: dict[str, Any] | None = None,
):
    try:
        return Qwen3TTSModel.from_pretrained(
            model_path_or_id,
            device_map=device,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
        )
    except Exception as e:
        # Common with local student checkpoints:
        #  - nested config.model_type issue
        #  - AutoProcessor tokenizer file resolution issue
        model_dir = Path(model_path_or_id)
        if (not model_dir.exists()) or (fallback_processor is None):
            raise
        print(f"[WARN] wrapper load failed for {model_path_or_id}: {e}")
        print("[INFO] fallback: load core model directly and reuse teacher processor.")
        try:
            cfg_path = model_dir / "config.json"
            if not cfg_path.exists():
                raise
            raw_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            clean_cfg = _sanitize_config_for_qwen3(raw_cfg)
            # Qwen3TTSForConditionalGeneration.from_pretrained() still reads config.json internally.
            # Create a sanitized temp model dir and point loading there.
            with tempfile.TemporaryDirectory(prefix="qwen3tts_sanitized_") as td:
                td_path = Path(td)
                for item in model_dir.iterdir():
                    dst = td_path / item.name
                    if item.name == "config.json":
                        continue
                    try:
                        os.symlink(item, dst, target_is_directory=item.is_dir())
                    except OSError:
                        # Fallback when symlink is unavailable.
                        if item.is_dir():
                            import shutil
                            shutil.copytree(item, dst)
                        else:
                            import shutil
                            shutil.copy2(item, dst)
                (td_path / "config.json").write_text(
                    json.dumps(clean_cfg, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                cfg_obj = Qwen3TTSConfig(**clean_cfg)
                core_model = Qwen3TTSForConditionalGeneration.from_pretrained(
                    str(td_path),
                    config=cfg_obj,
                    device_map=device,
                    torch_dtype=torch_dtype,
                    attn_implementation=attn_impl,
                )
            return Qwen3TTSModel(
                model=core_model,
                processor=fallback_processor,
                generate_defaults=fallback_generate_defaults or getattr(core_model, "generate_config", {}) or {},
            )
        except Exception:
            raise


def main() -> None:
    p = argparse.ArgumentParser(description="Generate teacher/student A-B wav pairs for eval.")
    p.add_argument("--teacher-model", required=True, help="Teacher model repo/local dir")
    p.add_argument("--student-model", required=True, help="Student model local dir")
    p.add_argument("--input-jsonl", required=True, help="Input texts jsonl (id,text,language optional)")
    p.add_argument("--ref-audio", required=True)
    p.add_argument("--ref-text-file", required=True)
    p.add_argument(
        "--x-vector-only",
        action="store_true",
        help="Use speaker embedding only (disable ICL ref_code path).",
    )
    p.add_argument("--out-dir", default="dist/eval_ab")
    p.add_argument("--default-language", default="Japanese")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--do-sample", action="store_true", help="Enable stochastic decoding (default: deterministic).")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--device", default=None, help="cuda:0 / cpu. default: auto")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_text = Path(args.ref_text_file).read_text(encoding="utf-8").strip()
    rows = _load_jsonl(Path(args.input_jsonl))

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32

    print(f"[INFO] loading teacher: {args.teacher_model}")
    attn_impl = "flash_attention_2" if str(device).startswith("cuda") else "sdpa"
    teacher = _load_qwen3_model(args.teacher_model, device=device, torch_dtype=torch_dtype, attn_impl=attn_impl)
    print(f"[INFO] loading student: {args.student_model}")
    student = _load_qwen3_model(
        args.student_model,
        device=device,
        torch_dtype=torch_dtype,
        attn_impl=attn_impl,
        fallback_processor=teacher.processor,
        fallback_generate_defaults=teacher.generate_defaults,
    )

    meta_rows: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        sid = str(r.get("id", f"{i:04d}"))
        text = str(r["text"])
        language = _norm_language(r.get("language"), args.default_language)

        t0 = time.perf_counter()
        twavs, tsr = _gen_one(
            teacher,
            text,
            language,
            args.ref_audio,
            ref_text,
            args.x_vector_only,
            args.max_new_tokens,
            args.do_sample,
            args.top_k,
            args.top_p,
            args.temperature,
        )
        teacher_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        swavs, ssr = _gen_one(
            student,
            text,
            language,
            args.ref_audio,
            ref_text,
            args.x_vector_only,
            args.max_new_tokens,
            args.do_sample,
            args.top_k,
            args.top_p,
            args.temperature,
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
