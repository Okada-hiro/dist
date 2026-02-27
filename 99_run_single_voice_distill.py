import argparse
import subprocess
import sys
from pathlib import Path

from transformers import AutoConfig


def run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Single-voice distillation pipeline runner (config -> teacher codes -> student train)"
    )
    p.add_argument("--teacher-model", required=True, help="Teacher model id/path")
    p.add_argument(
        "--teacher-config",
        default=None,
        help="Teacher config.json path (optional; auto-resolved from --teacher-model if omitted)",
    )
    p.add_argument("--input-jsonl", required=True, help="Training text jsonl")
    p.add_argument(
        "--work-dir",
        default="知識蒸留",
        help="Working dir for outputs (default: 知識蒸留)",
    )

    # Student shape ratios
    p.add_argument("--layer-ratio", type=float, default=0.5)
    p.add_argument("--hidden-ratio", type=float, default=0.8)
    p.add_argument("--ffn-ratio", type=float, default=0.8)

    # Teacher code generation options
    p.add_argument("--ref-audio", default="知識蒸留/ref_audio.WAV")
    p.add_argument("--ref-text-file", default="知識蒸留/ref_text.txt")
    p.add_argument("--x-vector-only", action="store_true")
    p.add_argument("--non-streaming-mode", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument(
        "--save-teacher-artifacts",
        action="store_true",
        help="Save teacher wav/codec artifacts under <work-dir>/teacher_artifacts",
    )

    # Training options
    p.add_argument("--kl-alpha", type=float, default=0.0)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=-1)

    args = p.parse_args()

    root = Path(args.work_dir)
    root.mkdir(parents=True, exist_ok=True)

    teacher_config = args.teacher_config
    if teacher_config is None:
        tm = Path(args.teacher_model)
        if tm.exists() and tm.is_dir() and (tm / "config.json").exists():
            teacher_config = str(tm / "config.json")
        else:
            # Resolve config from HF model id (or local model path) and materialize it for step-00.
            cfg = AutoConfig.from_pretrained(args.teacher_model, trust_remote_code=True)
            resolved_cfg_path = root / "teacher_config.resolved.json"
            resolved_cfg_path.write_text(cfg.to_json_string(use_diff=False), encoding="utf-8")
            teacher_config = str(resolved_cfg_path)

    ref_text_file = Path(args.ref_text_file)
    ref_text = ref_text_file.read_text(encoding="utf-8").strip() if ref_text_file.exists() else None

    py = sys.executable

    student_config = root / "student_config.json"
    teacher_codes = root / "train_teacher_codes.jsonl"
    student_ckpt = root / "student_ckpt"

    run(
        [
            py,
            "知識蒸留/00_make_student_config.py",
            "--teacher-config",
            teacher_config,
            "--output",
            str(student_config),
            "--layer-ratio",
            str(args.layer_ratio),
            "--hidden-ratio",
            str(args.hidden_ratio),
            "--ffn-ratio",
            str(args.ffn_ratio),
        ]
    )

    cmd_build = [
        py,
        "知識蒸留/10_build_teacher_codes.py",
        "--teacher-model",
        args.teacher_model,
        "--input-jsonl",
        args.input_jsonl,
        "--output-jsonl",
        str(teacher_codes),
        "--ref-audio",
        args.ref_audio,
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if ref_text:
        cmd_build += ["--ref-text", ref_text]
    if args.x_vector_only:
        cmd_build.append("--x-vector-only")
    if args.non_streaming_mode:
        cmd_build.append("--non-streaming-mode")
    if args.save_teacher_artifacts:
        cmd_build += ["--save-artifacts-dir", str(root / "teacher_artifacts")]
    run(cmd_build)

    run(
        [
            py,
            "知識蒸留/20_train_student_ce_kl.py",
            "--teacher-model",
            args.teacher_model,
            "--student-config",
            str(student_config),
            "--train-jsonl",
            str(teacher_codes),
            "--output-dir",
            str(student_ckpt),
            "--kl-alpha",
            str(args.kl_alpha),
            "--epochs",
            str(args.epochs),
            "--lr",
            str(args.lr),
            "--batch-size",
            str(args.batch_size),
            "--grad-accum",
            str(args.grad_accum),
            "--max-steps",
            str(args.max_steps),
        ]
    )

    print("[OK] Distillation pipeline completed")
    print(f"  student_config: {student_config}")
    print(f"  teacher_codes : {teacher_codes}")
    print(f"  student_ckpt  : {student_ckpt}")


if __name__ == "__main__":
    main()
