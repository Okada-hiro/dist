import argparse
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run official finetuning pipeline: prepare_data.py -> sft_12hz.py"
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tokenizer-model-path", default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    p.add_argument("--init-model-path", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    p.add_argument("--train-raw-jsonl", default="dist/train_raw.jsonl")
    p.add_argument("--train-with-codes-jsonl", default="dist/train_with_codes.jsonl")
    p.add_argument("--output-model-path", default="dist/sft_output")
    p.add_argument("--speaker-name", default="speaker_dist")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument(
        "--finetuning-dir",
        default=None,
        help="Path to finetuning directory (contains prepare_data.py and sft_12hz.py). "
             "If omitted, auto-detect from current cwd/script location.",
    )
    args = p.parse_args()

    py = sys.executable
    script_dir = Path(__file__).resolve().parent

    if args.finetuning_dir:
        finetuning_dir = Path(args.finetuning_dir).resolve()
    else:
        candidates = [
            Path(".").resolve() / "finetuning",
            script_dir.parent / "finetuning",
            script_dir / "finetuning",
        ]
        finetuning_dir = None
        for c in candidates:
            if (c / "prepare_data.py").exists() and (c / "sft_12hz.py").exists():
                finetuning_dir = c
                break
        if finetuning_dir is None:
            raise FileNotFoundError(
                "Could not find finetuning directory. "
                "Pass --finetuning-dir explicitly."
            )

    # sft_12hz.py uses shutil.copytree(MODEL_PATH, ...), so MODEL_PATH must be a local directory.
    init_model_path = args.init_model_path
    init_model_dir = Path(init_model_path)
    if not init_model_dir.exists():
        print(f"[INFO] resolving HF model to local snapshot: {init_model_path}")
        init_model_path = snapshot_download(repo_id=init_model_path)
        print(f"[INFO] local init model path: {init_model_path}")

    run(
        [
            py,
            str(finetuning_dir / "prepare_data.py"),
            "--device",
            args.device,
            "--tokenizer_model_path",
            args.tokenizer_model_path,
            "--input_jsonl",
            args.train_raw_jsonl,
            "--output_jsonl",
            args.train_with_codes_jsonl,
        ]
    )

    run(
        [
            py,
            str(finetuning_dir / "sft_12hz.py"),
            "--init_model_path",
            init_model_path,
            "--output_model_path",
            args.output_model_path,
            "--train_jsonl",
            args.train_with_codes_jsonl,
            "--batch_size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--num_epochs",
            str(args.num_epochs),
            "--speaker_name",
            args.speaker_name,
        ]
    )

    print("[OK] official sft pipeline completed")
    print(f"  finetuning_dir   : {finetuning_dir}")
    print(f"  raw_jsonl        : {args.train_raw_jsonl}")
    print(f"  train_with_codes : {args.train_with_codes_jsonl}")
    print(f"  output_model     : {args.output_model_path}")


if __name__ == "__main__":
    main()
