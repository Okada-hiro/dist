import argparse
import subprocess
import sys
from pathlib import Path


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
    args = p.parse_args()

    py = sys.executable
    repo_root = Path(".").resolve()

    run(
        [
            py,
            str(repo_root / "finetuning" / "prepare_data.py"),
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
            str(repo_root / "finetuning" / "sft_12hz.py"),
            "--init_model_path",
            args.init_model_path,
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
    print(f"  raw_jsonl        : {args.train_raw_jsonl}")
    print(f"  train_with_codes : {args.train_with_codes_jsonl}")
    print(f"  output_model     : {args.output_model_path}")


if __name__ == "__main__":
    main()

