import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration


class DistillJsonlDataset(Dataset):
    def __init__(self, path: str):
        self.rows = []
        self.skipped = 0
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)

                # Accept a few legacy/alternative key names.
                if "text_input_ids" not in row:
                    if "input_ids" in row and isinstance(row["input_ids"], list):
                        row["text_input_ids"] = row["input_ids"]
                    elif "text_ids" in row and isinstance(row["text_ids"], list):
                        row["text_input_ids"] = row["text_ids"]

                if "codec_ids" not in row:
                    if "codes" in row and isinstance(row["codes"], list):
                        row["codec_ids"] = row["codes"]

                if "text_input_ids" not in row or "codec_ids" not in row:
                    self.skipped += 1
                    continue

                self.rows.append(row)

        if len(self.rows) == 0:
            raise ValueError(
                f"No valid rows in {path}. Expected keys include text_input_ids and codec_ids."
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


@dataclass
class DistillCollator:
    bridge_token_id: int
    pad_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        seqs = []
        labels = []
        teacher_topk = []

        for row in features:
            text_ids = row.get("text_input_ids", row.get("input_ids", row.get("text_ids")))
            codec_ids = row.get("codec_ids", row.get("codes"))
            if text_ids is None or codec_ids is None:
                continue
            if len(text_ids) == 0 or len(codec_ids) == 0:
                continue

            input_ids = text_ids + [self.bridge_token_id] + codec_ids
            lab = ([-100] * (len(text_ids) + 1)) + codec_ids

            seqs.append(torch.tensor(input_ids, dtype=torch.long))
            labels.append(torch.tensor(lab, dtype=torch.long))
            teacher_topk.append(row.get("teacher_topk"))

        if not seqs:
            raise ValueError("Batch has no valid samples. Check keys: text_input_ids/codec_ids in train jsonl.")

        input_ids = torch.nn.utils.rnn.pad_sequence(
            seqs, batch_first=True, padding_value=self.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        attention_mask = (input_ids != self.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "teacher_topk": teacher_topk,
        }


class DistillTrainer(Trainer):
    def __init__(self, *args, kl_alpha: float = 0.0, temperature: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_alpha = kl_alpha
        self.temperature = temperature

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        teacher_topk = inputs.pop("teacher_topk", None)
        outputs = model(**inputs)
        ce_loss = outputs.loss

        # Optional sparse KL if dataset has teacher_topk.
        kl_loss = torch.tensor(0.0, device=ce_loss.device)
        if self.kl_alpha > 0 and teacher_topk is not None and any(x is not None for x in teacher_topk):
            logits = outputs.logits
            labels = inputs["labels"]

            # Shift for causal prediction: logits[t] predicts label[t].
            log_probs = F.log_softmax(logits / self.temperature, dim=-1)
            per_sample = []

            for b in range(labels.size(0)):
                target_pos = torch.where(labels[b] != -100)[0].tolist()
                topk_items = teacher_topk[b]
                if topk_items is None:
                    continue

                steps = min(len(target_pos), len(topk_items))
                if steps == 0:
                    continue

                sample_loss = torch.tensor(0.0, device=ce_loss.device)
                valid_steps = 0
                for i in range(steps):
                    item = topk_items[i]
                    if not item:
                        continue
                    idx = torch.tensor(item["indices"], dtype=torch.long, device=ce_loss.device)
                    t_logp = torch.tensor(item["log_probs"], dtype=torch.float, device=ce_loss.device)
                    t_prob = torch.softmax(t_logp / self.temperature, dim=0)

                    pos = target_pos[i]
                    s_logp = log_probs[b, pos, idx]
                    sample_loss = sample_loss + torch.sum(t_prob * (torch.log(t_prob + 1e-8) - s_logp))
                    valid_steps += 1

                if valid_steps > 0:
                    per_sample.append(sample_loss / valid_steps)

            if per_sample:
                kl_loss = torch.stack(per_sample).mean() * (self.temperature ** 2)

        loss = ce_loss + self.kl_alpha * kl_loss
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-model", required=True, help="Tokenizer source model id/path")
    p.add_argument("--student-config", required=True, help="Path to student config.json")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--eval-jsonl", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--bridge-token-id", type=int, default=None)
    p.add_argument("--pad-token-id", type=int, default=None)
    p.add_argument("--kl-alpha", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=-1)
    args = p.parse_args()

    # Student config is a local json produced by 00_make_student_config.py.
    cfg_dict = json.loads(Path(args.student_config).read_text(encoding="utf-8"))
    student_cfg = Qwen3TTSConfig(**cfg_dict)
    model = Qwen3TTSForConditionalGeneration(student_cfg)

    # We use pre-tokenized ids from teacher data.
    # Keep defaults compatible with observed Qwen3-TTS tokenizer ids.
    pad_token_id = args.pad_token_id if args.pad_token_id is not None else 2150
    bridge_token_id = args.bridge_token_id if args.bridge_token_id is not None else 2150

    train_ds = DistillJsonlDataset(args.train_jsonl)
    eval_ds = DistillJsonlDataset(args.eval_jsonl) if args.eval_jsonl else None
    print(
        f"[INFO] train rows: {len(train_ds)}"
        + (f" (skipped={train_ds.skipped})" if getattr(train_ds, "skipped", 0) else "")
    )
    if len(train_ds) > 0:
        print(f"[INFO] sample train keys: {sorted(train_ds[0].keys())}")
    if eval_ds is not None:
        print(
            f"[INFO] eval rows: {len(eval_ds)}"
            + (f" (skipped={eval_ds.skipped})" if getattr(eval_ds, "skipped", 0) else "")
        )
    collator = DistillCollator(bridge_token_id=bridge_token_id, pad_token_id=pad_token_id)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=20,
        save_steps=200,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=200 if eval_ds is not None else None,
        max_steps=args.max_steps,
        bf16=torch.cuda.is_available(),
        fp16=False,
        remove_unused_columns=False,
        report_to=[],
    )

    trainer = DistillTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        kl_alpha=args.kl_alpha,
        temperature=args.temperature,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
