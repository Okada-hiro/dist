import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


class DistillJsonlDataset(Dataset):
    def __init__(self, path: str):
        self.rows = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))

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
            text_ids = row["text_input_ids"]
            codec_ids = row["codec_ids"]

            input_ids = text_ids + [self.bridge_token_id] + codec_ids
            lab = ([-100] * (len(text_ids) + 1)) + codec_ids

            seqs.append(torch.tensor(input_ids, dtype=torch.long))
            labels.append(torch.tensor(lab, dtype=torch.long))
            teacher_topk.append(row.get("teacher_topk"))

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

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    pad_token_id = args.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    bridge_token_id = args.bridge_token_id
    if bridge_token_id is None:
        bridge_token_id = tokenizer.eos_token_id

    student_cfg = AutoConfig.from_pretrained(args.student_config, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(student_cfg, trust_remote_code=True)

    train_ds = DistillJsonlDataset(args.train_jsonl)
    eval_ds = DistillJsonlDataset(args.eval_jsonl) if args.eval_jsonl else None
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
