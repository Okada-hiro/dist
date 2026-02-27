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


def _to_int_list(x: Any) -> list[int]:
    """
    Normalize various json/tensor-like forms to flat list[int].
    Handles:
      - list[int]
      - list[list[int]] (flatten)
      - torch.Tensor
      - scalar int/float
    """
    if x is None:
        return []
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().tolist()
    if isinstance(x, (int, float)):
        return [int(x)]
    if isinstance(x, list):
        out: list[int] = []
        for v in x:
            if isinstance(v, list):
                out.extend(_to_int_list(v))
            elif isinstance(v, torch.Tensor):
                out.extend(_to_int_list(v))
            elif isinstance(v, (int, float)):
                out.append(int(v))
            else:
                # unknown nested type -> try best effort cast
                try:
                    out.append(int(v))
                except Exception:
                    continue
        return out
    try:
        return [int(x)]
    except Exception:
        return []


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

                # Prefer explicit flattened codec sequence if present.
                if "codec_ids_flat" in row:
                    row["codec_ids"] = row["codec_ids_flat"]
                elif "codec_ids_2d" in row and "codec_ids" not in row:
                    row["codec_ids"] = row["codec_ids_2d"]

                if "text_input_ids" not in row or "codec_ids" not in row:
                    self.skipped += 1
                    continue

                row["text_input_ids"] = _to_int_list(row["text_input_ids"])
                row["codec_ids"] = _to_int_list(row["codec_ids"])
                if len(row["text_input_ids"]) == 0 or len(row["codec_ids"]) == 0:
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
    pad_token_id: int
    max_codec_len: int | None = None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        seqs = []
        teacher_topk = []

        for row in features:
            codec_ids = _to_int_list(row.get("codec_ids", row.get("codes")))
            if codec_ids is None:
                continue
            if self.max_codec_len is not None and self.max_codec_len > 0:
                codec_ids = codec_ids[: self.max_codec_len]
            if len(codec_ids) < 2:
                continue

            seqs.append(torch.tensor(codec_ids, dtype=torch.long))
            teacher_topk.append(row.get("teacher_topk"))

        if not seqs:
            raise ValueError("Batch has no valid samples. Check keys: codec_ids in train jsonl.")

        input_ids = torch.nn.utils.rnn.pad_sequence(
            seqs, batch_first=True, padding_value=self.pad_token_id
        )
        labels = input_ids.clone()
        labels[input_ids == self.pad_token_id] = -100
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
        # Qwen3TTSForConditionalGeneration itself does not expose training forward(input_ids).
        # Train the talker directly with embeddings in prefill mode.
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

        embeds = model.talker.get_input_embeddings()(input_ids)
        outputs = model.talker(
            inputs_embeds=embeds[:, :-1, :],
            attention_mask=attention_mask[:, :-1],
            labels=labels[:, 1:],
        )
        ce_loss = outputs.loss

        # Optional sparse KL if dataset has teacher_topk.
        kl_loss = torch.tensor(0.0, device=ce_loss.device)
        if self.kl_alpha > 0 and teacher_topk is not None and any(x is not None for x in teacher_topk):
            logits = outputs.logits
            labels = labels[:, 1:]

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
    p.add_argument("--pad-token-id", type=int, default=None)
    p.add_argument("--kl-alpha", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--dry-run-only", action="store_true", help="Run one forward/backward sanity check and exit.")
    p.add_argument("--max-codec-len", type=int, default=None, help="Truncate codec_ids length per sample to reduce VRAM.")
    p.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing to reduce VRAM.")
    args = p.parse_args()

    # Student config is a local json produced by 00_make_student_config.py.
    cfg_dict = json.loads(Path(args.student_config).read_text(encoding="utf-8"))
    student_cfg = Qwen3TTSConfig(**cfg_dict)
    model = Qwen3TTSForConditionalGeneration(student_cfg)
    if args.gradient_checkpointing:
        enabled = False
        try:
            model.talker.gradient_checkpointing_enable()
            enabled = True
        except Exception:
            pass
        try:
            model.talker.model.gradient_checkpointing = True
            enabled = True
        except Exception:
            pass
        try:
            model.talker.code_predictor.gradient_checkpointing_enable()
            enabled = True
        except Exception:
            pass
        print(f"[INFO] gradient_checkpointing={'enabled' if enabled else 'requested_but_not_applied'}")

    pad_token_id = args.pad_token_id if args.pad_token_id is not None else 2150

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
    collator = DistillCollator(pad_token_id=pad_token_id, max_codec_len=args.max_codec_len)
    if args.max_codec_len is not None:
        print(f"[INFO] max_codec_len={args.max_codec_len}")

    # Sanity: model init + single forward/backward before full train.
    sample_batch = collator([train_ds[0]])
    sample_batch = {k: (v.to(model.device) if isinstance(v, torch.Tensor) else v) for k, v in sample_batch.items()}
    model.train()
    embeds = model.talker.get_input_embeddings()(sample_batch["input_ids"])
    sanity_out = model.talker(
        inputs_embeds=embeds[:, :-1, :],
        attention_mask=sample_batch["attention_mask"][:, :-1],
        labels=sample_batch["labels"][:, 1:],
    )
    sanity_loss = sanity_out.loss
    sanity_loss.backward()
    model.zero_grad(set_to_none=True)
    print(f"[OK] dry-run passed: talker forward/backward loss={float(sanity_loss.detach().cpu()):.4f}")
    if args.dry_run_only:
        print("[DONE] dry-run-only mode, exiting without Trainer loop.")
        return

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
