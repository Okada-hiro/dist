import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration, mel_spectrogram


def _to_int_list(x: Any) -> list[int]:
    if x is None:
        return []
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().tolist()
    if isinstance(x, (int, float)):
        return [int(x)]
    if isinstance(x, list):
        out: list[int] = []
        for v in x:
            out.extend(_to_int_list(v))
        return out
    try:
        return [int(x)]
    except Exception:
        return []


def _to_2d_codec(x: Any, num_code_groups: int | None = None) -> list[list[int]]:
    if x is None:
        return []
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().tolist()
    if not isinstance(x, list):
        return []
    if len(x) == 0:
        return []

    # Already 2D.
    if isinstance(x[0], list):
        out = []
        for row in x:
            r = _to_int_list(row)
            if r:
                out.append(r)
        return out

    # Flat -> reshape when group count available.
    flat = _to_int_list(x)
    if not flat:
        return []
    g = int(num_code_groups or 1)
    if g <= 1:
        return [[v] for v in flat]
    n = (len(flat) // g) * g
    flat = flat[:n]
    return [flat[i : i + g] for i in range(0, n, g)]


class DistillJsonlDataset(Dataset):
    def __init__(self, path: str, default_num_code_groups: int = 16):
        self.rows: list[dict[str, Any]] = []
        self.skipped = 0

        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)

                if "text_input_ids" not in row:
                    if "input_ids" in row:
                        row["text_input_ids"] = row["input_ids"]
                    elif "text_ids" in row:
                        row["text_input_ids"] = row["text_ids"]

                if "codec_ids_2d" in row:
                    codec_2d = _to_2d_codec(row["codec_ids_2d"])
                else:
                    n_groups = int(row.get("num_code_groups", default_num_code_groups))
                    src = row.get("codec_ids_flat", row.get("codec_ids"))
                    codec_2d = _to_2d_codec(src, n_groups)

                text_ids = _to_int_list(row.get("text_input_ids"))

                if len(text_ids) == 0 or len(codec_2d) == 0:
                    self.skipped += 1
                    continue

                row["text_input_ids"] = text_ids
                row["codec_ids_2d"] = codec_2d
                row["num_code_groups"] = len(codec_2d[0]) if codec_2d else default_num_code_groups
                self.rows.append(row)

        if not self.rows:
            raise ValueError(f"No valid rows in {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


@dataclass
class QwenLikeCollator:
    config: Qwen3TTSConfig
    max_codec_len: int | None = None
    fixed_spk_id: int = 0

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        b = len(batch)
        g = int(self.config.talker_config.num_code_groups)

        text_ids_list = [torch.tensor(x["text_input_ids"], dtype=torch.long) for x in batch]
        codec_2d_list = []
        for x in batch:
            c = x["codec_ids_2d"]
            if self.max_codec_len is not None and self.max_codec_len > 0:
                c = c[: self.max_codec_len]
            codec_2d_list.append(torch.tensor(c, dtype=torch.long))

        lens = [t.shape[0] + c.shape[0] for t, c in zip(text_ids_list, codec_2d_list)]
        max_len = max(lens) + 8

        input_ids = torch.zeros((b, max_len, 2), dtype=torch.long)
        codec_ids = torch.zeros((b, max_len, g), dtype=torch.long)
        text_embedding_mask = torch.zeros((b, max_len), dtype=torch.bool)
        codec_embedding_mask = torch.zeros((b, max_len), dtype=torch.bool)
        codec_mask = torch.zeros((b, max_len), dtype=torch.bool)
        attention_mask = torch.zeros((b, max_len), dtype=torch.long)
        codec_0_labels = torch.full((b, max_len), -100, dtype=torch.long)

        for i in range(b):
            text_ids = text_ids_list[i]
            codes = codec_2d_list[i]
            if codes.ndim != 2 or codes.shape[1] != g:
                raise ValueError(f"codec shape mismatch at i={i}: got {tuple(codes.shape)}, expected (*,{g})")

            audio_codec_0 = codes[:, 0]
            text_len = text_ids.shape[0]
            codec_len = audio_codec_0.shape[0]

            # text channel
            input_ids[i, :3, 0] = text_ids[:3]
            input_ids[i, 3:7, 0] = self.config.tts_pad_token_id
            input_ids[i, 7, 0] = self.config.tts_bos_token_id
            if text_len > 3:
                input_ids[i, 8 : 8 + text_len - 3, 0] = text_ids[3:]
            input_ids[i, 8 + text_len - 3, 0] = self.config.tts_eos_token_id
            input_ids[i, 8 + text_len - 2 : 8 + text_len + codec_len, 0] = self.config.tts_pad_token_id
            text_embedding_mask[i, : 8 + text_len + codec_len] = True

            # codec channel
            input_ids[i, 3:8, 1] = torch.tensor(
                [
                    self.config.talker_config.codec_nothink_id,
                    self.config.talker_config.codec_think_bos_id,
                    self.config.talker_config.codec_think_eos_id,
                    int(self.fixed_spk_id),  # fixed speaker embedding token id
                    self.config.talker_config.codec_pad_id,
                ],
                dtype=torch.long,
            )
            if text_len > 3:
                input_ids[i, 8 : 8 + text_len - 3, 1] = self.config.talker_config.codec_pad_id
            input_ids[i, 8 + text_len - 3, 1] = self.config.talker_config.codec_pad_id
            input_ids[i, 8 + text_len - 2, 1] = self.config.talker_config.codec_bos_id
            input_ids[i, 8 + text_len - 1 : 8 + text_len - 1 + codec_len, 1] = audio_codec_0
            input_ids[i, 8 + text_len - 1 + codec_len, 1] = self.config.talker_config.codec_eos_token_id

            codec_0_labels[i, 8 + text_len - 1 : 8 + text_len - 1 + codec_len] = audio_codec_0
            codec_0_labels[i, 8 + text_len - 1 + codec_len] = self.config.talker_config.codec_eos_token_id

            codec_ids[i, 8 + text_len - 1 : 8 + text_len - 1 + codec_len, :] = codes

            codec_embedding_mask[i, 3 : 8 + text_len + codec_len] = True
            # Slot 6 is reserved for explicit speaker embedding injection.
            codec_embedding_mask[i, 6] = False
            codec_mask[i, 8 + text_len - 1 : 8 + text_len - 1 + codec_len] = True
            attention_mask[i, : 8 + text_len + codec_len] = 1

        return {
            "input_ids": input_ids,
            "codec_ids": codec_ids,
            "attention_mask": attention_mask,
            "text_embedding_mask": text_embedding_mask.unsqueeze(-1),
            "codec_embedding_mask": codec_embedding_mask.unsqueeze(-1),
            "codec_0_labels": codec_0_labels,
            "codec_mask": codec_mask,
        }


def _extract_ref_mel_24k(audio_path: str) -> torch.Tensor:
    wav, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if int(sr) != 24000:
        try:
            import librosa  # type: ignore
        except Exception as e:
            raise RuntimeError(
                f"ref audio sample rate is {sr}, but librosa is unavailable for resampling to 24k: {e}"
            )
        wav = librosa.resample(wav, orig_sr=int(sr), target_sr=24000).astype(np.float32)
        sr = 24000

    mel = mel_spectrogram(
        torch.from_numpy(wav).unsqueeze(0),
        n_fft=1024,
        num_mels=128,
        sampling_rate=24000,
        hop_size=256,
        win_size=1024,
        fmin=0,
        fmax=12000,
    ).transpose(1, 2)
    return mel


@torch.inference_mode()
def _resolve_fixed_speaker_embedding(
    model: Qwen3TTSForConditionalGeneration,
    *,
    speaker_embedding_pt: str | None,
    speaker_ref_audio: str | None,
    fixed_spk_id: int,
) -> torch.Tensor:
    hidden_size = int(model.talker.config.hidden_size)

    if speaker_embedding_pt:
        emb = torch.load(speaker_embedding_pt, map_location="cpu")
        if isinstance(emb, dict):
            # Common payload key if user saved {"speaker_embedding": ...}
            emb = emb.get("speaker_embedding", emb.get("embedding", emb))
        if isinstance(emb, torch.Tensor):
            t = emb.detach().cpu().to(torch.float32)
        else:
            t = torch.tensor(emb, dtype=torch.float32)
        if t.ndim == 2 and t.shape[0] == 1:
            t = t[0]
        if t.ndim != 1 or t.numel() != hidden_size:
            raise ValueError(
                f"speaker embedding shape mismatch from {speaker_embedding_pt}: got {tuple(t.shape)}, "
                f"expected ({hidden_size},)"
            )
        print(f"[INFO] speaker_embedding source=pt path={speaker_embedding_pt}")
        return t

    if speaker_ref_audio:
        if model.speaker_encoder is None:
            raise RuntimeError("speaker_ref_audio was provided, but model.speaker_encoder is None.")
        mel = _extract_ref_mel_24k(speaker_ref_audio)
        speaker_emb = model.speaker_encoder(mel.to(model.device).to(model.dtype)).detach().float().cpu()
        if speaker_emb.ndim == 2 and speaker_emb.shape[0] == 1:
            speaker_emb = speaker_emb[0]
        if speaker_emb.ndim != 1 or speaker_emb.numel() != hidden_size:
            raise ValueError(
                f"speaker embedding shape mismatch from ref audio {speaker_ref_audio}: got {tuple(speaker_emb.shape)}, "
                f"expected ({hidden_size},)"
            )
        print(f"[INFO] speaker_embedding source=ref_audio path={speaker_ref_audio}")
        return speaker_emb

    # Fallback: fixed vector from codec embedding table row.
    weight = model.talker.model.codec_embedding.weight.detach().float().cpu()
    if fixed_spk_id < 0 or fixed_spk_id >= weight.shape[0]:
        raise ValueError(f"fixed_spk_id out of range: {fixed_spk_id} not in [0, {weight.shape[0]-1}]")
    print(f"[INFO] speaker_embedding source=codec_embedding_row spk_id={fixed_spk_id}")
    return weight[fixed_spk_id]


def _build_forward_inputs(
    model: Qwen3TTSForConditionalGeneration,
    batch: dict[str, torch.Tensor],
    fixed_speaker_embedding: torch.Tensor,
) -> dict[str, torch.Tensor]:
    input_ids = batch["input_ids"]
    codec_ids = batch["codec_ids"]
    text_embedding_mask = batch["text_embedding_mask"]
    codec_embedding_mask = batch["codec_embedding_mask"]
    attention_mask = batch["attention_mask"]
    codec_0_labels = batch["codec_0_labels"]
    codec_mask = batch["codec_mask"]

    input_text_ids = input_ids[:, :, 0]
    input_codec_ids = input_ids[:, :, 1]
    input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
    input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
    spk = fixed_speaker_embedding.to(input_codec_embedding.device).to(input_codec_embedding.dtype).view(1, 1, -1)
    input_codec_embedding[:, 6, :] = spk
    input_embeddings = input_text_embedding + input_codec_embedding

    for i in range(1, int(model.talker.config.num_code_groups)):
        codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
        codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
        input_embeddings = input_embeddings + codec_i_embedding

    return {
        "input_embeddings": input_embeddings,
        "attention_mask": attention_mask,
        "codec_0_labels": codec_0_labels,
        "codec_ids": codec_ids,
        "codec_mask": codec_mask,
    }


def _forward_losses(model: Qwen3TTSForConditionalGeneration, fwd_inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = model.talker(
        inputs_embeds=fwd_inputs["input_embeddings"][:, :-1, :],
        attention_mask=fwd_inputs["attention_mask"][:, :-1],
        labels=fwd_inputs["codec_0_labels"][:, 1:],
        output_hidden_states=True,
    )
    ce0 = outputs.loss

    hidden_states = outputs.hidden_states[0][-1]
    codec_mask = fwd_inputs["codec_mask"][:, 1:]
    talker_hidden_states = hidden_states[codec_mask]
    talker_codec_ids = fwd_inputs["codec_ids"][fwd_inputs["codec_mask"]]
    _, sub_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)
    total = ce0 + 0.3 * sub_loss
    return total, ce0, sub_loss


def main() -> None:
    p = argparse.ArgumentParser(description="Student training with text-conditioned talker + sub-talker loss.")
    p.add_argument("--teacher-model", default=None, help="Accepted for CLI compatibility; unused in this script.")
    p.add_argument("--kl-alpha", type=float, default=0.0, help="Accepted for CLI compatibility; KL is not used.")
    p.add_argument("--student-config", required=True)
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--max-codec-len", type=int, default=None)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--dry-run-only", action="store_true")
    p.add_argument("--save-strategy", choices=["no", "epoch"], default="no")
    p.add_argument("--fixed-spk-id", type=int, default=0)
    p.add_argument(
        "--adamw-foreach",
        action="store_true",
        help="Enable foreach AdamW kernels. Disabled by default for allocator stability.",
    )
    p.add_argument(
        "--speaker-embedding-pt",
        default=None,
        help="Path to fixed speaker embedding tensor (*.pt). Shape must be [hidden] or [1, hidden].",
    )
    p.add_argument(
        "--speaker-ref-audio",
        default=None,
        help="Reference audio path to extract one fixed speaker embedding (24k expected; auto-resample if librosa exists).",
    )
    args = p.parse_args()

    cfg_dict = json.loads(Path(args.student_config).read_text(encoding="utf-8"))
    model = Qwen3TTSForConditionalGeneration(Qwen3TTSConfig(**cfg_dict))

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    if device == "cuda":
        model = model.to(dtype=torch.bfloat16)

    ds = DistillJsonlDataset(args.train_jsonl, default_num_code_groups=int(model.talker.config.num_code_groups))
    print(f"[INFO] train rows: {len(ds)}" + (f" (skipped={ds.skipped})" if ds.skipped else ""))
    print(f"[INFO] sample train keys: {sorted(ds[0].keys())}")
    if args.max_codec_len is not None:
        print(f"[INFO] max_codec_len={args.max_codec_len}")

    collator = QwenLikeCollator(config=model.config, max_codec_len=args.max_codec_len, fixed_spk_id=args.fixed_spk_id)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        foreach=args.adamw_foreach,
    )
    print(f"[INFO] optimizer=AdamW foreach={'on' if args.adamw_foreach else 'off'}")
    fixed_speaker_embedding = _resolve_fixed_speaker_embedding(
        model,
        speaker_embedding_pt=args.speaker_embedding_pt,
        speaker_ref_audio=args.speaker_ref_audio,
        fixed_spk_id=args.fixed_spk_id,
    )

    # dry-run
    model.train()
    sample = next(iter(loader))
    sample = {k: v.to(device) for k, v in sample.items()}
    fwd_inputs = _build_forward_inputs(model, sample, fixed_speaker_embedding)
    loss, ce0, sub = _forward_losses(model, fwd_inputs)
    loss.backward()
    optimizer.zero_grad(set_to_none=True)
    print(
        f"[OK] dry-run passed: total={float(loss.detach().cpu()):.4f} "
        f"(ce0={float(ce0.detach().cpu()):.4f}, sub={float(sub.detach().cpu()):.4f})"
    )
    if args.dry_run_only:
        print("[DONE] dry-run-only mode.")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(round(args.epochs))
    if epochs <= 0:
        epochs = 1

    global_step = 0
    model.train()
    for ep in range(epochs):
        pbar = tqdm(loader, desc=f"epoch {ep+1}/{epochs}")
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}
            fwd_inputs = _build_forward_inputs(model, batch, fixed_speaker_embedding)
            loss, ce0, sub = _forward_losses(model, fwd_inputs)
            (loss / args.grad_accum).backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            running += float(loss.detach().cpu())
            pbar.set_postfix(
                loss=f"{running / (step + 1):.4f}",
                ce0=f"{float(ce0.detach().cpu()):.4f}",
                sub=f"{float(sub.detach().cpu()):.4f}",
                step=global_step,
            )

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        if args.save_strategy == "epoch":
            ep_dir = out_dir / f"checkpoint-epoch-{ep+1}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ep_dir), safe_serialization=True)

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    model.save_pretrained(str(out_dir), safe_serialization=True)
    print(f"[DONE] training finished. model saved -> {out_dir}")


if __name__ == "__main__":
    main()
