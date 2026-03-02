import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration, mel_spectrogram


def _copy_aux_runtime_files(src_dir: Path, out_dir: Path) -> None:
    if not src_dir.exists() or not src_dir.is_dir():
        return
    skip = {
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "model.safetensors.index.json",
    }
    for p in src_dir.iterdir():
        name = p.name
        if name in skip or name.startswith("model-"):
            continue
        dst = out_dir / name
        if dst.exists():
            continue
        if p.is_dir():
            shutil.copytree(p, dst)
        else:
            shutil.copy2(p, dst)


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
                # Keep top-level model_type for AutoModel loading, drop nested model_type only.
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


def _save_model_robust(
    model: Qwen3TTSForConditionalGeneration,
    out_dir: Path,
    init_model: str | None,
    canonical_cfg: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Avoid transformers config diff path (known KeyError:'dtype' on some envs).
        model.save_pretrained(str(out_dir), safe_serialization=True, save_config=False)
    except Exception as e:
        print(f"[WARN] save_pretrained failed, fallback weight save will be used: {e}")

        # Fallback: save a single safetensors when HF sharded save fails.
        state_dict = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
        save_file(state_dict, str(out_dir / "model.safetensors"))

    # Always write canonical sanitized config from student-config source.
    cfg = _sanitize_config_for_qwen3(canonical_cfg)
    (out_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(str(out_dir))

    if init_model:
        src = Path(init_model)
        if src.exists() and src.is_dir():
            _copy_aux_runtime_files(src, out_dir)
    print(f"[DONE] training finished. model saved -> {out_dir}")


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

    def _resolve_language_id(self, raw_language: Any) -> int | None:
        lang_map = getattr(self.config.talker_config, "codec_language_id", None) or {}
        if not lang_map:
            return None
        if raw_language is None:
            return None
        s = str(raw_language).strip()
        if not s:
            return None
        if s.lower() == "auto":
            return None
        low = s.lower()
        alias = {"ja": "japanese", "japanese": "japanese", "en": "english", "zh": "chinese"}
        cand = alias.get(low, low)
        for k, v in lang_map.items():
            if str(k).lower() == cand:
                return int(v)
        return None

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        b = len(batch)
        g = int(self.config.talker_config.num_code_groups)
        role_len = 3

        text_ids_list = [torch.tensor(x["text_input_ids"], dtype=torch.long) for x in batch]
        codec_2d_list = []
        language_ids: list[int | None] = []
        for x in batch:
            c = x["codec_ids_2d"]
            if self.max_codec_len is not None and self.max_codec_len > 0:
                c = c[: self.max_codec_len]
            codec_2d_list.append(torch.tensor(c, dtype=torch.long))
            language_ids.append(self._resolve_language_id(x.get("language")))

        lens = []
        for t, c, lang_id in zip(text_ids_list, codec_2d_list, language_ids):
            prefill_core_len = 3 if lang_id is None else 4
            base = role_len + prefill_core_len + 2  # 8(auto) / 9(language-specific)
            lens.append(t.shape[0] + c.shape[0] + (base - 8))
        max_len = max(lens) + 8

        input_ids = torch.zeros((b, max_len, 2), dtype=torch.long)
        codec_ids = torch.zeros((b, max_len, g), dtype=torch.long)
        text_embedding_mask = torch.zeros((b, max_len), dtype=torch.bool)
        codec_embedding_mask = torch.zeros((b, max_len), dtype=torch.bool)
        codec_mask = torch.zeros((b, max_len), dtype=torch.bool)
        attention_mask = torch.zeros((b, max_len), dtype=torch.long)
        codec_0_labels = torch.full((b, max_len), -100, dtype=torch.long)
        speaker_pos = torch.zeros((b,), dtype=torch.long)
        audio_start_pos = torch.full((b,), -1, dtype=torch.long)

        for i in range(b):
            text_ids = text_ids_list[i]
            codes = codec_2d_list[i]
            lang_id = language_ids[i]
            if codes.ndim != 2 or codes.shape[1] != g:
                raise ValueError(f"codec shape mismatch at i={i}: got {tuple(codes.shape)}, expected (*,{g})")

            audio_codec_0 = codes[:, 0]
            text_len = text_ids.shape[0]
            codec_len = audio_codec_0.shape[0]
            prefill_core_len = 3 if lang_id is None else 4
            # Talker prefix length before first trailing text token:
            # role(3) + [core + speaker + codec_pad] = role + core + 2
            base = role_len + prefill_core_len + 2
            text_tail_len = text_len - 3
            text_eos_pos = base + text_tail_len
            codec_bos_pos = text_eos_pos + 1
            audio_start = codec_bos_pos + 1
            audio_end = audio_start + codec_len
            codec_eos_pos = audio_end
            audio_start_pos[i] = int(audio_start)

            # text channel
            input_ids[i, :role_len, 0] = text_ids[:role_len]
            pad_count = prefill_core_len + 1
            input_ids[i, role_len : role_len + pad_count, 0] = self.config.tts_pad_token_id
            input_ids[i, role_len + pad_count, 0] = self.config.tts_bos_token_id
            if text_len > 3:
                input_ids[i, base : base + text_tail_len, 0] = text_ids[3:]
            input_ids[i, text_eos_pos, 0] = self.config.tts_eos_token_id
            input_ids[i, codec_bos_pos : codec_eos_pos + 1, 0] = self.config.tts_pad_token_id
            text_embedding_mask[i, : codec_eos_pos + 1] = True

            # codec channel
            if lang_id is None:
                codec_prefill = [
                    self.config.talker_config.codec_nothink_id,
                    self.config.talker_config.codec_think_bos_id,
                    self.config.talker_config.codec_think_eos_id,
                ]
            else:
                codec_prefill = [
                    self.config.talker_config.codec_think_id,
                    self.config.talker_config.codec_think_bos_id,
                    int(lang_id),
                    self.config.talker_config.codec_think_eos_id,
                ]
            codec_prefill += [int(self.fixed_spk_id), self.config.talker_config.codec_pad_id]
            input_ids[i, role_len : role_len + len(codec_prefill), 1] = torch.tensor(codec_prefill, dtype=torch.long)
            speaker_pos[i] = role_len + prefill_core_len

            if text_len > 3:
                input_ids[i, base : base + text_tail_len, 1] = self.config.talker_config.codec_pad_id
            input_ids[i, text_eos_pos, 1] = self.config.talker_config.codec_pad_id
            input_ids[i, codec_bos_pos, 1] = self.config.talker_config.codec_bos_id
            input_ids[i, audio_start:audio_end, 1] = audio_codec_0
            input_ids[i, codec_eos_pos, 1] = self.config.talker_config.codec_eos_token_id

            codec_0_labels[i, audio_start:audio_end] = audio_codec_0
            codec_0_labels[i, codec_eos_pos] = self.config.talker_config.codec_eos_token_id

            codec_ids[i, audio_start:audio_end, :] = codes

            codec_embedding_mask[i, role_len : codec_eos_pos + 1] = True
            # Speaker slot is overwritten by fixed embedding in forward.
            codec_embedding_mask[i, int(speaker_pos[i].item())] = False
            codec_mask[i, audio_start:audio_end] = True
            attention_mask[i, : codec_eos_pos + 1] = 1

        return {
            "input_ids": input_ids,
            "codec_ids": codec_ids,
            "attention_mask": attention_mask,
            "text_embedding_mask": text_embedding_mask.unsqueeze(-1),
            "codec_embedding_mask": codec_embedding_mask.unsqueeze(-1),
            "codec_0_labels": codec_0_labels,
            "codec_mask": codec_mask,
            "speaker_pos": speaker_pos,
            "audio_start_pos": audio_start_pos,
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
    speaker_pos = batch["speaker_pos"]
    audio_start_pos = batch["audio_start_pos"]

    input_text_ids = input_ids[:, :, 0]
    input_codec_ids = input_ids[:, :, 1]
    input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
    input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
    spk = fixed_speaker_embedding.to(input_codec_embedding.device).to(input_codec_embedding.dtype).view(1, -1)
    row = torch.arange(input_codec_embedding.shape[0], device=input_codec_embedding.device)
    input_codec_embedding[row, speaker_pos.to(input_codec_embedding.device), :] = spk.expand(input_codec_embedding.shape[0], -1)
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
        "audio_start_pos": audio_start_pos,
    }


def _forward_losses(
    model: Qwen3TTSForConditionalGeneration,
    fwd_inputs: dict[str, torch.Tensor],
    *,
    step0_weight: float = 0.0,
    infer_like_step0_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Important:
    # Qwen/Transformers causal LM loss path already applies internal next-token shift.
    # Pass full-length inputs/labels here to avoid accidental double-shift.
    outputs = model.talker(
        inputs_embeds=fwd_inputs["input_embeddings"],
        attention_mask=fwd_inputs["attention_mask"],
        labels=fwd_inputs["codec_0_labels"],
        output_hidden_states=True,
    )
    ce0 = outputs.loss

    hidden_states = outputs.hidden_states[0][-1]
    # Align sub-talker inputs explicitly to the same time base used by hidden_states.
    # Keep the previous L-1 time base for sub-talker features.
    cm = fwd_inputs["codec_mask"][:, 1:]
    talker_hidden_states = hidden_states[:, :-1, :][cm]
    talker_codec_ids = fwd_inputs["codec_ids"][:, 1:, :][cm]
    assert talker_codec_ids.shape[0] == talker_hidden_states.shape[0], (
        talker_codec_ids.shape,
        talker_hidden_states.shape,
    )
    sub_logits, _sub_loss_internal = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)
    # Use explicit no-shift CE for sub-talker to avoid hidden causal-LM shift mismatch.
    sub_labels = talker_codec_ids[:, 1:].to(torch.long)  # [N, G-1]
    sub_vocab = sub_logits.shape[-1]
    sub_loss = F.cross_entropy(
        sub_logits.reshape(-1, sub_vocab),
        sub_labels.reshape(-1),
        reduction="mean",
    )

    # Step-0 bootstrap loss: strongly supervise the first audio codec_0 token per sample.
    logits0 = outputs.logits  # [B, L, vocab]
    audio_start_pos = fwd_inputs["audio_start_pos"].to(logits0.device)
    bsz, seqlen = logits0.shape[0], logits0.shape[1]
    bidx = torch.arange(bsz, device=logits0.device, dtype=torch.long)
    valid = (audio_start_pos >= 0) & (audio_start_pos < seqlen)
    if valid.any():
        pos = audio_start_pos[valid].to(torch.long)
        target_step0 = fwd_inputs["codec_ids"][valid, pos, 0].to(torch.long)
        pred_step0 = logits0[valid, pos, :]
        step0_ce = F.cross_entropy(pred_step0, target_step0, reduction="mean")
        step0_acc = (pred_step0.argmax(dim=-1) == target_step0).float().mean()
    else:
        step0_ce = ce0.new_zeros(())
        step0_acc = ce0.new_zeros(())

    # Infer-like step0 stats/loss:
    # Use only prefix up to (audio_start - 1), then predict the first audio token.
    infer_like_losses: list[torch.Tensor] = []
    infer_like_accs: list[torch.Tensor] = []
    bsz = int(audio_start_pos.shape[0])
    for bi in range(bsz):
        start = int(audio_start_pos[bi].item())
        if start <= 0:
            continue
        pref_emb = fwd_inputs["input_embeddings"][bi : bi + 1, :start, :]
        pref_attn = fwd_inputs["attention_mask"][bi : bi + 1, :start]
        o_pref = model.talker(
            inputs_embeds=pref_emb,
            attention_mask=pref_attn,
            output_hidden_states=False,
        )
        logit_next = o_pref.logits[:, -1, :]  # predicts token at position == start
        target_next = fwd_inputs["codec_ids"][bi : bi + 1, start, 0].to(torch.long)
        ce_pref = F.cross_entropy(logit_next, target_next, reduction="mean")
        acc_pref = (logit_next.argmax(dim=-1) == target_next).float().mean()
        infer_like_losses.append(ce_pref)
        infer_like_accs.append(acc_pref)

    if infer_like_losses:
        infer_like_step0_ce = torch.stack(infer_like_losses).mean()
        infer_like_step0_acc = torch.stack(infer_like_accs).mean()
    else:
        infer_like_step0_ce = ce0.new_zeros(())
        infer_like_step0_acc = ce0.new_zeros(())

    total = ce0 + 0.3 * sub_loss + float(step0_weight) * step0_ce + float(infer_like_step0_weight) * infer_like_step0_ce
    return total, ce0, sub_loss, step0_ce, step0_acc, infer_like_step0_ce, infer_like_step0_acc


def main() -> None:
    p = argparse.ArgumentParser(description="Student training with text-conditioned talker + sub-talker loss.")
    p.add_argument("--teacher-model", default=None, help="Accepted for CLI compatibility; unused in this script.")
    p.add_argument("--kl-alpha", type=float, default=0.0, help="Accepted for CLI compatibility; KL is not used.")
    p.add_argument("--student-config", required=True)
    p.add_argument(
        "--init-model",
        default=None,
        help="Optional model dir/HF id to initialize weights from (must match student-config shape).",
    )
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
    p.add_argument(
        "--step0-weight",
        type=float,
        default=0.0,
        help="Additional loss weight for the first audio codec_0 token (step-0 bootstrap).",
    )
    p.add_argument(
        "--infer-like-step0-weight",
        type=float,
        default=0.0,
        help="Additional loss weight for infer-like step-0 (prefix-only next-token prediction).",
    )
    p.add_argument(
        "--log-step0-acc",
        action="store_true",
        help="If set, print step-0 token accuracy in dry-run and training logs.",
    )
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
    canonical_cfg = _sanitize_config_for_qwen3(cfg_dict)
    student_cfg = Qwen3TTSConfig(**canonical_cfg)
    if args.init_model:
        print(f"[INFO] model init source: pretrained ({args.init_model})")
        model = Qwen3TTSForConditionalGeneration.from_pretrained(
            args.init_model,
            torch_dtype=torch.float32,
        )
        # Safety check: fail fast on config mismatch.
        c = model.config.talker_config
        s = student_cfg.talker_config
        mismatch = []
        for k in (
            "num_hidden_layers",
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "num_code_groups",
        ):
            if int(getattr(c, k)) != int(getattr(s, k)):
                mismatch.append(f"{k}: loaded={getattr(c, k)} student_cfg={getattr(s, k)}")
        if mismatch:
            raise ValueError(
                "init-model and student-config mismatch:\n" + "\n".join(mismatch)
            )
    else:
        print("[INFO] model init source: random (from student-config)")
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
    loss, ce0, sub, step0, step0_acc, il_step0, il_step0_acc = _forward_losses(
        model,
        fwd_inputs,
        step0_weight=args.step0_weight,
        infer_like_step0_weight=args.infer_like_step0_weight,
    )
    loss.backward()
    optimizer.zero_grad(set_to_none=True)
    dry = (
        f"[OK] dry-run passed: total={float(loss.detach().cpu()):.4f} "
        f"(ce0={float(ce0.detach().cpu()):.4f}, sub={float(sub.detach().cpu()):.4f}, "
        f"step0={float(step0.detach().cpu()):.4f}, infer_like_step0={float(il_step0.detach().cpu()):.4f})"
    )
    if args.log_step0_acc:
        dry += (
            f" step0_acc={float(step0_acc.detach().cpu()):.4f}"
            f" infer_like_step0_acc={float(il_step0_acc.detach().cpu()):.4f}"
        )
    print(dry)
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
            loss, ce0, sub, step0, step0_acc, il_step0, il_step0_acc = _forward_losses(
                model,
                fwd_inputs,
                step0_weight=args.step0_weight,
                infer_like_step0_weight=args.infer_like_step0_weight,
            )
            (loss / args.grad_accum).backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            running += float(loss.detach().cpu())
            postfix = {
                "loss": f"{running / (step + 1):.4f}",
                "ce0": f"{float(ce0.detach().cpu()):.4f}",
                "sub": f"{float(sub.detach().cpu()):.4f}",
                "step0": f"{float(step0.detach().cpu()):.4f}",
                "il_s0": f"{float(il_step0.detach().cpu()):.4f}",
                "step": global_step,
            }
            if args.log_step0_acc:
                postfix["s0_acc"] = f"{float(step0_acc.detach().cpu()):.4f}"
                postfix["il_s0_acc"] = f"{float(il_step0_acc.detach().cpu()):.4f}"
            pbar.set_postfix(**postfix)

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        if args.save_strategy == "epoch":
            ep_dir = out_dir / f"checkpoint-epoch-{ep+1}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ep_dir), safe_serialization=True)

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    _save_model_robust(model, out_dir, args.init_model, canonical_cfg)


if __name__ == "__main__":
    main()
