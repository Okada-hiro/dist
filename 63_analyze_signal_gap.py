import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration, mel_spectrogram
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalize_text_ids_1d(x: Any) -> torch.Tensor:
    """
    Normalize text ids for single-sample analysis.
    Accepts:
      - [T]
      - [1, T]
      - torch.Tensor of shape [T] or [1, T]
    Returns:
      - torch.LongTensor shape [T]
    """
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu().to(torch.long)
    else:
        t = torch.tensor(x, dtype=torch.long)

    if t.ndim == 1:
        return t
    if t.ndim == 2 and t.shape[0] == 1:
        return t[0]
    raise ValueError(f"text_input_ids must be [T] or [1,T], got shape={tuple(t.shape)}")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


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


def _load_student(model_path_or_id: str, device: str, torch_dtype: torch.dtype, attn_impl: str) -> Qwen3TTSModel:
    try:
        return Qwen3TTSModel.from_pretrained(
            model_path_or_id,
            device_map=device,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
        )
    except Exception as e:
        model_dir = Path(model_path_or_id)
        if not model_dir.exists():
            raise
        print(f"[WARN] wrapper load failed for {model_path_or_id}: {e}")
        print("[INFO] fallback: sanitize config and load core model directly.")
        raw_cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        clean_cfg = _sanitize_config_for_qwen3(raw_cfg)

        with tempfile.TemporaryDirectory(prefix="qwen3tts_sanitized_") as td:
            td_path = Path(td)
            for item in model_dir.iterdir():
                if item.name == "config.json":
                    continue
                dst = td_path / item.name
                try:
                    os.symlink(item, dst, target_is_directory=item.is_dir())
                except OSError:
                    import shutil
                    if item.is_dir():
                        shutil.copytree(item, dst)
                    else:
                        shutil.copy2(item, dst)
            (td_path / "config.json").write_text(json.dumps(clean_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
            cfg_obj = Qwen3TTSConfig(**clean_cfg)
            core_model = Qwen3TTSForConditionalGeneration.from_pretrained(
                str(td_path),
                config=cfg_obj,
                device_map=device,
                torch_dtype=torch_dtype,
                attn_implementation=attn_impl,
            )

        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", fix_mistral_regex=True)
        return Qwen3TTSModel(model=core_model, processor=processor, generate_defaults={})


def _resolve_language_id(model: Qwen3TTSModel, raw_language: Any) -> tuple[str, int | None]:
    lang_map = getattr(model.model.config.talker_config, "codec_language_id", None) or {}
    if raw_language is None:
        return "Auto", None
    s = str(raw_language).strip()
    if not s:
        return "Auto", None
    if s.lower() == "auto":
        return "Auto", None
    alias = {"ja": "japanese", "japanese": "japanese", "en": "english", "zh": "chinese"}
    target = alias.get(s.lower(), s.lower())
    for k, v in lang_map.items():
        if str(k).lower() == target:
            return k, int(v)
    return s, None


def _extract_ref_mel_24k(audio_path: str) -> torch.Tensor:
    wav, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if int(sr) != 24000:
        try:
            import librosa  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Need librosa to resample {sr} -> 24000: {e}")
        wav = librosa.resample(wav, orig_sr=int(sr), target_sr=24000).astype(np.float32)

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
def _resolve_fixed_speaker_embedding(model: Qwen3TTSModel, speaker_ref_audio: str | None, fixed_spk_id: int) -> torch.Tensor:
    if speaker_ref_audio:
        mel = _extract_ref_mel_24k(speaker_ref_audio)
        emb = model.model.speaker_encoder(mel.to(model.model.device).to(model.model.dtype)).detach().float().cpu()
        return emb[0] if emb.ndim == 2 else emb
    w = model.model.talker.model.codec_embedding.weight.detach().float().cpu()
    return w[int(fixed_spk_id)]


def _build_train_like_batch(
    model: Qwen3TTSModel,
    row: dict[str, Any],
    fixed_spk_id: int,
    max_codec_len: int | None = None,
) -> dict[str, torch.Tensor]:
    cfg = model.model.config
    g = int(cfg.talker_config.num_code_groups)
    role_len = 3

    text_ids = row.get("text_input_ids")
    if text_ids is None:
        text_ids = model._tokenize_texts([model._build_assistant_text(str(row["text"]))])[0].detach().cpu().tolist()
    text_ids = _normalize_text_ids_1d(text_ids)

    codes = row.get("codec_ids_2d")
    if not codes:
        flat = row.get("codec_ids_flat", row.get("codec_ids", []))
        n = (len(flat) // g) * g
        flat = flat[:n]
        codes = [flat[i : i + g] for i in range(0, n, g)]
    if max_codec_len and max_codec_len > 0:
        codes = codes[:max_codec_len]
    codes = torch.tensor(codes, dtype=torch.long)

    language, lang_id = _resolve_language_id(model, row.get("language"))
    _ = language
    prefill_core_len = 3 if lang_id is None else 4
    base = role_len + prefill_core_len + 2

    text_len = int(text_ids.shape[0])
    codec_len = int(codes.shape[0])
    text_tail_len = text_len - 3
    text_eos_pos = base + text_tail_len
    codec_bos_pos = text_eos_pos + 1
    audio_start = codec_bos_pos + 1
    audio_end = audio_start + codec_len
    codec_eos_pos = audio_end
    L = codec_eos_pos + 1

    input_ids = torch.zeros((1, L, 2), dtype=torch.long)
    codec_ids = torch.zeros((1, L, g), dtype=torch.long)
    text_embedding_mask = torch.zeros((1, L), dtype=torch.bool)
    codec_embedding_mask = torch.zeros((1, L), dtype=torch.bool)
    codec_mask = torch.zeros((1, L), dtype=torch.bool)
    attention_mask = torch.zeros((1, L), dtype=torch.long)
    codec_0_labels = torch.full((1, L), -100, dtype=torch.long)
    speaker_pos = torch.tensor([role_len + prefill_core_len], dtype=torch.long)

    # text channel
    input_ids[0, :role_len, 0] = text_ids[:role_len]
    pad_count = prefill_core_len + 1
    input_ids[0, role_len : role_len + pad_count, 0] = cfg.tts_pad_token_id
    input_ids[0, role_len + pad_count, 0] = cfg.tts_bos_token_id
    if text_tail_len > 0:
        input_ids[0, base : base + text_tail_len, 0] = text_ids[3:]
    input_ids[0, text_eos_pos, 0] = cfg.tts_eos_token_id
    input_ids[0, codec_bos_pos : codec_eos_pos + 1, 0] = cfg.tts_pad_token_id
    text_embedding_mask[0, : codec_eos_pos + 1] = True

    # codec channel
    if lang_id is None:
        codec_prefill = [
            cfg.talker_config.codec_nothink_id,
            cfg.talker_config.codec_think_bos_id,
            cfg.talker_config.codec_think_eos_id,
        ]
    else:
        codec_prefill = [
            cfg.talker_config.codec_think_id,
            cfg.talker_config.codec_think_bos_id,
            int(lang_id),
            cfg.talker_config.codec_think_eos_id,
        ]
    codec_prefill += [int(fixed_spk_id), cfg.talker_config.codec_pad_id]
    input_ids[0, role_len : role_len + len(codec_prefill), 1] = torch.tensor(codec_prefill, dtype=torch.long)
    if text_tail_len > 0:
        input_ids[0, base : base + text_tail_len, 1] = cfg.talker_config.codec_pad_id
    input_ids[0, text_eos_pos, 1] = cfg.talker_config.codec_pad_id
    input_ids[0, codec_bos_pos, 1] = cfg.talker_config.codec_bos_id
    input_ids[0, audio_start:audio_end, 1] = codes[:, 0]
    input_ids[0, codec_eos_pos, 1] = cfg.talker_config.codec_eos_token_id

    codec_0_labels[0, audio_start:audio_end] = codes[:, 0]
    codec_0_labels[0, codec_eos_pos] = cfg.talker_config.codec_eos_token_id
    codec_ids[0, audio_start:audio_end, :] = codes
    codec_embedding_mask[0, role_len : codec_eos_pos + 1] = True
    codec_embedding_mask[0, int(speaker_pos[0].item())] = False
    codec_mask[0, audio_start:audio_end] = True
    attention_mask[0, : codec_eos_pos + 1] = 1

    return {
        "input_ids": input_ids,
        "codec_ids": codec_ids,
        "attention_mask": attention_mask,
        "text_embedding_mask": text_embedding_mask.unsqueeze(-1),
        "codec_embedding_mask": codec_embedding_mask.unsqueeze(-1),
        "codec_0_labels": codec_0_labels,
        "codec_mask": codec_mask,
        "speaker_pos": speaker_pos,
        "gt_codes": codes,
        "audio_start": audio_start,
        "audio_end": audio_end,
    }


def _teacher_forced_predict(model: Qwen3TTSModel, batch: dict[str, torch.Tensor], fixed_speaker_embedding: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    m = model.model
    device = m.device
    b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    input_ids = b["input_ids"]
    codec_ids = b["codec_ids"]
    text_embedding_mask = b["text_embedding_mask"]
    codec_embedding_mask = b["codec_embedding_mask"]
    attention_mask = b["attention_mask"]
    codec_0_labels = b["codec_0_labels"]
    codec_mask = b["codec_mask"]
    speaker_pos = b["speaker_pos"]
    gt_codes = b["gt_codes"]

    input_text_ids = input_ids[:, :, 0]
    input_codec_ids = input_ids[:, :, 1]
    input_text_embedding = m.talker.model.text_embedding(input_text_ids) * text_embedding_mask
    input_codec_embedding = m.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
    spk = fixed_speaker_embedding.to(input_codec_embedding.device).to(input_codec_embedding.dtype).view(1, -1)
    row = torch.arange(input_codec_embedding.shape[0], device=input_codec_embedding.device)
    input_codec_embedding[row, speaker_pos, :] = spk.expand(input_codec_embedding.shape[0], -1)
    input_embeddings = input_text_embedding + input_codec_embedding

    for i in range(1, int(m.talker.config.num_code_groups)):
        codec_i_embedding = m.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
        codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
        input_embeddings = input_embeddings + codec_i_embedding

    outputs = m.talker(
        inputs_embeds=input_embeddings,
        attention_mask=attention_mask,
        labels=codec_0_labels,
        output_hidden_states=True,
    )
    ce0 = float(outputs.loss.detach().cpu())
    logits0 = outputs.logits[0][:-1]  # [L-1, vocab]
    labels0 = codec_0_labels[:, 1:][0]
    valid = labels0 != -100
    pred0_all = torch.argmax(logits0, dim=-1)
    pred0 = pred0_all[valid]
    gt0 = labels0[valid]

    # Drop EOS token from codec_0 stream for full-code assembly.
    eos_id = int(m.config.talker_config.codec_eos_token_id)
    if pred0.numel() > 0 and int(gt0[-1].item()) == eos_id:
        pred0_audio = pred0[:-1]
        gt0_audio = gt0[:-1]
    else:
        pred0_audio = pred0
        gt0_audio = gt0

    hidden_states = outputs.hidden_states[0][-1]
    cm = codec_mask[:, 1:]
    talker_hidden_states = hidden_states[:, :-1, :][cm]
    talker_codec_ids = codec_ids[:, 1:, :][cm]
    assert talker_codec_ids.shape[0] == talker_hidden_states.shape[0], (
        talker_codec_ids.shape,
        talker_hidden_states.shape,
    )
    sub_logits, sub_loss = m.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)
    sub_loss_v = float(sub_loss.detach().cpu())
    pred_sub = torch.argmax(sub_logits, dim=-1)  # [T, G-1]

    t = min(int(pred0_audio.shape[0]), int(pred_sub.shape[0]), int(gt_codes.shape[0]))
    pred_full = torch.cat([pred0_audio[:t].unsqueeze(-1), pred_sub[:t]], dim=-1).to(torch.long)
    gt_full = gt_codes[:t].to(torch.long)

    codec0_acc = float((pred0_audio[:t] == gt0_audio[:t]).float().mean().item()) if t > 0 else 0.0
    gt_sub = gt_full[:t, 1:]
    sub_acc = float((pred_sub[:t] == gt_sub).float().mean().item()) if t > 0 else 0.0
    full_acc = float((pred_full == gt_full).float().mean().item()) if t > 0 else 0.0

    # Diagnose label/logit alignment explicitly:
    # manual CE on same index (shift 0) vs one-step shifted comparison.
    ce_shift0 = float("nan")
    ce_shift1 = float("nan")
    with torch.no_grad():
        if valid.any():
            ce_shift0 = float(F.cross_entropy(logits0[valid], labels0[valid], reduction="mean").detach().cpu().item())
        # Compare logits[t] with labels[t+1] to detect off-by-one training target alignment.
        if logits0.shape[0] > 1 and labels0.shape[0] > 1:
            logits1 = logits0[:-1]
            labels1 = labels0[1:]
            valid1 = labels1 != -100
            if valid1.any():
                ce_shift1 = float(F.cross_entropy(logits1[valid1], labels1[valid1], reduction="mean").detach().cpu().item())

    c0_best_shift, c0_best_shift_acc = _best_shift_acc(pred0_audio, gt0_audio, max_shift=3)

    # Sub diagnostics: alignment and optional double-shift suspicion.
    sub_acc_shift0 = sub_acc
    sub_acc_shift1 = 0.0
    if t > 1:
        sub_acc_shift1 = float((pred_sub[: t - 1] == gt_sub[1:t]).float().mean().item())

    sub_best_shift = 0
    sub_best_shift_acc = 0.0
    for s in range(-3, 4):
        if s > 0:
            p = pred_sub[s:t]
            g = gt_sub[: t - s]
        elif s < 0:
            p = pred_sub[: t + s]
            g = gt_sub[-s:t]
        else:
            p = pred_sub[:t]
            g = gt_sub[:t]
        n = min(int(p.shape[0]), int(g.shape[0]))
        if n <= 0:
            continue
        a = float((p[:n] == g[:n]).float().mean().item())
        if a > sub_best_shift_acc:
            sub_best_shift_acc = a
            sub_best_shift = s

    sub_ce_shift0 = float("nan")
    sub_ce_shift1 = float("nan")
    with torch.no_grad():
        if t > 0:
            v = int(sub_logits.shape[-1])
            sub_ce_shift0 = float(
                F.cross_entropy(
                    sub_logits[:t].reshape(-1, v),
                    gt_sub.reshape(-1),
                    reduction="mean",
                ).detach().cpu().item()
            )
        if t > 1:
            v = int(sub_logits.shape[-1])
            sub_ce_shift1 = float(
                F.cross_entropy(
                    sub_logits[: t - 1].reshape(-1, v),
                    gt_sub[1:t].reshape(-1),
                    reduction="mean",
                ).detach().cpu().item()
            )

    return pred_full.detach().cpu(), {
        "ce0": ce0,
        "ce0_manual_shift0": ce_shift0,
        "ce0_manual_shift1": ce_shift1,
        "sub_loss": sub_loss_v,
        "codec0_acc_teacher_forced": codec0_acc,
        "codec0_best_shift": int(c0_best_shift),
        "codec0_best_shift_acc": float(c0_best_shift_acc),
        "sub_acc_teacher_forced": sub_acc,
        "sub_acc_shift0": float(sub_acc_shift0),
        "sub_acc_shift1": float(sub_acc_shift1),
        "sub_best_shift": int(sub_best_shift),
        "sub_best_shift_acc": float(sub_best_shift_acc),
        "sub_ce_shift0": sub_ce_shift0,
        "sub_ce_shift1": sub_ce_shift1,
        "full_acc_teacher_forced": full_acc,
        "teacher_forced_len": int(t),
    }


def _infer_generate_codes(
    model: Qwen3TTSModel,
    text: str,
    language: str,
    ref_audio: str | None,
    ref_text: str | None,
    x_vector_only: bool,
    non_streaming_mode: bool,
    max_new_tokens: int,
    use_voice_clone: bool,
    input_ids_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if input_ids_override is not None:
        iid = input_ids_override.detach().clone().to(model.model.device).to(torch.long)
        if iid.ndim == 1:
            iid = iid.unsqueeze(0)
        input_ids = [iid]
        input_ids_source = "train_text_input_ids"
    else:
        input_ids = model._tokenize_texts([model._build_assistant_text(text)])
        input_ids_source = "tokenized_from_text"
    ref_ids = None
    voice_clone_prompt_dict = None
    if use_voice_clone and ref_audio is not None:
        prompt = model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only,
        )
        items = prompt if isinstance(prompt, list) else [prompt]
        voice_clone_prompt_dict = model._prompt_items_to_voice_clone_prompt(items)
        ref_ids = [
            model._tokenize_texts([model._build_ref_text(it.ref_text)])[0] if getattr(it, "ref_text", None) else None
            for it in items
        ]

    codes_list, _ = model.model.generate(
        input_ids=input_ids,
        ref_ids=ref_ids,
        voice_clone_prompt=voice_clone_prompt_dict,
        languages=[language],
        non_streaming_mode=non_streaming_mode,
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )
    c = codes_list[0].detach().cpu().to(torch.long)
    if c.ndim == 1:
        c = c.unsqueeze(-1)
    input_text_len = None
    try:
        if isinstance(input_ids, torch.Tensor):
            input_text_len = int(input_ids.shape[-1])
        elif isinstance(input_ids, list) and len(input_ids) > 0 and hasattr(input_ids[0], "__len__"):
            input_text_len = int(len(input_ids[0]))
    except Exception:
        input_text_len = None

    meta = {
        "use_voice_clone": bool(use_voice_clone and ref_audio is not None),
        "x_vector_only": bool(x_vector_only),
        "non_streaming_mode": bool(non_streaming_mode),
        "language": language,
        "max_new_tokens": int(max_new_tokens),
        "input_ids_source": input_ids_source,
        "input_text_len": input_text_len,
        "ref_ids_len": int(ref_ids[0].shape[-1]) if ref_ids and ref_ids[0] is not None else None,
        "has_voice_clone_prompt": voice_clone_prompt_dict is not None,
    }
    return c, meta


@torch.no_grad()
def _inject_fixed_speaker_row(
    model: Qwen3TTSModel,
    fixed_spk_id: int,
    fixed_speaker_embedding: torch.Tensor,
) -> torch.Tensor:
    """
    Temporarily overwrite codec_embedding row used as speaker token in generate path.
    Returns backup row tensor (cpu) to restore later.
    """
    w = model.model.talker.model.codec_embedding.weight
    idx = int(fixed_spk_id)
    if idx < 0 or idx >= w.shape[0]:
        raise ValueError(f"fixed_spk_id out of range: {idx} not in [0, {w.shape[0]-1}]")
    backup = w[idx].detach().float().cpu().clone()
    w[idx].copy_(fixed_speaker_embedding.to(w.device).to(w.dtype))
    return backup


@torch.no_grad()
def _restore_speaker_row(model: Qwen3TTSModel, fixed_spk_id: int, backup_row: torch.Tensor) -> None:
    w = model.model.talker.model.codec_embedding.weight
    w[int(fixed_spk_id)].copy_(backup_row.to(w.device).to(w.dtype))


def _decode_codes_to_wav(model: Qwen3TTSModel, codes_2d: torch.Tensor, out_path: Path) -> None:
    wavs, sr = model.model.speech_tokenizer.decode([{"audio_codes": codes_2d}])
    sf.write(str(out_path), wavs[0], sr)


def _first_mismatch_step(a: torch.Tensor, b: torch.Tensor) -> int:
    """
    Return first time index where rows differ; -1 if all equal in compared range.
    """
    n = min(int(a.shape[0]), int(b.shape[0]))
    if n <= 0:
        return -1
    diff = (a[:n] != b[:n]).any(dim=-1)
    idx = torch.nonzero(diff, as_tuple=False)
    return int(idx[0].item()) if idx.numel() > 0 else -1


def _prefix_curve(a: torch.Tensor, b: torch.Tensor, max_points: int = 10) -> list[dict[str, float]]:
    """
    Prefix accuracy curve to observe autoregressive drift.
    """
    n = min(int(a.shape[0]), int(b.shape[0]))
    if n <= 0:
        return []
    steps = []
    k = 1
    while k < n and len(steps) < max_points - 1:
        steps.append(k)
        k *= 2
    if not steps or steps[-1] != n:
        steps.append(n)
    out: list[dict[str, float]] = []
    for s in steps:
        acc = float((a[:s] == b[:s]).float().mean().item())
        out.append({"step": float(s), "acc": acc})
    return out


def _best_shift_acc(pred: torch.Tensor, gt: torch.Tensor, max_shift: int = 3) -> tuple[int, float]:
    """
    Compare pred vs gt with integer time shift.
      shift > 0 : pred is delayed (compare pred[shift:] with gt[:-shift])
      shift < 0 : pred is advanced (compare pred[:shift] with gt[-shift:])
    """
    pred = pred.detach().cpu()
    gt = gt.detach().cpu()
    best_s = 0
    best_a = 0.0
    for s in range(-max_shift, max_shift + 1):
        if s > 0:
            p = pred[s:]
            g = gt[:-s]
        elif s < 0:
            p = pred[:s]
            g = gt[-s:]
        else:
            p = pred
            g = gt
        n = min(int(p.shape[0]), int(g.shape[0]))
        if n <= 0:
            continue
        a = float((p[:n] == g[:n]).float().mean().item())
        if a > best_a:
            best_a = a
            best_s = s
    return best_s, best_a


def _codec_value_stats(codes_2d: torch.Tensor) -> dict[str, Any]:
    t = codes_2d.detach().cpu().to(torch.long)
    return {
        "shape": list(t.shape),
        "min": int(t.min().item()) if t.numel() else 0,
        "max": int(t.max().item()) if t.numel() else 0,
    }


def _find_invalid_codec_entries(model: Qwen3TTSModel, codes_2d: torch.Tensor, max_report: int = 16) -> tuple[list[dict[str, int]], int]:
    """
    Return invalid codec entries where value is outside [0, codebook_size-1].
    """
    t = codes_2d.detach().cpu().to(torch.long)
    code_vocab = int(model.model.speech_tokenizer.model.config.decoder_config.codebook_size)
    bad = (t < 0) | (t >= code_vocab)
    idx = torch.nonzero(bad, as_tuple=False)
    total_bad = int(idx.shape[0])
    out = []
    for r in idx[:max_report]:
        rr = int(r[0].item())
        cc = int(r[1].item())
        out.append({"time_index": rr, "group_index": cc, "value": int(t[rr, cc].item())})
    return out, total_bad


def _is_decode_safe(model: Qwen3TTSModel, codes_2d: torch.Tensor) -> tuple[bool, str, list[dict[str, int]], int]:
    """
    Conservative range check before passing codec ids to speech tokenizer decoder.
    """
    t = codes_2d.detach().cpu().to(torch.long)
    if t.ndim != 2:
        return False, f"codes rank must be 2, got {tuple(t.shape)}", [], 0
    if t.shape[1] != int(model.model.config.talker_config.num_code_groups):
        return False, f"num_code_groups mismatch: got {t.shape[1]} expected {int(model.model.config.talker_config.num_code_groups)}", [], 0
    if t.numel() == 0:
        return False, "empty codes", [], 0

    # Speech tokenizer decoder ultimately indexes codebooks with this bound.
    code_vocab = int(model.model.speech_tokenizer.model.config.decoder_config.codebook_size)
    mn = int(t.min().item())
    mx = int(t.max().item())
    if mn < 0 or mx >= code_vocab:
        examples, total_bad = _find_invalid_codec_entries(model, t, max_report=16)
        return False, f"out-of-range codes: min={mn} max={mx} valid=[0,{code_vocab-1}]", examples, total_bad
    return True, "ok", [], 0


def _clamp_codes_for_decode(model: Qwen3TTSModel, codes_2d: torch.Tensor) -> torch.Tensor:
    """
    Best-effort rescue decode for diagnostics only.
    """
    code_vocab = int(model.model.speech_tokenizer.model.config.decoder_config.codebook_size)
    return codes_2d.detach().cpu().to(torch.long).clamp(min=0, max=code_vocab - 1)


def main() -> None:
    p = argparse.ArgumentParser(description="Diagnose 1/2/3 signal gap: teacher vs student-train-vs student-infer.")
    p.add_argument("--student-model", required=True)
    p.add_argument("--teacher-codes-jsonl", required=True)
    p.add_argument("--out-dir", default="dist/signal_gap")
    p.add_argument("--sample-id", default=None, help="Optional id in jsonl to analyze only one sample.")
    p.add_argument("--max-samples", type=int, default=3)
    p.add_argument("--ref-audio", default=None)
    p.add_argument("--ref-text-file", default=None)
    p.add_argument("--x-vector-only", action="store_true")
    p.add_argument("--non-streaming-mode", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--fixed-spk-id", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--force-clamped-decode", action="store_true", help="If set, also decode clamped student codes when out-of-range.")
    p.add_argument("--infer-no-voice-clone", action="store_true", help="Disable voice-clone prompt in inference path even if ref-audio is given.")
    p.add_argument(
        "--infer-use-train-text-ids",
        action="store_true",
        help="Use row.text_input_ids directly for inference input_ids instead of re-tokenizing row.text.",
    )
    p.add_argument(
        "--infer-max-new-tokens-like-teacher",
        action="store_true",
        help="Use teacher codec length as max_new_tokens for each sample.",
    )
    p.add_argument(
        "--infer-inject-fixed-speaker-row",
        action="store_true",
        help="Before inference, temporarily overwrite codec_embedding[fixed_spk_id] with the same fixed speaker vector used in train-like path.",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_text = Path(args.ref_text_file).read_text(encoding="utf-8").strip() if args.ref_text_file else None

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    attn_impl = "flash_attention_2" if str(device).startswith("cuda") else "sdpa"

    model = _load_student(args.student_model, device=device, torch_dtype=torch_dtype, attn_impl=attn_impl)
    fixed_speaker_embedding = _resolve_fixed_speaker_embedding(model, args.ref_audio, args.fixed_spk_id)

    rows = _load_jsonl(Path(args.teacher_codes_jsonl))
    if args.sample_id is not None:
        rows = [r for r in rows if str(r.get("id")) == str(args.sample_id)]
    rows = rows[: args.max_samples]

    report = []
    for i, row in enumerate(rows):
        sid = str(row.get("id", f"{i:06d}"))
        text = str(row["text"])
        language, _ = _resolve_language_id(model, row.get("language"))

        gt_codes = row.get("codec_ids_2d")
        if not gt_codes:
            g = int(model.model.config.talker_config.num_code_groups)
            flat = row.get("codec_ids_flat", row.get("codec_ids", []))
            n = (len(flat) // g) * g
            gt_codes = [flat[j : j + g] for j in range(0, n, g)]
        gt_codes_t = torch.tensor(gt_codes, dtype=torch.long)

        batch = _build_train_like_batch(model, row, fixed_spk_id=args.fixed_spk_id, max_codec_len=None)
        train_pred_codes, train_stats = _teacher_forced_predict(model, batch, fixed_speaker_embedding)

        backup_row = None
        try:
            if args.infer_inject_fixed_speaker_row:
                backup_row = _inject_fixed_speaker_row(model, args.fixed_spk_id, fixed_speaker_embedding)
            infer_input_ids_override = None
            if args.infer_use_train_text_ids and row.get("text_input_ids") is not None:
                infer_input_ids_override = _normalize_text_ids_1d(row.get("text_input_ids"))
            infer_max_new_tokens = int(gt_codes_t.shape[0]) if args.infer_max_new_tokens_like_teacher else int(args.max_new_tokens)
            infer_codes, infer_meta = _infer_generate_codes(
                model=model,
                text=text,
                language=language,
                ref_audio=args.ref_audio,
                ref_text=ref_text,
                x_vector_only=args.x_vector_only,
                non_streaming_mode=args.non_streaming_mode,
                max_new_tokens=infer_max_new_tokens,
                use_voice_clone=not args.infer_no_voice_clone,
                input_ids_override=infer_input_ids_override,
            )
            infer_meta["inject_fixed_speaker_row"] = bool(args.infer_inject_fixed_speaker_row)
            infer_meta["max_new_tokens_like_teacher"] = bool(args.infer_max_new_tokens_like_teacher)
        finally:
            if backup_row is not None:
                _restore_speaker_row(model, args.fixed_spk_id, backup_row)

        n_ti = min(int(train_pred_codes.shape[0]), int(infer_codes.shape[0]))
        ti_acc = float((train_pred_codes[:n_ti] == infer_codes[:n_ti]).float().mean().item()) if n_ti > 0 else 0.0
        n_gt_inf = min(int(gt_codes_t.shape[0]), int(infer_codes.shape[0]))
        gi_acc = float((gt_codes_t[:n_gt_inf] == infer_codes[:n_gt_inf]).float().mean().item()) if n_gt_inf > 0 else 0.0
        best_shift_t2, best_acc_t2 = _best_shift_acc(train_pred_codes, gt_codes_t, max_shift=3)
        best_shift_t3, best_acc_t3 = _best_shift_acc(infer_codes, gt_codes_t, max_shift=3)
        first_mismatch_2v3 = _first_mismatch_step(train_pred_codes, infer_codes)
        first_mismatch_t3 = _first_mismatch_step(gt_codes_t, infer_codes)
        curve_2v3 = _prefix_curve(train_pred_codes, infer_codes, max_points=10)
        curve_t3 = _prefix_curve(gt_codes_t, infer_codes, max_points=10)

        codec_eos_id = int(model.model.config.talker_config.codec_eos_token_id)
        infer_codec0 = infer_codes[:, 0] if infer_codes.ndim == 2 and infer_codes.shape[1] > 0 else torch.empty(0, dtype=torch.long)
        eos_pos = torch.nonzero(infer_codec0 == codec_eos_id, as_tuple=False).flatten()
        eos_count = int(eos_pos.numel())
        eos_first = int(eos_pos[0].item()) if eos_count > 0 else None

        teacher_wav = out_dir / f"{sid}.teacher_signal.wav"
        student_train_wav = out_dir / f"{sid}.student_train_signal.wav"
        student_infer_wav = out_dir / f"{sid}.student_infer_signal.wav"

        decode_notes: list[str] = []
        ok_t, msg_t, ex_t, nbad_t = _is_decode_safe(model, gt_codes_t)
        ok_2, msg_2, ex_2, nbad_2 = _is_decode_safe(model, train_pred_codes)
        ok_3, msg_3, ex_3, nbad_3 = _is_decode_safe(model, infer_codes)

        if ok_t:
            _decode_codes_to_wav(model, gt_codes_t, teacher_wav)
        else:
            decode_notes.append(f"teacher_decode_skipped: {msg_t}")
        if ok_2:
            _decode_codes_to_wav(model, train_pred_codes, student_train_wav)
        else:
            decode_notes.append(f"student_train_decode_skipped: {msg_2}")
            decode_notes.append(f"student_train_invalid_total: {nbad_2}")
            if ex_2:
                decode_notes.append(f"student_train_invalid_examples: {json.dumps(ex_2, ensure_ascii=False)}")
            if args.force_clamped_decode:
                clamped = _clamp_codes_for_decode(model, train_pred_codes)
                clamped_wav = out_dir / f"{sid}.student_train_signal.clamped.wav"
                _decode_codes_to_wav(model, clamped, clamped_wav)
                decode_notes.append(f"student_train_clamped_wav: {str(clamped_wav)}")
        if ok_3:
            _decode_codes_to_wav(model, infer_codes, student_infer_wav)
        else:
            decode_notes.append(f"student_infer_decode_skipped: {msg_3}")
            decode_notes.append(f"student_infer_invalid_total: {nbad_3}")
            if ex_3:
                decode_notes.append(f"student_infer_invalid_examples: {json.dumps(ex_3, ensure_ascii=False)}")
            if args.force_clamped_decode:
                clamped = _clamp_codes_for_decode(model, infer_codes)
                clamped_wav = out_dir / f"{sid}.student_infer_signal.clamped.wav"
                _decode_codes_to_wav(model, clamped, clamped_wav)
                decode_notes.append(f"student_infer_clamped_wav: {str(clamped_wav)}")

        item = {
            "id": sid,
            "text": text,
            "language": language,
            "train_like_condition": {
                "speaker_pos": int(batch["speaker_pos"][0].item()),
                "input_ids_text_prefix": batch["input_ids"][0, :12, 0].tolist(),
                "input_ids_codec_prefix": batch["input_ids"][0, :12, 1].tolist(),
                "codec_mask_true_count": int(batch["codec_mask"][0].sum().item()),
            },
            "infer_condition": infer_meta,
            "infer_diagnostics": {
                "codec_eos_id": codec_eos_id,
                "codec0_eos_count": eos_count,
                "codec0_first_eos_pos": eos_first,
                "infer_hit_max_new_tokens": bool(int(infer_codes.shape[0]) >= int(infer_meta["max_new_tokens"])),
                "infer_len_ratio_vs_teacher": float(int(infer_codes.shape[0]) / max(1, int(gt_codes_t.shape[0]))),
                "first_mismatch_step_train_vs_infer": int(first_mismatch_2v3),
                "first_mismatch_step_teacher_vs_infer": int(first_mismatch_t3),
                "prefix_curve_train_vs_infer": curve_2v3,
                "prefix_curve_teacher_vs_infer": curve_t3,
            },
            "teacher_len": int(gt_codes_t.shape[0]),
            "student_train_len": int(train_pred_codes.shape[0]),
            "student_infer_len": int(infer_codes.shape[0]),
            "teacher_vs_train_full_acc": train_stats["full_acc_teacher_forced"],
            "teacher_vs_train_codec0_acc": train_stats["codec0_acc_teacher_forced"],
            "teacher_vs_train_codec0_best_shift": train_stats["codec0_best_shift"],
            "teacher_vs_train_codec0_best_shift_acc": train_stats["codec0_best_shift_acc"],
            "teacher_vs_train_sub_acc": train_stats["sub_acc_teacher_forced"],
            "teacher_vs_train_sub_acc_shift0": train_stats["sub_acc_shift0"],
            "teacher_vs_train_sub_acc_shift1": train_stats["sub_acc_shift1"],
            "teacher_vs_train_sub_best_shift": train_stats["sub_best_shift"],
            "teacher_vs_train_sub_best_shift_acc": train_stats["sub_best_shift_acc"],
            "teacher_vs_train_sub_ce_shift0": train_stats["sub_ce_shift0"],
            "teacher_vs_train_sub_ce_shift1": train_stats["sub_ce_shift1"],
            "teacher_vs_infer_full_acc_minlen": gi_acc,
            "train_vs_infer_full_acc_minlen": ti_acc,
            "teacher_vs_train_best_shift": int(best_shift_t2),
            "teacher_vs_train_best_shift_acc": float(best_acc_t2),
            "teacher_vs_infer_best_shift": int(best_shift_t3),
            "teacher_vs_infer_best_shift_acc": float(best_acc_t3),
            "ce0": train_stats["ce0"],
            "ce0_manual_shift0": train_stats["ce0_manual_shift0"],
            "ce0_manual_shift1": train_stats["ce0_manual_shift1"],
            "sub_loss": train_stats["sub_loss"],
            "teacher_forced_len": train_stats["teacher_forced_len"],
            "teacher_codec_stats": _codec_value_stats(gt_codes_t),
            "student_train_codec_stats": _codec_value_stats(train_pred_codes),
            "student_infer_codec_stats": _codec_value_stats(infer_codes),
            "teacher_invalid_count": nbad_t,
            "student_train_invalid_count": nbad_2,
            "student_infer_invalid_count": nbad_3,
            "teacher_invalid_examples": ex_t,
            "student_train_invalid_examples": ex_2,
            "student_infer_invalid_examples": ex_3,
            "teacher_wav": str(teacher_wav) if ok_t else None,
            "student_train_wav": str(student_train_wav) if ok_2 else None,
            "student_infer_wav": str(student_infer_wav) if ok_3 else None,
            "decode_notes": decode_notes,
        }
        report.append(item)
        print(
            f"[ROW] {sid} "
            f"acc(Tvs2_full)={item['teacher_vs_train_full_acc']:.4f} "
            f"acc(Tvs2_c0)={item['teacher_vs_train_codec0_acc']:.4f} "
            f"bestShift_c0={item['teacher_vs_train_codec0_best_shift']}@{item['teacher_vs_train_codec0_best_shift_acc']:.4f} "
            f"acc(Tvs2_sub)={item['teacher_vs_train_sub_acc']:.4f} "
            f"sub(s0/s1)={item['teacher_vs_train_sub_acc_shift0']:.4f}/{item['teacher_vs_train_sub_acc_shift1']:.4f} "
            f"bestShift_sub={item['teacher_vs_train_sub_best_shift']}@{item['teacher_vs_train_sub_best_shift_acc']:.4f} "
            f"subCE(s0/s1)={item['teacher_vs_train_sub_ce_shift0']:.4f}/{item['teacher_vs_train_sub_ce_shift1']:.4f} "
            f"acc(Tvs3)={item['teacher_vs_infer_full_acc_minlen']:.4f} "
            f"bestShift(Tvs2)={item['teacher_vs_train_best_shift']}@{item['teacher_vs_train_best_shift_acc']:.4f} "
            f"bestShift(Tvs3)={item['teacher_vs_infer_best_shift']}@{item['teacher_vs_infer_best_shift_acc']:.4f} "
            f"ce0(manual s0/s1)={item['ce0_manual_shift0']:.4f}/{item['ce0_manual_shift1']:.4f} "
            f"acc(2vs3)={item['train_vs_infer_full_acc_minlen']:.4f} "
            f"len(T/2/3)={item['teacher_len']}/{item['student_train_len']}/{item['student_infer_len']} "
            f"infer(vc/xv/lang)={int(item['infer_condition']['use_voice_clone'])}/{int(item['infer_condition']['x_vector_only'])}/{item['infer_condition']['language']} "
            f"eos(c0)={item['infer_diagnostics']['codec0_eos_count']} first={item['infer_diagnostics']['codec0_first_eos_pos']} "
            f"mismatch(t->i)={item['infer_diagnostics']['first_mismatch_step_teacher_vs_infer']}"
        )
        for n in decode_notes:
            print(f"[NOTE] {sid}: {n}")

    _write_json(out_dir / "signal_gap_report.json", report)
    print(f"[DONE] wrote report -> {out_dir / 'signal_gap_report.json'}")


if __name__ == "__main__":
    main()
