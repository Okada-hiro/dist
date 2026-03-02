import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

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
        inputs_embeds=input_embeddings[:, :-1, :],
        attention_mask=attention_mask[:, :-1],
        labels=codec_0_labels[:, 1:],
        output_hidden_states=True,
    )
    ce0 = float(outputs.loss.detach().cpu())
    logits0 = outputs.logits[0]  # [L-1, vocab]
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
    talker_hidden_states = hidden_states[codec_mask[:, 1:]]
    talker_codec_ids = codec_ids[codec_mask]
    sub_logits, sub_loss = m.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)
    sub_loss_v = float(sub_loss.detach().cpu())
    pred_sub = torch.argmax(sub_logits, dim=-1)  # [T, G-1]

    t = min(int(pred0_audio.shape[0]), int(pred_sub.shape[0]), int(gt_codes.shape[0]))
    pred_full = torch.cat([pred0_audio[:t].unsqueeze(-1), pred_sub[:t]], dim=-1).to(torch.long)
    gt_full = gt_codes[:t].to(torch.long)

    codec0_acc = float((pred0_audio[:t] == gt0_audio[:t]).float().mean().item()) if t > 0 else 0.0
    sub_acc = float((pred_sub[:t] == gt_full[:t, 1:]).float().mean().item()) if t > 0 else 0.0
    full_acc = float((pred_full == gt_full).float().mean().item()) if t > 0 else 0.0

    return pred_full.detach().cpu(), {
        "ce0": ce0,
        "sub_loss": sub_loss_v,
        "codec0_acc_teacher_forced": codec0_acc,
        "sub_acc_teacher_forced": sub_acc,
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
) -> torch.Tensor:
    input_ids = model._tokenize_texts([model._build_assistant_text(text)])
    ref_ids = None
    voice_clone_prompt_dict = None
    if ref_audio is not None:
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
    return c


def _decode_codes_to_wav(model: Qwen3TTSModel, codes_2d: torch.Tensor, out_path: Path) -> None:
    wavs, sr = model.model.speech_tokenizer.decode([{"audio_codes": codes_2d}])
    sf.write(str(out_path), wavs[0], sr)


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

        infer_codes = _infer_generate_codes(
            model=model,
            text=text,
            language=language,
            ref_audio=args.ref_audio,
            ref_text=ref_text,
            x_vector_only=args.x_vector_only,
            non_streaming_mode=args.non_streaming_mode,
            max_new_tokens=args.max_new_tokens,
        )

        n_ti = min(int(train_pred_codes.shape[0]), int(infer_codes.shape[0]))
        ti_acc = float((train_pred_codes[:n_ti] == infer_codes[:n_ti]).float().mean().item()) if n_ti > 0 else 0.0
        n_gt_inf = min(int(gt_codes_t.shape[0]), int(infer_codes.shape[0]))
        gi_acc = float((gt_codes_t[:n_gt_inf] == infer_codes[:n_gt_inf]).float().mean().item()) if n_gt_inf > 0 else 0.0

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
            "teacher_len": int(gt_codes_t.shape[0]),
            "student_train_len": int(train_pred_codes.shape[0]),
            "student_infer_len": int(infer_codes.shape[0]),
            "teacher_vs_train_full_acc": train_stats["full_acc_teacher_forced"],
            "teacher_vs_train_codec0_acc": train_stats["codec0_acc_teacher_forced"],
            "teacher_vs_train_sub_acc": train_stats["sub_acc_teacher_forced"],
            "teacher_vs_infer_full_acc_minlen": gi_acc,
            "train_vs_infer_full_acc_minlen": ti_acc,
            "ce0": train_stats["ce0"],
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
            f"acc(Tvs2_sub)={item['teacher_vs_train_sub_acc']:.4f} "
            f"acc(Tvs3)={item['teacher_vs_infer_full_acc_minlen']:.4f} "
            f"acc(2vs3)={item['train_vs_infer_full_acc_minlen']:.4f} "
            f"len(T/2/3)={item['teacher_len']}/{item['student_train_len']}/{item['student_infer_len']}"
        )
        for n in decode_notes:
            print(f"[NOTE] {sid}: {n}")

    _write_json(out_dir / "signal_gap_report.json", report)
    print(f"[DONE] wrote report -> {out_dir / 'signal_gap_report.json'}")


if __name__ == "__main__":
    main()
