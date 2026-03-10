"""
autotranslator.py  —  HuggingFace-native version
=================================================
Replaces:
  ❌  deep-translator  →  ✅  facebook/nllb-200-distilled-600M
  ❌  gtts / edge-tts  →  ✅  microsoft/speecht5_tts        (English)
                             facebook/mms-tts-{lang}        (every other language)

All models run fully offline inside HF Spaces — zero external API calls.

Public API is 100% unchanged (same function names / signatures as before):
  detect_language(text)            → str
  translate_to_english(text, src)  → str
  translate_to(text, target)       → str
  text_to_speech_bytes(text, lang) → bytes  (MP3)
  speech_to_text(audio_bytes)      → (str, str)
  get_user_language(sender_id)     → str | None
  set_user_language(sender_id, lang)
  download_audio(url, token)       → bytes  (async)
"""

from __future__ import annotations

import io
import os
import tempfile
import logging
from typing import Optional

import numpy as np
import noisereduce as nr
import soundfile as sf
import torch
import httpx
from langdetect import detect_langs, DetectorFactory
from faster_whisper import WhisperModel
from pydub import AudioSegment
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    VitsModel,
)

logger = logging.getLogger(__name__)
DetectorFactory.seed = 0

# ─────────────────────────────────────────────────────────────────────────────
# LANGUAGE CODE MAPS
# ─────────────────────────────────────────────────────────────────────────────

# ISO 639-1 (2-letter)  →  NLLB-200 BCP-47 tag
NLLB_LANG_MAP: dict[str, str] = {
    # ── Indian ───────────────────────────────────────────────────────────────
    "en":    "eng_Latn",
    "hi":    "hin_Deva",
    "te":    "tel_Telu",
    "ta":    "tam_Taml",
    "kn":    "kan_Knda",
    "ml":    "mal_Mlym",
    "bn":    "ben_Beng",
    "mr":    "mar_Deva",
    "gu":    "guj_Gujr",
    "pa":    "pan_Guru",
    "ur":    "urd_Arab",
    "or":    "ory_Orya",
    "si":    "sin_Sinh",
    "ne":    "npi_Deva",
    # ── European ─────────────────────────────────────────────────────────────
    "fr":    "fra_Latn",
    "de":    "deu_Latn",
    "es":    "spa_Latn",
    "pt":    "por_Latn",
    "it":    "ita_Latn",
    "ru":    "rus_Cyrl",
    "pl":    "pol_Latn",
    "nl":    "nld_Latn",
    "sv":    "swe_Latn",
    "da":    "dan_Latn",
    "fi":    "fin_Latn",
    "no":    "nob_Latn",
    "nb":    "nob_Latn",
    "cs":    "ces_Latn",
    "sk":    "slk_Latn",
    "ro":    "ron_Latn",
    "hu":    "hun_Latn",
    "el":    "ell_Grek",
    "bg":    "bul_Cyrl",
    "hr":    "hrv_Latn",
    "uk":    "ukr_Cyrl",
    "sr":    "srp_Cyrl",
    "sl":    "slv_Latn",
    "lt":    "lit_Latn",
    "lv":    "lvs_Latn",
    "et":    "est_Latn",
    "ca":    "cat_Latn",
    "gl":    "glg_Latn",
    "eu":    "eus_Latn",
    "cy":    "cym_Latn",
    "ga":    "gle_Latn",
    "is":    "isl_Latn",
    "mt":    "mlt_Latn",
    # ── Middle East / Central Asia ───────────────────────────────────────────
    "tr":    "tur_Latn",
    "ar":    "arb_Arab",
    "fa":    "pes_Arab",
    "he":    "heb_Hebr",
    "iw":    "heb_Hebr",   # legacy ISO code for Hebrew
    "az":    "azj_Latn",
    "kk":    "kaz_Cyrl",
    "uz":    "uzn_Latn",
    "ky":    "kir_Cyrl",
    "ka":    "kat_Geor",
    "hy":    "hye_Armn",
    # ── East / SE Asia ───────────────────────────────────────────────────────
    "zh":    "zho_Hans",
    "zh-cn": "zho_Hans",
    "zh-tw": "zho_Hant",
    "ja":    "jpn_Jpan",
    "ko":    "kor_Hang",
    "id":    "ind_Latn",
    "ms":    "zsm_Latn",
    "th":    "tha_Thai",
    "vi":    "vie_Latn",
    "km":    "khm_Khmr",
    "my":    "mya_Mymr",
    "lo":    "lao_Laoo",
    "mn":    "khk_Cyrl",
    "fil":   "fil_Latn",
    # ── Africa ───────────────────────────────────────────────────────────────
    "af":    "afr_Latn",
    "sw":    "swh_Latn",
    "am":    "amh_Ethi",
    "yo":    "yor_Latn",
    "ig":    "ibo_Latn",
    "zu":    "zul_Latn",
    "xh":    "xho_Latn",
    "so":    "som_Latn",
}

# ISO 639-1  →  MMS-TTS ISO 639-3 suffix  (model ID: facebook/mms-tts-{code})
MMS_LANG_MAP: dict[str, str] = {
    "en":    "eng",
    "hi":    "hin",
    "te":    "tel",
    "ta":    "tam",
    "kn":    "kan",
    "ml":    "mal",
    "bn":    "ben",
    "mr":    "mar",
    "gu":    "guj",
    "pa":    "pan",
    "ur":    "urd",
    "or":    "ory",
    "si":    "sin",
    "ne":    "npi",
    "fr":    "fra",
    "de":    "deu",
    "es":    "spa",
    "pt":    "por",
    "it":    "ita",
    "ru":    "rus",
    "pl":    "pol",
    "nl":    "nld",
    "sv":    "swe",
    "da":    "dan",
    "fi":    "fin",
    "no":    "nor",
    "nb":    "nor",
    "cs":    "ces",
    "sk":    "slk",
    "ro":    "ron",
    "hu":    "hun",
    "el":    "ell",
    "bg":    "bul",
    "hr":    "hrv",
    "uk":    "ukr",
    "sr":    "srp",
    "sl":    "slv",
    "lt":    "lit",
    "lv":    "lav",
    "et":    "est",
    "ca":    "cat",
    "gl":    "glg",
    "eu":    "eus",
    "cy":    "cym",
    "ga":    "gle",
    "mt":    "mlt",
    "tr":    "tur",
    "ar":    "ara",   # facebook/mms-tts-ara  (arb is not a valid MMS model ID)
    "fa":    "pes",
    "he":    "heb",
    "iw":    "heb",
    "az":    "azj",
    "kk":    "kaz",
    "uz":    "uzn",
    "ka":    "kat",
    "hy":    "hye",
    "zh":    "cmn",
    "zh-cn": "cmn",
    "zh-tw": "cmn",
    "ja":    "jpn",
    "ko":    "kor",
    "id":    "ind",
    "ms":    "zsm",
    "th":    "tha",
    "vi":    "vie",
    "km":    "khm",
    "my":    "mya",
    "mn":    "khk",
    "fil":   "fil",
    "af":    "afr",
    "sw":    "swh",
    "am":    "amh",
    "yo":    "yor",
    "zu":    "zul",
    "xh":    "xho",
    "so":    "som",
}

# In-memory language store:  sender_id → ISO 639-1 code
user_languages: dict[str, str] = {}


# ─────────────────────────────────────────────────────────────────────────────
# LANGUAGE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

# Unicode script ranges → language code (fast, no model needed)
_SCRIPT_RANGES: list[tuple[int, int, str | None]] = [
    (0x0900, 0x097F, "hi"),     # Devanagari → Hindi
    (0x0980, 0x09FF, "bn"),     # Bengali
    (0x0A00, 0x0A7F, "pa"),     # Gurmukhi → Punjabi
    (0x0A80, 0x0AFF, "gu"),     # Gujarati
    (0x0B00, 0x0B7F, "or"),     # Odia
    (0x0B80, 0x0BFF, "ta"),     # Tamil
    (0x0C00, 0x0C7F, "te"),     # Telugu
    (0x0C80, 0x0CFF, "kn"),     # Kannada
    (0x0D00, 0x0D7F, "ml"),     # Malayalam
    (0x0D80, 0x0DFF, "si"),     # Sinhala
    (0x0E00, 0x0E7F, "th"),     # Thai
    (0x0E80, 0x0EFF, "lo"),     # Lao
    (0x0F00, 0x0FFF, "bo"),     # Tibetan
    (0x1000, 0x109F, "my"),     # Myanmar / Burmese
    (0x10A0, 0x10FF, "ka"),     # Georgian
    (0x1200, 0x137F, "am"),     # Ethiopic → Amharic
    (0x1780, 0x17FF, "km"),     # Khmer
    (0x1800, 0x18AF, "mn"),     # Mongolian
    (0x0530, 0x058F, "hy"),     # Armenian
    (0x0590, 0x05FF, "he"),     # Hebrew
    (0x0600, 0x06FF, "ar"),     # Arabic
    (0x0750, 0x077F, "ar"),     # Arabic Supplement
    (0x3040, 0x309F, "ja"),     # Hiragana
    (0x30A0, 0x30FF, "ja"),     # Katakana
    (0x4E00, 0x9FFF, "zh"),     # CJK → Chinese
    (0xAC00, 0xD7AF, "ko"),     # Hangul → Korean
    (0x0400, 0x04FF, None),     # Cyrillic — delegate to langdetect (ru/uk/bg/sr)
]


def _detect_script_language(text: str) -> str | None:
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        for start, end, lang in _SCRIPT_RANGES:
            if start <= cp <= end:
                if lang:
                    counts[lang] = counts.get(lang, 0) + 1
                break
    if not counts:
        return None
    return max(counts, key=counts.__getitem__)


def detect_language(text: str) -> str:
    """
    Detect language of a typed message.
    1. Unicode script ranges  (zero latency, handles all non-Latin scripts)
    2. langdetect             (Latin / Cyrillic disambiguation)
    Returns ISO 639-1 code, defaults to 'en'.
    """
    script = _detect_script_language(text)
    if script:
        return script
    try:
        langs = detect_langs(text)
        # Short Latin text (≤15 chars) needs higher confidence — langdetect is
        # unreliable on single words, e.g. "welcome" gets labelled Dutch (nl).
        min_prob = 0.95 if len(text.strip()) <= 15 else 0.80
        if langs and langs[0].prob >= min_prob:
            return langs[0].lang
    except Exception:
        pass
    return "en"


# ─────────────────────────────────────────────────────────────────────────────
# TRANSLATION  —  facebook/nllb-200-distilled-600M
# ─────────────────────────────────────────────────────────────────────────────

_NLLB_MODEL_ID = "facebook/nllb-200-distilled-600M"
_nllb_tokenizer: AutoTokenizer | None = None
_nllb_model: AutoModelForSeq2SeqLM | None = None


def _get_nllb() -> tuple:
    """Load NLLB tokenizer + model once; reuse on every subsequent call.
    Uses model directly (not pipeline) so it works in transformers v4 AND v5.
    The transformers v5 pipeline removed the 'translation' task name entirely.
    """
    global _nllb_tokenizer, _nllb_model
    if _nllb_model is None:
        logger.info("Loading NLLB-200 translation model…")
        _nllb_tokenizer = AutoTokenizer.from_pretrained(_NLLB_MODEL_ID)
        _nllb_model = AutoModelForSeq2SeqLM.from_pretrained(_NLLB_MODEL_ID)
        _nllb_model.eval()
        logger.info("NLLB-200 ready.")
    return _nllb_tokenizer, _nllb_model


def _nllb_translate(text: str, src_nllb: str, tgt_nllb: str) -> str:
    """Translate text between two NLLB BCP-47 language codes."""
    tokenizer, model = _get_nllb()
    # Force the correct source language token
    tokenizer.src_lang = src_nllb
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    # Get the target language token id for forced_bos
    tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_nllb)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_id,
            max_length=512,
            num_beams=4,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def _nllb_code(lang: str) -> str:
    """ISO 639-1 → NLLB BCP-47 tag. Falls back to English."""
    code = lang.lower()
    return (
        NLLB_LANG_MAP.get(code)
        or NLLB_LANG_MAP.get(code.split("-")[0], "eng_Latn")
    )


def translate_to_english(text: str, source_lang: str = "auto") -> str:
    """Translate any text → English using NLLB-200."""
    if not text or not text.strip():
        return text
    src = (
        source_lang
        if source_lang and source_lang not in ("auto", "en")
        else detect_language(text)
    )
    if src == "en":
        return text
    try:
        return _nllb_translate(text, src_nllb=_nllb_code(src), tgt_nllb="eng_Latn")
    except Exception as e:
        logger.error(f"NLLB →en error [{src}]: {e}")
        return text


def translate_to(text: str, target_lang: str) -> str:
    """Translate English text → target language using NLLB-200."""
    if not text or not text.strip() or target_lang == "en":
        return text
    try:
        return _nllb_translate(text, src_nllb="eng_Latn", tgt_nllb=_nllb_code(target_lang))
    except Exception as e:
        logger.error(f"NLLB →{target_lang} error: {e}")
        return text


# ─────────────────────────────────────────────────────────────────────────────
# TTS  —  SpeechT5 (English)  +  MMS-TTS (everything else)
# ─────────────────────────────────────────────────────────────────────────────

# ── SpeechT5 (English) ───────────────────────────────────────────────────────
_t5_processor: SpeechT5Processor | None = None
_t5_model: SpeechT5ForTextToSpeech | None = None
_t5_vocoder: SpeechT5HifiGan | None = None
_t5_speaker_emb: torch.Tensor | None = None


def _get_speecht5():
    global _t5_processor, _t5_model, _t5_vocoder, _t5_speaker_emb
    if _t5_model is None:
        logger.info("Loading SpeechT5 TTS…")
        _t5_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        _t5_model     = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        _t5_vocoder   = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        # Default speaker embedding — load from pre-cached .npy or fallback to zeros
        _HF_CACHE = "/app/.cache/huggingface"
        _EMB_PATH = os.path.join(_HF_CACHE, "speecht5_speaker_embedding.npy")
        try:
            import numpy as _np
            xvec = _np.load(_EMB_PATH)
            _t5_speaker_emb = torch.tensor(xvec).unsqueeze(0)
            logger.info("SpeechT5 speaker embedding loaded from cache.")
        except Exception:
            logger.warning("Speaker embedding cache not found — using zero fallback.")
            _t5_speaker_emb = torch.zeros(1, 512)
        logger.info("SpeechT5 ready.")
    return _t5_processor, _t5_model, _t5_vocoder, _t5_speaker_emb


def _speecht5_tts(text: str) -> bytes:
    """English TTS — SpeechT5 + HiFiGAN vocoder → MP3 bytes."""
    proc, model, vocoder, spk = _get_speecht5()
    inputs = proc(text=text, return_tensors="pt")
    with torch.no_grad():
        speech = model.generate_speech(inputs["input_ids"], spk, vocoder=vocoder)
    wav_buf = io.BytesIO()
    sf.write(wav_buf, speech.numpy(), samplerate=16_000, format="WAV")
    wav_buf.seek(0)
    mp3_buf = io.BytesIO()
    AudioSegment.from_wav(wav_buf).export(mp3_buf, format="mp3", bitrate="64k")
    return mp3_buf.getvalue()


# ── MMS-TTS (multilingual VITS) ──────────────────────────────────────────────
_mms_cache: dict[str, tuple] = {}   # mms_code → (VitsModel, tokenizer)


def _get_mms(mms_code: str):
    if mms_code not in _mms_cache:
        model_id = f"facebook/mms-tts-{mms_code}"
        logger.info(f"Loading MMS-TTS: {model_id}")
        tok   = AutoTokenizer.from_pretrained(model_id)
        model = VitsModel.from_pretrained(model_id)
        model.eval()
        _mms_cache[mms_code] = (model, tok)
        logger.info(f"MMS-TTS [{mms_code}] ready.")
    return _mms_cache[mms_code]


def _mms_tts(text: str, mms_code: str) -> bytes:
    """Multilingual TTS — MMS-TTS VITS → MP3 bytes."""
    model, tok = _get_mms(mms_code)
    inputs = tok(text, return_tensors="pt")
    with torch.no_grad():
        waveform = model(**inputs).waveform.squeeze().cpu().numpy()
    sr = model.config.sampling_rate   # 16 000 or 22 050 depending on language
    wav_buf = io.BytesIO()
    sf.write(wav_buf, waveform, samplerate=sr, format="WAV")
    wav_buf.seek(0)
    mp3_buf = io.BytesIO()
    AudioSegment.from_wav(wav_buf).export(mp3_buf, format="mp3", bitrate="64k")
    return mp3_buf.getvalue()


def text_to_speech_bytes(text: str, lang: str) -> bytes:
    """
    Convert text → MP3 bytes.

    Routing:
      English → SpeechT5      (clearest quality for en)
      Other   → MMS-TTS       (1100+ languages, same HF ecosystem)
      Fallback → translate to English, then SpeechT5
    """
    if not text or not text.strip():
        return b""

    lang_lower = lang.lower()

    # ── English path ─────────────────────────────────────────────────────────
    if lang_lower.startswith("en"):
        try:
            logger.info("TTS: SpeechT5 (English)")
            return _speecht5_tts(text)
        except Exception as e:
            logger.error(f"SpeechT5 error: {e}")
            return b""

    # ── Multilingual path ────────────────────────────────────────────────────
    mms_code = MMS_LANG_MAP.get(lang_lower) or MMS_LANG_MAP.get(lang_lower.split("-")[0])
    if mms_code:
        try:
            logger.info(f"TTS: MMS-TTS [{lang} → mms-tts-{mms_code}]")
            return _mms_tts(text, mms_code)
        except Exception as e:
            logger.warning(f"MMS-TTS [{mms_code}] failed: {e} — falling back to SpeechT5")

    # ── Fallback: translate → English → SpeechT5 ────────────────────────────
    try:
        en_text = translate_to_english(text, lang)
        logger.info(f"TTS fallback: SpeechT5 on English translation [{lang}→en]")
        return _speecht5_tts(en_text)
    except Exception as e:
        logger.error(f"TTS fallback error: {e}")
        return b""


# ─────────────────────────────────────────────────────────────────────────────
# SESSION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_user_language(sender_id: str) -> str | None:
    return user_languages.get(sender_id)


def set_user_language(sender_id: str, lang: str) -> None:
    user_languages[sender_id] = lang


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO DOWNLOAD  (Messenger voice messages)
# ─────────────────────────────────────────────────────────────────────────────

async def download_audio(url: str, access_token: str) -> bytes:
    async with httpx.AsyncClient(follow_redirects=True) as client:
        resp = await client.get(url, headers={"Authorization": f"Bearer {access_token}"})
        return resp.content


# ─────────────────────────────────────────────────────────────────────────────
# WHISPER STT  (unchanged — faster-whisper is already HF-native)
# ─────────────────────────────────────────────────────────────────────────────

_whisper_model: WhisperModel | None = None


def _get_whisper() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        logger.info("Loading Whisper 'small' model…")
        _whisper_model = WhisperModel(
            "small", device="cpu", compute_type="int8",
            download_root="/app/.cache/whisper",
        )
        logger.info("Whisper ready.")
    return _whisper_model


def _normalize_whisper_lang(code: str) -> str:
    _MAP = {"zh-cn": "zh", "zh-tw": "zh", "nb": "no", "iw": "he"}
    return _MAP.get(code.lower(), code.lower())


def preprocess_audio(audio_bytes: bytes) -> AudioSegment:
    """Denoise + normalise before STT (handles AC hum, crowd noise, etc.)."""
    try:
        audio = (
            AudioSegment.from_file(io.BytesIO(audio_bytes))
            .set_channels(1)
            .set_frame_rate(16_000)
            .set_sample_width(2)
            .normalize()
        )
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        cleaned = nr.reduce_noise(
            y=samples, sr=16_000, stationary=False,
            prop_decrease=0.85, n_fft=512, win_length=512, hop_length=128,
        )
        return AudioSegment(
            cleaned.astype(np.int16).tobytes(),
            frame_rate=16_000, sample_width=2, channels=1,
        ).normalize()
    except Exception as e:
        logger.warning(f"Audio preprocessing error (using raw): {e}")
        return (
            AudioSegment.from_file(io.BytesIO(audio_bytes))
            .set_channels(1)
            .set_frame_rate(16_000)
            .set_sample_width(2)
            .normalize()
        )


def speech_to_text(
    audio_bytes: bytes,
    lang_hint: str | None = None,
) -> tuple[str, str]:
    """
    Transcribe audio → (transcript, detected_lang_code).

    Two-pass strategy:
      Pass 1 — auto-detect language
      Pass 2 — re-run with pinned lang_hint when:
               (a) confidence < threshold, OR
               (b) hint is an Indian language  ← Whisper "small" frequently
                   mis-labels Indian audio as Russian / Portuguese etc.
    """
    # Languages where Whisper "small" is unreliable — always pin to hint if available
    _UNRELIABLE = {
        "hi", "te", "ta", "kn", "ml", "bn", "mr",
        "gu", "pa", "ur", "or", "ne", "si",
    }
    CONFIDENCE_THRESHOLD = 0.75
    tmp_path: str | None = None
    try:
        audio = preprocess_audio(audio_bytes)
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        audio.export(tmp_path, format="wav")

        model = _get_whisper()
        VAD   = dict(min_silence_duration_ms=500)

        # Pass 1 — auto-detect
        segs, info = model.transcribe(
            tmp_path, beam_size=5, language=None,
            vad_filter=True, vad_parameters=VAD,
        )
        text     = " ".join(s.text.strip() for s in segs).strip()
        detected = _normalize_whisper_lang(info.language)
        prob     = info.language_probability
        logger.info(f"Whisper auto: lang={detected} prob={prob:.2f} text='{text}'")

        # Pass 2 — re-run with hint when:
        #   (a) low confidence (original logic), OR
        #   (b) hint is an Indian language and Whisper detected something different
        #       (Whisper "small" mis-labels Indian audio as ru/pt/etc. with high confidence)
        if (
            lang_hint
            and lang_hint != "en"
            and detected != lang_hint
            and (prob < CONFIDENCE_THRESHOLD or lang_hint in _UNRELIABLE)
        ):
            logger.info(f"Low confidence ({prob:.2f}) — retrying with lang='{lang_hint}'")
            segs2, info2 = model.transcribe(
                tmp_path, beam_size=5, language=lang_hint,
                vad_filter=True, vad_parameters=VAD,
            )
            text2 = " ".join(s.text.strip() for s in segs2).strip()
            logger.info(
                f"Whisper pinned={lang_hint}: prob={info2.language_probability:.2f} text='{text2}'"
            )
            if text2:
                return text2, lang_hint

        return text, detected

    except Exception as e:
        logger.error(f"STT error: {e}")
        return "", lang_hint or "en"

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)