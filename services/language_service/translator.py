"""
services/language_service/translator.py
-----------------------------------------
Translation layer using facebook/nllb-200-distilled-600M.
Caches translations in Redis.
"""

from __future__ import annotations

import hashlib
import logging
import os

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "")
_redis = None


def _get_redis():
    global _redis
    if _redis is None and REDIS_URL:
        try:
            import redis as rlib
            _redis = rlib.from_url(REDIS_URL, decode_responses=True)
        except Exception:
            pass
    return _redis


_NLLB_LANG_MAP: dict[str, str] = {
    "en": "eng_Latn", "hi": "hin_Deva", "ar": "arb_Arab", "fr": "fra_Latn",
    "de": "deu_Latn", "es": "spa_Latn", "pt": "por_Latn", "it": "ita_Latn",
    "ru": "rus_Cyrl", "zh": "zho_Hans", "ja": "jpn_Jpan", "ko": "kor_Hang",
    "id": "ind_Latn", "tr": "tur_Latn", "th": "tha_Thai", "vi": "vie_Latn",
    "bn": "ben_Beng", "ur": "urd_Arab", "te": "tel_Telu", "ta": "tam_Taml",
    "ml": "mal_Mlym", "kn": "kan_Knda", "gu": "guj_Gujr", "pa": "pan_Guru",
    "mr": "mar_Deva", "sw": "swh_Latn", "ne": "npi_Deva", "si": "sin_Sinh",
    "pl": "pol_Latn", "nl": "nld_Latn", "sv": "swe_Latn", "fi": "fin_Latn",
    "da": "dan_Latn", "no": "nob_Latn", "cs": "ces_Latn", "ro": "ron_Latn",
    "hu": "hun_Latn", "el": "ell_Grek", "he": "heb_Hebr", "fa": "pes_Arab",
    "fil": "fil_Latn", "ms": "zsm_Latn", "km": "khm_Khmr", "my": "mya_Mymr",
    "ka": "kat_Geor", "hy": "hye_Armn", "am": "amh_Ethi",
}

_tokenizer = None
_model = None


def _get_model():
    global _tokenizer, _model
    if _model is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        mid = "facebook/nllb-200-distilled-600M"
        _tokenizer = AutoTokenizer.from_pretrained(mid)
        _model = AutoModelForSeq2SeqLM.from_pretrained(mid)
        _model.eval()
    return _tokenizer, _model


def _nllb_code(lang: str) -> str:
    code = lang.lower()
    return _NLLB_LANG_MAP.get(code) or _NLLB_LANG_MAP.get(code.split("-")[0], "eng_Latn")


def _cache_key(src: str, tgt: str, text: str) -> str:
    h = hashlib.md5(text.encode()).hexdigest()[:12]
    return f"trans:{src}:{tgt}:{h}"


def translate(text: str, src: str, tgt: str) -> str:
    """Translate text from src to tgt language (both ISO 639-1)."""
    if not text.strip() or src == tgt:
        return text

    # Check Redis cache first
    r = _get_redis()
    ck = _cache_key(src, tgt, text)
    if r:
        try:
            cached = r.get(ck)
            if cached:
                return cached
        except Exception:
            pass

    try:
        import torch
        tokenizer, model = _get_model()
        src_nllb = _nllb_code(src)
        tgt_nllb = _nllb_code(tgt)
        tokenizer.src_lang = src_nllb
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        tgt_id = tokenizer.convert_tokens_to_ids(tgt_nllb)
        with torch.no_grad():
            out = model.generate(**inputs, forced_bos_token_id=tgt_id,
                                 max_length=512, num_beams=4)
        result = tokenizer.decode(out[0], skip_special_tokens=True)

        if r:
            try:
                r.setex(ck, 86400, result)
            except Exception:
                pass
        return result
    except Exception as e:
        logger.error("NLLB translate %s→%s error: %s", src, tgt, e)
        return text


def to_english(text: str, src: str = "auto") -> str:
    """Translate any text → English."""
    if src in ("auto", "en") or not text.strip():
        return text
    return translate(text, src, "en")


def from_english(text: str, tgt: str) -> str:
    """Translate English text → target language."""
    if not text.strip() or tgt == "en":
        return text
    return translate(text, "en", tgt)
