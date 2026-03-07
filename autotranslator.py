import io
import httpx
import numpy as np
import noisereduce as nr
from langdetect import detect_langs, DetectorFactory
from deep_translator import GoogleTranslator
from gtts import gTTS
from gtts.lang import tts_langs
import speech_recognition as sr
from pydub import AudioSegment

DetectorFactory.seed = 0

# In-memory store: sender_id -> detected language code
user_languages: dict = {}

# Dynamically fetch all languages supported by deep-translator and gTTS at startup
try:
    _lang_dict = GoogleTranslator.get_supported_languages(as_dict=True)
    # _lang_dict maps name→code e.g. {"telugu": "te", "hindi": "hi", ...}
    TRANSLATE_LANGS = set(_lang_dict.values())  # set of codes: {"te", "hi", "en", ...}
    print(f"Translation language codes loaded: {len(TRANSLATE_LANGS)}")
except Exception:
    TRANSLATE_LANGS = {"en"}

try:
    TTS_LANGS = set(tts_langs().keys())
    print(f"TTS languages loaded: {len(TTS_LANGS)}")
except Exception:
    TTS_LANGS = {"en"}

# Complete BCP-47 locale map for ALL Google Translate languages
# Used for Google Speech Recognition (STT)
SPEECH_LOCALE_MAP = {
    "af": "af-ZA",       # Afrikaans
    "sq": "sq-AL",       # Albanian
    "am": "am-ET",       # Amharic
    "ar": "ar-SA",       # Arabic
    "hy": "hy-AM",       # Armenian
    "as": "as-IN",       # Assamese
    "ay": "ay-BO",       # Aymara
    "az": "az-AZ",       # Azerbaijani
    "bm": "bm-ML",       # Bambara
    "eu": "eu-ES",       # Basque
    "be": "be-BY",       # Belarusian
    "bn": "bn-BD",       # Bengali
    "bho": "bho-IN",     # Bhojpuri
    "bs": "bs-BA",       # Bosnian
    "bg": "bg-BG",       # Bulgarian
    "ca": "ca-ES",       # Catalan
    "ceb": "ceb-PH",     # Cebuano
    "ny": "ny-MW",       # Chichewa
    "zh-cn": "zh-CN",    # Chinese Simplified
    "zh-tw": "zh-TW",    # Chinese Traditional
    "zh": "zh-CN",       # Chinese (default to Simplified)
    "co": "co-FR",       # Corsican
    "hr": "hr-HR",       # Croatian
    "cs": "cs-CZ",       # Czech
    "da": "da-DK",       # Danish
    "dv": "dv-MV",       # Dhivehi
    "doi": "doi-IN",     # Dogri
    "nl": "nl-NL",       # Dutch
    "en": "en-US",       # English
    "eo": "eo",          # Esperanto
    "et": "et-EE",       # Estonian
    "ee": "ee-GH",       # Ewe
    "tl": "tl-PH",       # Filipino
    "fi": "fi-FI",       # Finnish
    "fr": "fr-FR",       # French
    "fy": "fy-NL",       # Frisian
    "gl": "gl-ES",       # Galician
    "ka": "ka-GE",       # Georgian
    "de": "de-DE",       # German
    "el": "el-GR",       # Greek
    "gn": "gn-PY",       # Guarani
    "gu": "gu-IN",       # Gujarati
    "ht": "ht-HT",       # Haitian Creole
    "ha": "ha-NG",       # Hausa
    "haw": "haw-US",     # Hawaiian
    "he": "iw-IL",       # Hebrew (langdetect→he, Google Speech uses iw)
    "iw": "iw-IL",       # Hebrew (Google code)
    "hi": "hi-IN",       # Hindi
    "hmn": "hmn-CN",     # Hmong
    "hu": "hu-HU",       # Hungarian
    "is": "is-IS",       # Icelandic
    "ig": "ig-NG",       # Igbo
    "ilo": "ilo-PH",     # Ilocano
    "id": "id-ID",       # Indonesian
    "ga": "ga-IE",       # Irish
    "it": "it-IT",       # Italian
    "ja": "ja-JP",       # Japanese
    "jv": "jv-ID",       # Javanese
    "kn": "kn-IN",       # Kannada
    "kk": "kk-KZ",       # Kazakh
    "km": "km-KH",       # Khmer
    "rw": "rw-RW",       # Kinyarwanda
    "gom": "gom-IN",     # Konkani
    "ko": "ko-KR",       # Korean
    "kri": "kri-SL",     # Krio
    "ku": "ku-TR",       # Kurdish (Kurmanji)
    "ckb": "ckb-IQ",     # Kurdish (Sorani)
    "ky": "ky-KG",       # Kyrgyz
    "lo": "lo-LA",       # Lao
    "la": "la-VA",       # Latin
    "lv": "lv-LV",       # Latvian
    "ln": "ln-CD",       # Lingala
    "lt": "lt-LT",       # Lithuanian
    "lg": "lg-UG",       # Luganda
    "lb": "lb-LU",       # Luxembourgish
    "mk": "mk-MK",       # Macedonian
    "mai": "mai-IN",     # Maithili
    "mg": "mg-MG",       # Malagasy
    "ms": "ms-MY",       # Malay
    "ml": "ml-IN",       # Malayalam
    "mt": "mt-MT",       # Maltese
    "mi": "mi-NZ",       # Maori
    "mr": "mr-IN",       # Marathi
    "mn": "mn-MN",       # Mongolian
    "my": "my-MM",       # Myanmar (Burmese)
    "ne": "ne-NP",       # Nepali
    "no": "no-NO",       # Norwegian
    "nb": "nb-NO",       # Norwegian Bokmål
    "or": "or-IN",       # Odia
    "om": "om-ET",       # Oromo
    "ps": "ps-AF",       # Pashto
    "fa": "fa-IR",       # Persian
    "pl": "pl-PL",       # Polish
    "pt": "pt-BR",       # Portuguese
    "pa": "pa-IN",       # Punjabi
    "qu": "qu-PE",       # Quechua
    "ro": "ro-RO",       # Romanian
    "ru": "ru-RU",       # Russian
    "sm": "sm-WS",       # Samoan
    "sa": "sa-IN",       # Sanskrit
    "gd": "gd-GB",       # Scots Gaelic
    "nso": "nso-ZA",     # Sepedi
    "sr": "sr-RS",       # Serbian
    "st": "st-ZA",       # Sesotho
    "sn": "sn-ZW",       # Shona
    "sd": "sd-PK",       # Sindhi
    "si": "si-LK",       # Sinhala
    "sk": "sk-SK",       # Slovak
    "sl": "sl-SI",       # Slovenian
    "so": "so-SO",       # Somali
    "es": "es-ES",       # Spanish
    "su": "su-ID",       # Sundanese
    "sw": "sw-KE",       # Swahili
    "sv": "sv-SE",       # Swedish
    "tg": "tg-TJ",       # Tajik
    "ta": "ta-IN",       # Tamil
    "tt": "tt-RU",       # Tatar
    "te": "te-IN",       # Telugu
    "th": "th-TH",       # Thai
    "ti": "ti-ET",       # Tigrinya
    "ts": "ts-ZA",       # Tsonga
    "tr": "tr-TR",       # Turkish
    "tk": "tk-TM",       # Turkmen
    "ak": "ak-GH",       # Twi
    "uk": "uk-UA",       # Ukrainian
    "ur": "ur-PK",       # Urdu
    "ug": "ug-CN",       # Uyghur
    "uz": "uz-UZ",       # Uzbek
    "vi": "vi-VN",       # Vietnamese
    "cy": "cy-GB",       # Welsh
    "xh": "xh-ZA",       # Xhosa
    "yi": "yi-001",      # Yiddish
    "yo": "yo-NG",       # Yoruba
    "zu": "zu-ZA",       # Zulu
}

# Normalize langdetect codes to deep-translator compatible codes
LANG_NORMALIZE = {
    "zh-cn": "zh-CN",
    "zh-tw": "zh-TW",
    "he": "iw",   # deep-translator uses 'iw' for Hebrew
    "nb": "no",   # Norwegian Bokmål → Norwegian
}


def normalize_lang(code: str) -> str:
    """Normalize langdetect output to a deep-translator compatible code."""
    lower = code.lower()
    return LANG_NORMALIZE.get(lower, lower)

def get_gtts_lang(lang_code: str) -> str:
    """Return a valid gTTS language code, falling back to 'en'."""
    code = lang_code.lower()
    if code in TTS_LANGS:
        return code
    base = code.split("-")[0]
    if base in TTS_LANGS:
        return base
    return "en"


def get_speech_locale(lang_code: str) -> str:
    """Return a BCP-47 locale for Google Speech Recognition."""
    code = lang_code.lower()
    if code in SPEECH_LOCALE_MAP:
        return SPEECH_LOCALE_MAP[code]
    base = code.split("-")[0]
    if base in SPEECH_LOCALE_MAP:
        return SPEECH_LOCALE_MAP[base]
    return "en-US"


# Unicode script → language code mapping for reliable detection
# of non-Latin scripts even on very short text
UNICODE_SCRIPT_MAP = [
    (0x0900, 0x097F, "hi"),    # Devanagari → Hindi
    (0x0980, 0x09FF, "bn"),    # Bengali
    (0x0A00, 0x0A7F, "pa"),    # Gurmukhi → Punjabi
    (0x0A80, 0x0AFF, "gu"),    # Gujarati
    (0x0B00, 0x0B7F, "or"),    # Odia
    (0x0B80, 0x0BFF, "ta"),    # Tamil
    (0x0C00, 0x0C7F, "te"),    # Telugu
    (0x0C80, 0x0CFF, "kn"),    # Kannada
    (0x0D00, 0x0D7F, "ml"),    # Malayalam
    (0x0D80, 0x0DFF, "si"),    # Sinhala
    (0x0E00, 0x0E7F, "th"),    # Thai
    (0x0E80, 0x0EFF, "lo"),    # Lao
    (0x0F00, 0x0FFF, "bo"),    # Tibetan
    (0x1000, 0x109F, "my"),    # Myanmar/Burmese
    (0x10A0, 0x10FF, "ka"),    # Georgian
    (0x1200, 0x137F, "am"),    # Ethiopic → Amharic
    (0x1780, 0x17FF, "km"),    # Khmer
    (0x1800, 0x18AF, "mn"),    # Mongolian
    (0x0530, 0x058F, "hy"),    # Armenian
    (0x0590, 0x05FF, "iw"),    # Hebrew
    (0x0600, 0x06FF, "ar"),    # Arabic
    (0x0750, 0x077F, "ar"),    # Arabic Supplement
    (0x3040, 0x309F, "ja"),    # Hiragana → Japanese
    (0x30A0, 0x30FF, "ja"),    # Katakana → Japanese
    (0x4E00, 0x9FFF, "zh-CN"), # CJK Unified Ideographs → Chinese
    (0xAC00, 0xD7AF, "ko"),    # Hangul → Korean
    (0x0400, 0x04FF, None),    # Cyrillic (needs langdetect: ru/uk/bg/sr...)
]


def detect_script_language(text: str):
    """Detect language based on Unicode script ranges. Returns None for Latin/Cyrillic."""
    counts = {}
    for ch in text:
        cp = ord(ch)
        for start, end, lang in UNICODE_SCRIPT_MAP:
            if start <= cp <= end:
                counts[lang] = counts.get(lang, 0) + 1
                break

    if not counts:
        return None

    best_lang = max(counts, key=counts.__getitem__)
    # Need at least 1 character from the script to be confident
    if best_lang and counts[best_lang] >= 1:
        return best_lang
    return None


def detect_language(text: str) -> str:
    # Unicode script detection first — handles all non-Latin scripts reliably
    script_lang = detect_script_language(text)
    if script_lang:
        return normalize_lang(script_lang)

    # For Latin/Cyrillic: use langdetect with confidence threshold
    # Short words like "hello" have low confidence and should stay as "en"
    try:
        langs = detect_langs(text)
        if langs and langs[0].prob >= 0.80:
            return normalize_lang(langs[0].lang)
        # Low confidence = ambiguous short word → default to English
        return "en"
    except Exception:
        return "en"


def translate_to_english(text: str, source_lang: str = "auto") -> str:
    """Translate any text to English for intent/greeting detection.
    Passing source_lang (e.g. 'te', 'ta', 'hi') avoids transliteration and
    forces Google Translate to return an actual English equivalent."""
    try:
        src = source_lang if source_lang and source_lang != "en" else "auto"
        return GoogleTranslator(source=src, target="en").translate(text)
    except Exception:
        try:
            return GoogleTranslator(source="auto", target="en").translate(text)
        except Exception:
            return text


def get_user_language(sender_id: str):
    return user_languages.get(sender_id)


def set_user_language(sender_id: str, lang: str):
    user_languages[sender_id] = lang


def translate_to(text: str, target_lang: str) -> str:
    if target_lang == "en":
        return text
    try:
        return GoogleTranslator(source="en", target=target_lang).translate(text)
    except Exception as e:
        print(f"Translation error [{target_lang}]: {e}")
        return text


def text_to_speech_bytes(text: str, lang: str) -> bytes:
    gtts_lang = get_gtts_lang(lang)
    try:
        tts = gTTS(text=text, lang=gtts_lang)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        print(f"TTS error [{gtts_lang}]: {e}")
        return b""


async def download_audio(url: str, access_token: str) -> bytes:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            url,
            headers={"Authorization": f"Bearer {access_token}"},
            follow_redirects=True
        )
        return response.content


# Locales to probe when user language is not yet known.
# Ordered by global usage — early exit fires when confidence >= 0.85.
PROBE_LOCALES = [
    "en-US", "hi-IN", "te-IN", "ta-IN", "kn-IN", "ml-IN",
    "bn-IN", "mr-IN", "gu-IN", "pa-IN", "ur-PK",
    "ar-SA", "fr-FR", "es-ES", "de-DE", "pt-BR",
    "ru-RU", "ja-JP", "ko-KR", "zh-CN",
]

# Top 8 locales probed without early exit so the globally best locale wins.
# Keeps the probe count small (≈ 8 API calls) while covering all major
# Indian languages + English.
TOP_LOCALES = [
    "en-US", "hi-IN", "te-IN", "ta-IN", "kn-IN", "ml-IN",
    "bn-IN", "mr-IN",
]

# Short English words that are unambiguous regardless of what Indian-language
# STT returns (e.g. "hello" → Tamil "ஹலோ").  When en-US returns one of
# these, it overrides a non-English winner.
ENGLISH_TRIGGER_WORDS = {
    "hi", "hello", "hey", "help", "book", "start", "yes", "no",
    "cancel", "back", "ok", "okay", "stop", "menu",
}


def _try_locales(audio_data, locales: list, early_exit: bool = True) -> tuple:
    """Try each locale with show_all, return (best_text, best_confidence, best_locale).

    When early_exit=False all locales are tried so the globally best match
    wins.  Use early_exit=True only when a quick best-effort result is enough.
    """
    recognizer = sr.Recognizer()
    best_text, best_conf, best_locale = "", -1.0, ""
    for locale in locales:
        try:
            raw = recognizer.recognize_google(audio_data, language=locale, show_all=True)
            if not raw:
                continue
            if isinstance(raw, dict):
                alts = raw.get("alternative", [])
                if alts:
                    text = alts[0].get("transcript", "")
                    # Default 0.88: when Google omits the confidence field the
                    # result is still a solid match — treat it as 0.88 rather
                    # than 0.5 so it can beat genuinely lower-confidence hits
                    # from other locales, while still being beatable by any
                    # locale that returns an explicit confidence > 0.88.
                    conf = float(alts[0].get("confidence", 0.88))
                    if text and conf > best_conf:
                        best_conf, best_text, best_locale = conf, text, locale
                    if early_exit and best_conf >= 0.85:
                        break
            elif isinstance(raw, str) and raw:
                best_text, best_locale = raw, locale
                best_conf = 0.7
                if early_exit:
                    break
        except sr.UnknownValueError:
            continue
        except Exception as e:
            print(f"STT locale [{locale}] error: {e}")
            continue
    return best_text, best_conf, best_locale


def preprocess_audio(audio_bytes: bytes) -> AudioSegment:
    """
    Clean audio before STT:
    - Convert to mono 16 kHz (optimal for speech recognition)
    - Normalize loudness so soft/loud voices both work
    - Spectral noise reduction to filter background noise,
      wind, AC hum, crowd sounds, etc.
    """
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        # Mono + 16 kHz is the standard for all STT engines
        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        # Normalize loudness
        audio = audio.normalize()

        # Convert to float32 numpy array for noise reduction
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

        # Non-stationary noise reduction — handles crowd, wind, AC, traffic
        cleaned = nr.reduce_noise(
            y=samples,
            sr=16000,
            stationary=False,    # adapts to changing background noise
            prop_decrease=0.85,  # remove 85 % of detected noise
            n_fft=512,
            win_length=512,
            hop_length=128,
        )

        # Rebuild AudioSegment from cleaned samples
        audio = AudioSegment(
            cleaned.astype(np.int16).tobytes(),
            frame_rate=16000,
            sample_width=2,
            channels=1,
        )
        # Final normalize after noise reduction
        audio = audio.normalize()
        return audio

    except Exception as e:
        print(f"Audio preprocessing error (using raw): {e}")
        # Fallback: basic conversion without noise reduction
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        return audio.set_channels(1).set_frame_rate(16000).set_sample_width(2).normalize()


def speech_to_text(audio_bytes: bytes, lang_code: str = "en") -> str:
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 150        # catch very soft speech
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 1.5         # don't cut off slow speakers

    try:
        # Step 1 — clean the audio (denoise + normalize + mono + 16kHz)
        audio = preprocess_audio(audio_bytes)

        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)

        with sr.AudioFile(wav_io) as source:
            audio_data = recognizer.record(source)

        # Step 2 — probe TOP_LOCALES WITHOUT early exit so every locale is
        # compared and the one with the globally highest confidence wins.
        # Putting the user's known locale first maximises its chance when
        # confidence values are tied at the default (0.88).
        if lang_code and lang_code != "en":
            known_locale = get_speech_locale(lang_code)
            locales = [known_locale] + [l for l in TOP_LOCALES if l != known_locale]
        else:
            locales = TOP_LOCALES

        text, conf, locale = _try_locales(audio_data, locales, early_exit=False)

        # For languages outside TOP_LOCALES (e.g. French, Arabic) fall back
        # to the full PROBE_LOCALES with early exit if TOP_LOCALES found nothing.
        if not text:
            remaining = [l for l in PROBE_LOCALES if l not in locales]
            if remaining:
                text, conf, locale = _try_locales(audio_data, remaining, early_exit=True)

        # Step 3 — English trigger-word bias.
        # Short common English words (hello, help, book …) can score higher
        # confidence on Indian-language STT because they are loan words.
        # If en-US returns one of these known English words, always prefer it.
        if locale and not locale.startswith("en"):
            en_text, en_conf, _ = _try_locales(audio_data, ["en-US"], early_exit=False)
            if en_text and en_text.strip().lower() in ENGLISH_TRIGGER_WORDS:
                text, conf, locale = en_text, en_conf, "en-US"
                print(f"STT English trigger applied: conf={en_conf:.2f}, text='{en_text}'")

        print(f"STT best: locale={locale}, conf={conf:.2f}, text='{text}'")
        return text

    except Exception as e:
        print(f"STT error: {e}")
        return ""


