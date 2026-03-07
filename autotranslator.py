import io
import os
import asyncio
import tempfile
import httpx
import numpy as np
import noisereduce as nr
from langdetect import detect_langs, DetectorFactory
from deep_translator import GoogleTranslator
import edge_tts
from faster_whisper import WhisperModel
from pydub import AudioSegment

DetectorFactory.seed = 0

# Edge TTS voice map — 100+ languages
EDGE_TTS_VOICES = {
    # Indian Languages
    "en": "en-US-JennyNeural",
    "hi": "hi-IN-SwaraNeural",
    "te": "te-IN-ShrutiNeural",
    "ta": "ta-IN-PallaviNeural",
    "kn": "kn-IN-SapnaNeural",
    "ml": "ml-IN-SobhanaNeural",
    "bn": "bn-IN-TanishaaNeural",
    "mr": "mr-IN-AarohiNeural",
    "gu": "gu-IN-DhwaniNeural",
    "pa": "pa-IN-OjasveeNeural",
    "ur": "ur-PK-UzmaNeural",
    "or": "or-IN-SubhasiniNeural",

    # East Asian Languages
    "zh-CN": "zh-CN-XiaoxiaoNeural",
    "zh-cn": "zh-CN-XiaoxiaoNeural",
    "zh-TW": "zh-TW-HsiaoChenNeural",
    "zh-tw": "zh-TW-HsiaoChenNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "mn": "mn-MN-YesuiNeural",

    # Southeast Asian Languages
    "id": "id-ID-GadisNeural",
    "ms": "ms-MY-YasminNeural",
    "th": "th-TH-PremwadeeNeural",
    "vi": "vi-VN-HoaiMyNeural",
    "fil": "fil-PH-BlessicaNeural",
    "km": "km-KH-SreymomNeural",
    "lo": "lo-LA-KeomanyNeural",
    "my": "my-MM-NilarNeural",
    "jv": "jv-ID-SitiNeural",
    "su": "su-ID-TutiNeural",

    # European Languages
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "es": "es-ES-ElviraNeural",
    "pt": "pt-BR-FranciscaNeural",
    "pt-PT": "pt-PT-RaquelNeural",
    "it": "it-IT-ElsaNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "pl": "pl-PL-ZofiaNeural",
    "nl": "nl-NL-ColetteNeural",
    "sv": "sv-SE-SofieNeural",
    "da": "da-DK-ChristelNeural",
    "fi": "fi-FI-NooraNeural",
    "no": "nb-NO-PernilleNeural",
    "nb": "nb-NO-PernilleNeural",
    "cs": "cs-CZ-VlastaNeural",
    "sk": "sk-SK-ViktoriaNeural",
    "ro": "ro-RO-AlinaNeural",
    "hu": "hu-HU-NoemiNeural",
    "el": "el-GR-AthinaNeural",
    "bg": "bg-BG-KalinaNeural",
    "hr": "hr-HR-GabrijelaNeural",
    "uk": "uk-UA-PolinaNeural",
    "sr": "sr-RS-SophieNeural",
    "sl": "sl-SI-PetraNeural",
    "lt": "lt-LT-OnaNeural",
    "lv": "lv-LV-EveritaNeural",
    "et": "et-EE-AnuNeural",
    "ca": "ca-ES-JoanaNeural",
    "gl": "gl-ES-SabelaNeural",
    "eu": "eu-ES-AinhoaNeural",
    "mt": "mt-MT-GraceNeural",
    "cy": "cy-GB-NiaNeural",
    "ga": "ga-IE-OrlaNeural",
    "is": "is-IS-GudrunNeural",
    "mk": "mk-MK-MarijaNeural",
    "sq": "sq-AL-AnilaNeural",
    "bs": "bs-BA-VesnaNeural",
    "tr": "tr-TR-EmelNeural",
    "az": "az-AZ-BanuNeural",
    "kk": "kk-KZ-AigulNeural",
    "uz": "uz-UZ-MadinaNeural",
    "ky": "ky-KG-AynaNeural",
    "ka": "ka-GE-EkaNeural",
    "hy": "hy-AM-AnahitNeural",

    # Middle Eastern Languages
    "ar": "ar-SA-ZariyahNeural",
    "iw": "he-IL-HilaNeural",
    "he": "he-IL-HilaNeural",
    "fa": "fa-IR-DilaraNeural",
    "ps": "ps-AF-LatifaNeural",

    # African Languages
    "af": "af-ZA-AdriNeural",
    "sw": "sw-KE-ZuriNeural",
    "am": "am-ET-MekdesNeural",
    "so": "so-SO-UbaxNeural",
    "zu": "zu-ZA-ThandoNeural",
    "yo": "yo-NG-AdesuaNeural",
    "ig": "ig-NG-EzinneNeural",
    "st": "st-ZA-LeahNeural",
    "tn": "tn-ZA-MosakiNeural",
    "xh": "xh-ZA-NomvulaNeural",
    "nr": "nr-ZA-SibongileNeural",
    "ss": "ss-ZA-ThandiwéNeural",
    "ts": "ts-ZA-HluviNeural",
    "ve": "ve-ZA-SefatulaNeural",

    # Central Asian
    "tk": "tk-TM-AyşeNeural",
    "tt": "tt-RU-SadiNeural",

    # Others
    "si": "si-LK-ThiliniNeural",
    "ne": "ne-NP-HemkalaNeural",
    "wuu": "wuu-CN-XiaotongNeural",
    "yue": "yue-CN-XiaoMinNeural",
    "jw": "jv-ID-SitiNeural",
    "en-GB": "en-GB-SoniaNeural",
    "en-AU": "en-AU-NatashaNeural",
    "en-IN": "en-IN-NeerjaNeural",
    "fr-CA": "fr-CA-SylvieNeural",
    "es-MX": "es-MX-DaliaNeural",
    "de-AT": "de-AT-IngridNeural",
    "de-CH": "de-CH-LeniNeural",
    "fr-BE": "fr-BE-CharlineNeural",
    "fr-CH": "fr-CH-ArianeNeural",
    "nl-BE": "nl-BE-DenaNeural",
    "pt-BR": "pt-BR-FranciscaNeural",
    "ar-EG": "ar-EG-SalmaNeural",
    "ar-AE": "ar-AE-FatimaNeural",
    "zh-HK": "zh-HK-HiuGaaiNeural",
}

# In-memory store: sender_id -> detected language code
user_languages: dict = {}

# Dynamically fetch supported languages
try:
    _lang_dict = GoogleTranslator.get_supported_languages(as_dict=True)
    TRANSLATE_LANGS = set(_lang_dict.values())
    print(f"Translation language codes loaded: {len(TRANSLATE_LANGS)}")
except Exception:
    TRANSLATE_LANGS = {"en"}

# Normalize language codes to deep-translator compatible codes
LANG_NORMALIZE = {
    "zh-cn": "zh-CN",
    "zh-tw": "zh-TW",
    "he": "iw",
    "nb": "no",
}

def normalize_lang(code: str) -> str:
    """Normalize language codes to deep-translator compatible codes."""
    lower = code.lower()
    return LANG_NORMALIZE.get(lower, lower)

def get_edge_voice(lang_code: str) -> str:
    """Get Edge TTS voice for language, fallback to English."""
    code = lang_code.lower()
    if code in EDGE_TTS_VOICES:
        return EDGE_TTS_VOICES[code]
    base = code.split("-")[0]
    if base in EDGE_TTS_VOICES:
        return EDGE_TTS_VOICES[base]
    return "en-US-JennyNeural"


def text_to_speech_bytes(text: str, lang: str) -> bytes:
    """Convert text to speech using Edge TTS — no rate limits, 100+ languages."""
    voice = get_edge_voice(lang)
    try:
        # Edge TTS requires async — run in event loop
        audio_bytes = asyncio.run(_edge_tts_generate(text, voice))
        if audio_bytes:
            print(f"Edge TTS success [{lang}] voice={voice}")
            return audio_bytes
    except RuntimeError:
        # If event loop already running (inside async context)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            audio_bytes = loop.run_until_complete(
                _edge_tts_generate(text, voice)
            )
            return audio_bytes
        finally:
            loop.close()
    except Exception as e:
        print(f"Edge TTS error [{lang}]: {e}")
    return b""


async def _edge_tts_generate(text: str, voice: str) -> bytes:
    """Generate speech bytes using edge-tts."""
    communicate = edge_tts.Communicate(text, voice)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    buf.seek(0)
    return buf.read()

# Unicode script → language code mapping for reliable detection of non-Latin
# scripts in text messages (not used for voice — Whisper handles that).
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
    """Detect language from Unicode script ranges. Returns None for Latin/Cyrillic."""
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
    if best_lang and counts[best_lang] >= 1:
        return best_lang
    return None


def detect_language(text: str) -> str:
    """Detect language of a text string. Used for typed messages."""
    script_lang = detect_script_language(text)
    if script_lang:
        return normalize_lang(script_lang)
    try:
        langs = detect_langs(text)
        if langs and langs[0].prob >= 0.80:
            return normalize_lang(langs[0].lang)
        return "en"
    except Exception:
        return "en"


def translate_to_english(text: str, source_lang: str = "auto") -> str:
    """Translate any text to English for intent/greeting detection.
    Passing source_lang avoids transliteration and returns a real English word."""
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


async def download_audio(url: str, access_token: str) -> bytes:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            url,
            headers={"Authorization": f"Bearer {access_token}"},
            follow_redirects=True
        )
        return response.content


# ---------------------------------------------------------------------------
# Whisper STT — replaces the old Google Speech + multi-locale probing approach.
# Whisper detects the spoken language automatically in a single pass with no
# hardcoded locale lists, no confidence hacking, and no per-language priors.
# ---------------------------------------------------------------------------

_whisper_model: WhisperModel | None = None


def _get_whisper() -> WhisperModel:
    """Load the Whisper model once and reuse it for all requests."""
    global _whisper_model
    if _whisper_model is None:
        print("Loading Whisper 'small' model (first request only)…")
        # int8 quantisation: ~2× faster on CPU with no accuracy loss for STT
        _whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
        print("Whisper model ready.")
    return _whisper_model


def preprocess_audio(audio_bytes: bytes) -> AudioSegment:
    """
    Prepare audio for Whisper:
    - Convert to mono 16 kHz (Whisper's native sample rate)
    - Normalize loudness
    - Spectral noise reduction (handles background noise / AC hum / crowd)
    """
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        audio = audio.normalize()

        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        cleaned = nr.reduce_noise(
            y=samples,
            sr=16000,
            stationary=False,
            prop_decrease=0.85,
            n_fft=512,
            win_length=512,
            hop_length=128,
        )
        audio = AudioSegment(
            cleaned.astype(np.int16).tobytes(),
            frame_rate=16000,
            sample_width=2,
            channels=1,
        )
        return audio.normalize()

    except Exception as e:
        print(f"Audio preprocessing error (using raw): {e}")
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        return audio.set_channels(1).set_frame_rate(16000).set_sample_width(2).normalize()


def speech_to_text(audio_bytes: bytes) -> tuple[str, str]:
    """
    Transcribe audio with Whisper and return (transcript, detected_lang_code).

    Whisper detects the spoken language automatically — no locale lists,
    no multi-probe hacks, works for English, Telugu, Tamil, Hindi and
    99 other languages out of the box.
    """
    tmp_path = None
    try:
        audio = preprocess_audio(audio_bytes)

        # Write to a temp WAV file — Whisper reads from disk
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        audio.export(tmp_path, format="wav")

        model = _get_whisper()
        segments, info = model.transcribe(
            tmp_path,
            beam_size=5,
            language=None,       # auto-detect spoken language
            vad_filter=True,     # skip leading/trailing silence
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        text = " ".join(seg.text.strip() for seg in segments).strip()
        detected_lang = normalize_lang(info.language)  # e.g. "en", "te", "ta", "hi"

        print(
            f"Whisper STT: lang={detected_lang} "
            f"(prob={info.language_probability:.2f}), text='{text}'"
        )
        return text, detected_lang

    except Exception as e:
        print(f"STT error: {e}")
        return "", "en"

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
