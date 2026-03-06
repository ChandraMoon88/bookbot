import io
import httpx
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator
from gtts import gTTS
import speech_recognition as sr
from pydub import AudioSegment

DetectorFactory.seed = 0  # consistent detection results

# In-memory store: sender_id -> detected language code
user_languages: dict = {}

# Map langdetect codes to Google Speech API locale codes
SPEECH_LOCALE_MAP = {
    "zh-cn": "zh-CN", "zh-tw": "zh-TW", "zh": "zh-CN",
    "pt": "pt-BR", "en": "en-US", "fr": "fr-FR",
    "de": "de-DE", "es": "es-ES", "hi": "hi-IN",
    "ta": "ta-IN", "te": "te-IN", "ml": "ml-IN",
    "ar": "ar-SA", "ja": "ja-JP", "ko": "ko-KR",
    "ru": "ru-RU", "it": "it-IT", "nl": "nl-NL",
    "tr": "tr-TR", "pl": "pl-PL", "uk": "uk-UA",
    "vi": "vi-VN", "th": "th-TH", "id": "id-ID",
    "ms": "ms-MY", "bn": "bn-BD", "ur": "ur-PK",
    "fa": "fa-IR", "he": "iw-IL", "sv": "sv-SE",
    "da": "da-DK", "fi": "fi-FI", "nb": "nb-NO",
    "el": "el-GR", "cs": "cs-CZ", "sk": "sk-SK",
    "hu": "hu-HU", "ro": "ro-RO", "bg": "bg-BG",
    "hr": "hr-HR", "sr": "sr-RS", "ca": "ca-ES",
    "af": "af-ZA", "sw": "sw-KE",
}


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "en"


def get_user_language(sender_id: str):
    return user_languages.get(sender_id)


def set_user_language(sender_id: str, lang: str):
    user_languages[sender_id] = lang


def get_speech_locale(lang_code: str) -> str:
    code = lang_code.lower()
    if code in SPEECH_LOCALE_MAP:
        return SPEECH_LOCALE_MAP[code]
    # fallback: "fr" -> "fr-FR"
    return f"{code}-{code.upper()}"


def translate_to(text: str, target_lang: str) -> str:
    if target_lang == "en":
        return text
    try:
        return GoogleTranslator(source="en", target=target_lang).translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        return text


def text_to_speech_bytes(text: str, lang: str) -> bytes:
    try:
        # gTTS uses 2-letter codes or locale codes
        gtts_lang = lang.split("-")[0].lower()
        tts = gTTS(text=text, lang=gtts_lang)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        print(f"TTS error: {e}")
        return b""


async def download_audio(url: str, access_token: str) -> bytes:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            url,
            headers={"Authorization": f"Bearer {access_token}"},
            follow_redirects=True
        )
        return response.content


def speech_to_text(audio_bytes: bytes, lang_code: str = "en") -> str:
    recognizer = sr.Recognizer()
    try:
        # pydub converts any format (m4a, mp4, ogg) to wav via ffmpeg
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)

        with sr.AudioFile(wav_io) as source:
            audio_data = recognizer.record(source)

        locale = get_speech_locale(lang_code)
        return recognizer.recognize_google(audio_data, language=locale)
    except Exception as e:
        print(f"STT error: {e}")
        return ""
