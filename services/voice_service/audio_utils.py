"""
services/voice_service/audio_utils.py
----------------------------------------
Audio pre-processing helpers shared by STT and TTS modules.
"""

import io
import logging

log = logging.getLogger(__name__)


def convert_to_wav(audio_bytes: bytes, src_format: str = "ogg") -> bytes:
    """
    Converts audio to 16-bit PCM WAV at 16 000 Hz mono.
    Uses pydub + ffmpeg.  Returns raw WAV bytes.
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        raise RuntimeError("pydub is required: pip install pydub")

    buf = io.BytesIO(audio_bytes)
    seg = AudioSegment.from_file(buf, format=src_format)
    seg = seg.set_channels(1).set_frame_rate(16000).set_sample_width(2)

    out = io.BytesIO()
    seg.export(out, format="wav")
    return out.getvalue()


def denoise(audio_bytes: bytes, sample_rate: int = 16000) -> bytes:
    """
    Applies noisereduce to WAV bytes.
    Returns denoised WAV bytes.
    """
    try:
        import numpy as np
        import noisereduce as nr
        import scipy.io.wavfile as wav_io
    except ImportError:
        log.debug("noisereduce not installed; skipping denoising")
        return audio_bytes

    buf        = io.BytesIO(audio_bytes)
    sr, data   = wav_io.read(buf)
    reduced    = nr.reduce_noise(y=data.astype(float), sr=sr)
    out        = io.BytesIO()
    wav_io.write(out, sr, reduced.astype(np.int16))
    return out.getvalue()


def audio_bytes_to_base64(audio_bytes: bytes) -> str:
    import base64
    return base64.b64encode(audio_bytes).decode()
