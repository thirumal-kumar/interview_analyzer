# transcribe.py
import os
import subprocess
import tempfile
from faster_whisper import WhisperModel


def ensure_wav_16k(input_path, out_path=None):
    """
    Convert any audio file to 16 kHz mono WAV using ffmpeg CLI.
    SAFE for Python 3.13 (no pydub, no audioop).
    """
    if out_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        out_path = os.path.join(tempfile.gettempdir(), f"{base}_16k_mono.wav")

    cmd = [
        "ffmpeg",
        "-y",                  # overwrite existing output
        "-i", input_path,      # input file
        "-ar", "16000",        # 16 kHz sample rate
        "-ac", "1",            # mono
        "-f", "wav",
        out_path
    ]

    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return out_path


class Transcriber:
    def __init__(self, model_size="small", device="cpu", compute_type="int8"):
        """
        model_size: tiny, base, small, medium, large-v3 (recommended: small)
        compute_type: int8 → fastest on CPU
        """
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )

    def transcribe_file(
        self,
        input_audio_path,
        language="en",
        progress_callback=None,
        word_timestamps=False
    ):
        """
        Transcribe audio using Faster Whisper.
        Returns:
        {
            "segments": [{start, end, text}, ...],
            "text": "<full transcript>",
            "duration": float_seconds
        }
        """

        # Convert audio to WAV 16k mono
        wav_path = ensure_wav_16k(input_audio_path)

        # Faster-whisper returns: (segments_generator, info_object)
        segments, info = self.model.transcribe(
            wav_path,
            language=language,
            word_timestamps=word_timestamps
        )

        duration = info.duration  # Total audio length in seconds
        segments_list = []
        full_text = []

        # Iterate through segments
        for seg in segments:
            seg_data = {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text.strip()
            }
            segments_list.append(seg_data)
            full_text.append(seg.text.strip())

            # Update Streamlit progress bar if callback exists
            if progress_callback and duration and duration > 0:
                pct = min(1.0, seg.end / duration)
                progress_callback(pct)

        return {
            "segments": segments_list,
            "text": " ".join(full_text).strip(),
            "duration": duration
        }
