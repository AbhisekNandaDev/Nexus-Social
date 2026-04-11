"""Audio extraction and transcription pipeline using faster-whisper."""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import ClassVar, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TranscriptionSegment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    full_text: str
    segments: List[TranscriptionSegment]
    language: str
    has_audio: bool
    duration: float = 0.0


class AudioPipeline:
    _model: ClassVar = None
    _model_params: ClassVar[tuple] = ("", "", "")

    @classmethod
    def get_model(cls, model_size: str = "base", device: str = "cpu",
                  compute_type: str = "int8"):
        params = (model_size, device, compute_type)
        if cls._model is None or cls._model_params != params:
            from faster_whisper import WhisperModel  # type: ignore
            logger.info("Loading WhisperModel | size=%s device=%s compute=%s",
                        model_size, device, compute_type)
            cls._model = WhisperModel(model_size, device=device, compute_type=compute_type)
            cls._model_params = params
            logger.info("WhisperModel loaded")
        return cls._model

    def _has_audio_stream(self, video_path: str) -> bool:
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "a",
                 "-show_entries", "stream=codec_type", "-of", "csv=p=0", video_path],
                capture_output=True, text=True, timeout=30,
            )
            return bool(result.stdout.strip())
        except (subprocess.SubprocessError, FileNotFoundError) as exc:
            logger.warning("ffprobe failed | error=%s — assuming audio present", exc)
            return True

    def extract_audio(self, video_path: str) -> Optional[str]:
        """Extract 16kHz mono WAV. Returns path, or None if no audio stream."""
        if not self._has_audio_stream(video_path):
            logger.info("No audio stream | path=%s", video_path)
            return None

        out_path = tempfile.mktemp(suffix=".wav")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", video_path, "-vn",
                 "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", out_path],
                capture_output=True, check=True, timeout=120,
            )
            logger.debug("Audio extracted | wav=%s", out_path)
            return out_path
        except subprocess.CalledProcessError as exc:
            logger.warning("ffmpeg failed | stderr=%s",
                           exc.stderr.decode(errors="replace")[:500])
            return None
        except subprocess.TimeoutExpired:
            logger.warning("ffmpeg timed out")
            return None

    def transcribe(
        self,
        video_path: str,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        wav_path: Optional[str] = None
        try:
            wav_path = self.extract_audio(video_path)
            if wav_path is None:
                return TranscriptionResult(full_text="", segments=[], language="",
                                           has_audio=False, duration=0.0)

            model = self.get_model(model_size, device, compute_type)
            logger.info("Transcribing | wav=%s language=%s", wav_path, language or "auto")

            segments_iter, info = model.transcribe(
                wav_path, beam_size=5, language=language, vad_filter=True,
            )

            segments: List[TranscriptionSegment] = []
            for seg in segments_iter:
                segments.append(TranscriptionSegment(
                    start=seg.start, end=seg.end, text=seg.text.strip(),
                ))

            full_text = " ".join(s.text for s in segments).strip()
            duration = float(getattr(info, "duration", 0.0))

            logger.info("Transcription done | lang=%s prob=%.2f chars=%d dur=%.1fs",
                        info.language, info.language_probability, len(full_text), duration)

            return TranscriptionResult(
                full_text=full_text,
                segments=segments,
                language=info.language,
                has_audio=True,
                duration=duration,
            )
        except Exception as exc:
            logger.warning("Transcription failed | error=%s", exc)
            return TranscriptionResult(full_text="", segments=[], language="",
                                       has_audio=True, duration=0.0)
        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass
