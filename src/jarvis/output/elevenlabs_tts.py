"""
ElevenLabs TTS implementation for Jarvis.

Provides cloud-based Text-to-Speech using ElevenLabs API with Polish
multilingual support. Audio is streamed back as PCM (16kHz, 16-bit, mono)
so it can be played directly with sounddevice without ffmpeg conversion.

Configuration (config.json):
    "tts_engine": "elevenlabs",
    "tts_elevenlabs_api_key": "sk_...",
    "tts_elevenlabs_voice_id": "pNInz6obpgDQGcFmaJgB",  # Adam
    "tts_elevenlabs_model_id": "eleven_multilingual_v2",
    "tts_elevenlabs_stability": 0.5,
    "tts_elevenlabs_similarity_boost": 0.75,
    "tts_elevenlabs_style": 0.0,
    "tts_elevenlabs_speaker_boost": true

Created for the Kawior (Polish Jarvis) project by Damian Kleszcz.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from typing import Callable, Optional

import numpy as np
import requests
import sounddevice as sd

from ..debug import debug_log
from ..utils.audio_lock import portaudio_lock


ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1"
ELEVENLABS_DEFAULT_VOICE_ID = "pNInz6obpgDQGcFmaJgB"  # Adam (English male, multilingual)
ELEVENLABS_DEFAULT_MODEL_ID = "eleven_multilingual_v2"
ELEVENLABS_OUTPUT_FORMAT = "pcm_16000"  # 16kHz mono PCM16


class ElevenLabsTTS:
    """Cloud-based TTS using ElevenLabs API.

    Streams audio from ElevenLabs as raw PCM (16kHz, mono, 16-bit)
    so it can be played through sounddevice with no extra dependencies.
    """

    def __init__(
        self,
        enabled: bool = True,
        voice: Optional[str] = None,
        rate: Optional[int] = None,
        api_key: Optional[str] = None,
        voice_id: Optional[str] = None,
        model_id: Optional[str] = None,
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        style: float = 0.0,
        speaker_boost: bool = True,
    ) -> None:
        self.enabled = enabled
        self.voice = voice
        self.rate = rate
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY", "")
        self.voice_id = voice_id or ELEVENLABS_DEFAULT_VOICE_ID
        self.model_id = model_id or ELEVENLABS_DEFAULT_MODEL_ID
        self.stability = stability
        self.similarity_boost = similarity_boost
        self.style = style
        self.speaker_boost = speaker_boost

        self._q: "queue.Queue[str]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._is_speaking = threading.Event()
        self._last_spoken_text: str = ""
        self._completion_callback: Optional[Callable[[], None]] = None
        self._duration_callback: Optional[Callable[[float], None]] = None
        self._should_interrupt = threading.Event()

        self._audio_stream: Optional[sd.OutputStream] = None
        self._audio_lock = threading.Lock()
        self._sample_rate: int = 16000

    def _have_credentials(self) -> bool:
        return bool(self.api_key) and bool(self.voice_id)

    def _synthesize_pcm(self, text: str) -> Optional[bytes]:
        """Call ElevenLabs API and return raw PCM16 bytes (mono, 16kHz)."""
        if not self._have_credentials():
            debug_log("ElevenLabs TTS: missing api_key or voice_id", "tts")
            return None

        url = f"{ELEVENLABS_API_BASE}/text-to-speech/{self.voice_id}"
        params = {"output_format": ELEVENLABS_OUTPUT_FORMAT}
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "audio/pcm",
        }
        body = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": {
                "stability": self.stability,
                "similarity_boost": self.similarity_boost,
                "style": self.style,
                "use_speaker_boost": self.speaker_boost,
            },
        }

        try:
            debug_log(f"ElevenLabs TTS: requesting {len(text)} chars", "tts")
            resp = requests.post(url, params=params, headers=headers, json=body, timeout=30)
            if resp.status_code != 200:
                debug_log(f"ElevenLabs TTS API error {resp.status_code}: {resp.text[:200]}", "tts")
                print(f"  ElevenLabs TTS API error {resp.status_code}: {resp.text[:200]}", flush=True)
                return None
            debug_log(f"ElevenLabs TTS: received {len(resp.content)} PCM bytes", "tts")
            return resp.content
        except requests.RequestException as e:
            debug_log(f"ElevenLabs TTS network error: {e}", "tts")
            print(f"  ElevenLabs TTS network error: {e}", flush=True)
            return None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="ElevenLabsTTS")
        self._thread.start()
        debug_log("ElevenLabsTTS worker thread started", "tts")

    def stop(self) -> None:
        self._stop.set()
        self._should_interrupt.set()
        try:
            self._q.put_nowait("")
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._close_audio_stream()
        debug_log("ElevenLabsTTS worker thread stopped", "tts")

    def _close_audio_stream(self) -> None:
        with self._audio_lock:
            if self._audio_stream is not None:
                try:
                    with portaudio_lock:
                        self._audio_stream.abort()
                        self._audio_stream.close()
                except Exception:
                    pass
                self._audio_stream = None

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                text = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if not text or self._stop.is_set():
                continue
            self._speak_one(text)

    def _speak_one(self, text: str) -> None:
        if self._stop.is_set():
            return

        self._should_interrupt.clear()
        self._is_speaking.set()
        self._last_spoken_text = text
        self._notify_speaking_state(True)
        interrupted = False
        start_time = time.time()

        try:
            pcm = self._synthesize_pcm(text)
            if pcm is None or len(pcm) == 0:
                debug_log("ElevenLabs TTS: empty PCM, skipping playback", "tts")
                return

            if self._should_interrupt.is_set():
                debug_log("ElevenLabs TTS interrupted before playback", "tts")
                return

            audio = np.frombuffer(pcm, dtype=np.int16)
            exact_duration = len(audio) / self._sample_rate
            debug_log(f"ElevenLabs TTS: {exact_duration:.2f}s, {len(audio)} samples", "tts")

            if self._duration_callback is not None:
                try:
                    self._duration_callback(exact_duration)
                except Exception as e:
                    debug_log(f"ElevenLabs TTS duration callback error: {e}", "tts")

            play_position = [0]
            blocksize = 1024

            def audio_callback(outdata, frames, time_info, status):
                if self._should_interrupt.is_set():
                    raise sd.CallbackAbort()
                start = play_position[0]
                end = start + frames
                chunk = audio[start:end]
                if len(chunk) < frames:
                    outdata[:len(chunk), 0] = chunk
                    outdata[len(chunk):, 0] = 0
                    raise sd.CallbackStop()
                else:
                    outdata[:, 0] = chunk
                play_position[0] = end

            with self._audio_lock:
                with portaudio_lock:
                    self._audio_stream = sd.OutputStream(
                        samplerate=self._sample_rate,
                        channels=1,
                        dtype="int16",
                        blocksize=blocksize,
                        callback=audio_callback,
                    )
                    self._audio_stream.start()

            try:
                while self._audio_stream is not None and self._audio_stream.active:
                    if self._should_interrupt.is_set():
                        interrupted = True
                        self._close_audio_stream()
                        break
                    time.sleep(0.05)
            finally:
                self._close_audio_stream()

            actual_duration = time.time() - start_time
            debug_log(f"ElevenLabs TTS complete: actual={actual_duration:.2f}s (audio={exact_duration:.2f}s)", "tts")

        except Exception as e:
            debug_log(f"ElevenLabs TTS error: {e}", "tts")
            print(f"  ElevenLabs TTS error: {e}", flush=True)
        finally:
            self._is_speaking.clear()
            self._notify_speaking_state(False)
            if self._completion_callback is not None and not interrupted:
                try:
                    self._completion_callback()
                except Exception as e:
                    print(f"  ElevenLabs TTS completion callback error: {e}", flush=True)
                self._completion_callback = None

    def speak(
        self,
        text: str,
        completion_callback: Optional[Callable[[], None]] = None,
        duration_callback: Optional[Callable[[float], None]] = None,
    ) -> None:
        if not self.enabled or not text:
            return
        self._completion_callback = completion_callback
        self._duration_callback = duration_callback
        try:
            self._q.put_nowait(text)
        except Exception as e:
            debug_log(f"ElevenLabs TTS enqueue error: {e}", "tts")

    def interrupt(self) -> None:
        self._should_interrupt.set()

    def _notify_speaking_state(self, is_speaking: bool) -> None:
        try:
            from desktop_app.face_widget import get_jarvis_state, JarvisState
            state_manager = get_jarvis_state()
            if is_speaking:
                debug_log("setting face state to SPEAKING (elevenlabs)", "tts")
                state_manager.set_state(JarvisState.SPEAKING)
        except ImportError:
            pass
        except Exception as e:
            debug_log(f"failed to set face state to SPEAKING (elevenlabs): {e}", "tts")

    def is_speaking(self) -> bool:
        return self._is_speaking.is_set()

    def get_last_spoken_text(self) -> str:
        return self._last_spoken_text