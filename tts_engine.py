# tts_engine.py - Text-to-Speech engine using Edge TTS

import os
import asyncio
import tempfile
import subprocess
import threading

from config import TTS_RATE, NARRATOR_VOICE

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False


class TTSEngine:
    """Edge TTS wrapper with background playback support"""

    def __init__(self, enabled: bool = True, rate: str = TTS_RATE):
        self.enabled = enabled and EDGE_TTS_AVAILABLE
        self.rate = rate
        self._voice_map = {}  # player_name -> voice_id
        self._name_cache = {}  # player_name -> cached audio path
        self._current_thread = None  # Track current TTS thread
        if enabled and not EDGE_TTS_AVAILABLE:
            print("[TTS] edge-tts not installed. Run: pip install edge-tts")

    def register_player(self, name: str, voice: str):
        self._voice_map[name] = voice

    def wait_for_speech(self):
        """Wait for current speech to finish"""
        if self._current_thread and self._current_thread.is_alive():
            self._current_thread.join()

    def _get_cached_name(self, player_name: str) -> str:
        """Get or create cached audio file for player name announcement."""
        if player_name in self._name_cache:
            path = self._name_cache[player_name]
            if os.path.exists(path):
                return path

        # Generate and cache name audio in narrator voice
        cache_dir = os.path.join(tempfile.gettempdir(), "mafia_tts_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"name_{player_name}.mp3")

        if not os.path.exists(cache_path):
            try:
                asyncio.run(self._generate_audio(f"{player_name}.", NARRATOR_VOICE, cache_path))
            except Exception as e:
                print(f"[TTS] Failed to cache name: {e}")
                return None

        self._name_cache[player_name] = cache_path
        return cache_path

    async def _generate_audio(self, text: str, voice: str, output_path: str):
        """Generate audio file from text."""
        communicate = edge_tts.Communicate(text, voice, rate=self.rate)
        await asyncio.wait_for(communicate.save(output_path), timeout=30.0)

    def speak(self, text: str, player_name: str = None, voice: str = None, background: bool = False, announce_name: bool = False):
        """Speak text. If background=True, runs in background thread. If announce_name=True, plays cached name in narrator voice first."""
        if not self.enabled or not text or not text.strip():
            return

        path = self.prepare_speech(text, player_name, voice, announce_name)
        if path:
            self.play_file(path, background)

    def prepare_speech(self, text: str, player_name: str = None, voice: str = None, announce_name: bool = False) -> str:
        """Generate audio file and return path. Blocks until generation complete."""
        if not self.enabled or not text or not text.strip():
            return None

        use_voice = voice or self._voice_map.get(player_name, "en-US-AriaNeural")
        try:
             # Pre-generate main speech audio (strip markdown emphasis)
            clean_text = text.replace("*", "")
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                speech_path = f.name
            asyncio.run(self._generate_audio(clean_text, use_voice, speech_path))

            # Get name audio and concatenate if needed
            if announce_name and player_name:
                name_audio = self._get_cached_name(player_name)
                if name_audio:
                     # Concatenate name + speech
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                        combined_path = f.name

                    list_path = speech_path + ".list"
                    with open(list_path, "w") as f:
                         f.write(f"file '{name_audio}'\nfile '{speech_path}'")

                    subprocess.run(
                        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", combined_path],
                        check=True, capture_output=True, timeout=30
                    )
                    os.unlink(list_path)
                    os.unlink(speech_path) # Delete original speech part
                    return combined_path

            return speech_path

        except Exception as e:
            print(f"[TTS Error in prepare] {e}")
            return None

    def play_file(self, path: str, background: bool = False):
        """Play an existing audio file"""
        if background:
            self.wait_for_speech()
            self._current_thread = threading.Thread(
                target=self._play_file_sync, args=(path,), daemon=True
            )
            self._current_thread.start()
        else:
            self.wait_for_speech()
            self._play_file_sync(path)

    def _play_file_sync(self, path: str):
        try:
            subprocess.run(["afplay", path], check=True)
        except Exception as e:
            print(f"[TTS Play Error] {e}")
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def _speak_sync(self, text: str, voice: str):
        """Synchronous speech (runs TTS and plays audio)"""
        try:
            asyncio.run(self._speak_async(text, voice))
        except Exception as e:
            print(f"[TTS Error] {e}")

    async def _speak_async(self, text: str, voice: str):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            temp_path = f.name
        try:
            communicate = edge_tts.Communicate(text, voice, rate=self.rate)
            await communicate.save(temp_path)
            subprocess.run(["afplay", temp_path], check=True)  # macOS
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
