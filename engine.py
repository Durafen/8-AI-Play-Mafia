import random
import asyncio
import json
from typing import List, Dict
from models import Player
from api_clients import UnifiedLLMClient
from schemas import GameState, LogEntry, TurnOutput

# TTS Config
TTS_ENABLED = True   # Set to False to disable text-to-speech
TTS_RATE = "+20%"    # Speech speed: "+20%" = 20% faster, "-10%" = 10% slower
AUTO_CONTINUE = True # Set to True to run without user intervention
MEMORY_ENABLED = False # Set to True to enable distinct memories per player from previous games
REVEAL_ROLE_ON_DEATH = True # Set to False to hide role when player dies

# Config for Roster (with TTS voices)
# active: True = participates in game, False = disabled
# use_cli: True = CLI tool, False = API
ROSTER_CONFIG = [
    # OPENAI
    {"active": False, "use_cli": True, "name": "Rick", "provider": "openai", "model": "gpt-5.2", "voice": "en-US-GuyNeural"},
    {"active": False, "use_cli": True, "name": "Morty", "provider": "openai", "model": "gpt-5.1", "voice": "en-US-ChristopherNeural"},
    {"active": True, "use_cli": True, "name": "Gpt", "provider": "openai", "model": "gpt-5.1-codex-mini", "voice": "en-US-ChristopherNeural"},

    # ANTHROPIC
    {"active": True, "use_cli": True, "name": "Haiku", "provider": "anthropic", "model": "haiku", "voice": "en-GB-RyanNeural"},
    {"active": False, "use_cli": True, "name": "Sonnet", "provider": "anthropic", "model": "sonnet", "voice": "en-AU-WilliamNeural"},
    {"active": False, "use_cli": True, "name": "Opus", "provider": "anthropic", "model": "opus", "voice": "en-US-GuyNeural"},

    # OPENROUTER (API only)
    {"active": True, "use_cli": False, "name": "Chimera", "provider": "openrouter", "model": "tngtech/deepseek-r1t2-chimera:free", "voice": "en-AU-WilliamNeural"},
    {"active": False, "use_cli": False, "name": "Deepseek", "provider": "openrouter", "model": "nex-agi/deepseek-v3.1-nex-n1:free", "voice": "en-US-ChristopherNeural"},
    {"active": True, "use_cli": False, "name": "Devstral", "provider": "openrouter", "model": "mistralai/devstral-2512:free", "voice": "en-CA-LiamNeural"},
    {"active": False, "use_cli": False, "name": "Olmo", "provider": "openrouter", "model": "allenai/olmo-3.1-32b-think:free", "voice": "en-US-BrianNeural"},
    {"active": True, "use_cli": False, "name": "Oss", "provider": "openrouter", "model": "openai/gpt-oss-120b:free", "voice": "en-PH-JamesNeural"},

    # GOOGLE
    {"active": False, "use_cli": True, "name": "Pro", "provider": "google", "model": "gemini-2.5-pro", "voice": "en-NZ-MitchellNeural"},
    {"active": True, "use_cli": True, "name": "Flash", "provider": "google", "model": "gemini-2.5-flash", "voice": "en-IE-ConnorNeural"},
    {"active": True, "use_cli": True, "name": "Preview", "provider": "google", "model": "gemini-3-flash-preview", "voice": "en-CA-LiamNeural"},

    # QWEN (via qwen CLI)
    {"active": True, "use_cli": True, "name": "Qwen", "provider": "qwen", "model": "coder-model", "voice": "en-ZA-LukeNeural"},
    {"active": False, "use_cli": True, "name": "Vision", "provider": "qwen", "model": "vision-model", "voice": "en-CA-LiamNeural"},

    # OLLAMA (local)
    {"active": False, "use_cli": True, "name": "Nemotron", "provider": "ollama", "model": "nemotron-3-nano:30b-cloud", "voice": "en-US-GuyNeural"},
]

NARRATOR_VOICE = "en-US-AriaNeural"

import shutil
import os
import tempfile
import subprocess
import threading
import time
from datetime import datetime

import sys
import termios
import tty
import select

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False


class InputListener:
    """Handles non-blocking keyboard input"""
    def __init__(self):
        self.old_settings = None
        
    def __enter__(self):
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        return self
        
    def __exit__(self, type, value, traceback):
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            
    def is_data(self):
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])
        
    def check_for_space(self):
        if self.is_data():
            c = sys.stdin.read(1)
            if c == ' ':
                return True
        return False



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
        await communicate.save(output_path)

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
                        check=True, capture_output=True
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

    def _speak_with_name(self, text: str, voice: str, announce_player: str = None):
         # Deprecated
         pass


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


class GameEngine:
    def __init__(self, tts_enabled: bool = TTS_ENABLED):
        # Clean Logs - User requested logs/ be wiped on new game
        if os.path.exists("logs"):
            shutil.rmtree("logs")
        os.makedirs("logs", exist_ok=True)

        # Create persistent games directory
        os.makedirs("games", exist_ok=True)
        # Create memories directory
        os.makedirs("memories", exist_ok=True)

        # Initialize Game Log as a flat file in games/
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.game_log_path = os.path.join("games", f"game_{timestamp}.txt")

        with open(self.game_log_path, "w", encoding='utf-8') as f:
            f.write(f"=== MAFIA GAME LOG ({timestamp}) ===\n")

        # Restore individual player logs in logs/ dir
        self.client = UnifiedLLMClient(debug=True, log_dir="logs")
        self.state = GameState(reveal_role_on_death=REVEAL_ROLE_ON_DEATH)
        self.players: List[Player] = []
        self.active_players: Dict[str, Player] = {}  # Name -> Player obj

        # Initialize TTS
        self.tts = TTSEngine(enabled=tts_enabled)



    def _print(self, text):
        print(text)
        try:
            with open(self.game_log_path, "a", encoding='utf-8') as f:
                f.write(str(text) + "\n")
        except:
            pass

    def _announce(self, text: str, background: bool = False):
        """Speak system announcement with narrator voice"""
        self.tts.speak(text, voice=NARRATOR_VOICE, background=background)

    def log(self, phase: str, actor: str, action: str, content: str, is_secret: bool = False, vote_target: str = None, target_log: str = None):
        # Determine display name with number if actor is a player
        actor_display = actor
        player = next((p for p in self.players if p.state.name == actor), None)
        if player:
            idx = self.players.index(player) + 1
            actor_display = f"{idx}. {actor}:"

            # Add Mafia/Cop Icon for terminal display only
            if player.state.role == "Mafia":
                actor_display = f"👺 {actor_display}"
            elif player.state.role == "Cop":
                actor_display = f"👮 {actor_display}"

        # Append vote text to content for persistent history
        if vote_target:
             vote_marker = "(Secret Vote)" if is_secret else f"[VOTE: {vote_target}]"
             if "Voted for" not in content and "Suggests killing" not in content and "[Nominated" not in content and "[Suggests" not in content:
                  content = f"{content} {vote_marker}"

        entry = LogEntry(
            turn=self.state.turn,
            phase=phase,
            actor=actor,
            action=action,
            content=content
        )
        
        # Add Phase Icon
        phase_icon = ""
        if phase == "Day": phase_icon = "☀️ "
        elif phase == "Night": phase_icon = "🌙 "
        elif phase == "Trial": phase_icon = "⚖️ "
        elif phase == "Setup": phase_icon = "⚙️ "
        elif phase == "Reflection": phase_icon = ""

        # Append vote visual
        vote_str = ""
        if vote_target:
             icon = "🔪" if is_secret else "🗳️"
             vote_str = f" [{icon} {vote_target}]"

        # Determine Display Icon Rules
        # DEFAULT: No icon
        display_icon = ""
        
        # Rule 1: Phase Start ALWAYS gets the phase icon
        if action == "PhaseStart":
            display_icon = phase_icon
        
        # Rule 2: Speaking events get specific icons
        elif phase == "Day" and action == "speak":
            display_icon = "🗣️ "
        elif phase == "Night" and action == "whisper":
            display_icon = "🌚 "
        
        # Rule 3: Short/Intense phases always use their icon for visibility
        elif phase in ["Trial", "Setup", "Reflection"]:
             display_icon = phase_icon
             
        # Rule 4: Day/Night System messages (Info, etc) -> NO ICON
        # This prevents the "sun/moon on every line" issue.
        
        if is_secret:
            # Route secret logs to correct private log based on actor/context
            
            # Explicit override
            if target_log == "Cop":
                 self.state.cop_logs.append(entry)
            elif target_log == "Mafia":
                 self.state.mafia_logs.append(entry)
            
            # Auto-detect Cop actions
            elif (player and player.state.role == "Cop") or action == "investigate":
                 self.state.cop_logs.append(entry)
            
            # Auto-detect Cop System logs
            elif str(content).startswith("Investigation Result") or str(content).startswith("Investigation failed"):
                 self.state.cop_logs.append(entry)
            
            # Default: Mafia (including System messages for Night phase calls)
            else:
                 self.state.mafia_logs.append(entry)

            display_content = content.replace("[Nominated", "[👉 Nominated").replace("[Suggests killing", "[🔪 Suggests killing").replace("[Defense]", "[🛡️ Defense]").replace("votes guilty", "👎 votes guilty").replace("votes innocent", "👍 votes innocent").replace("abstains", "⏸️ abstains")
            self._print(f"\n{display_icon}{actor_display} {vote_str} {display_content}")
        else:
            self.state.public_logs.append(entry)
            display_content = content.replace("[Nominated", "[👉 Nominated").replace("[Suggests killing", "[🔪 Suggests killing").replace("[Defense]", "[🛡️ Defense]").replace("votes guilty", "👎 votes guilty").replace("votes innocent", "👍 votes innocent").replace("abstains", "⏸️ abstains")
            self._print(f"\n{display_icon}{actor_display}{vote_str} {display_content}")

    def setup_game(self):
        self._print("Initializing Game...")

        # 1. Filter active players and randomize order
        roster = [p for p in ROSTER_CONFIG if p.get("active", True)]
        random.shuffle(roster)

        player_count = len(roster)
        if player_count < 3:
            raise ValueError(f"Need at least 3 active players, got {player_count}")

        # 2. Assign Roles (2 Mafia, 1 Cop, rest Villagers)
        # Ensure we have enough players for special roles
        indices = list(range(player_count))
        mafia_indices = set(random.sample(indices, 2))
        
        cop_index = -1
        remaining_indices = [i for i in indices if i not in mafia_indices]
        if remaining_indices:
             cop_index = random.choice(remaining_indices)

        mafia_names = []
        
        # Create Players
        for i, config in enumerate(roster):
            role = "Villager"
            if i in mafia_indices:
                role = "Mafia"
            elif i == cop_index:
                role = "Cop"
            
            p = Player(
                name=config["name"],
                role=role,
                provider=config["provider"],
                model_name=config["model"],
                client=self.client,
                player_index=i+1,
                use_cli=config.get("use_cli", True),
                memory_enabled=MEMORY_ENABLED
            )
            self.players.append(p)
            self.state.players.append(p.state)
            self.active_players[p.state.name] = p

            # Register TTS voice for player
            self.tts.register_player(config["name"], config.get("voice", "en-US-AriaNeural"))

            if role == "Mafia":
                mafia_names.append(p.state.name)

        # Introduce Partners
        for p in self.players:
            if p.state.role == "Mafia":
                partner = [n for n in mafia_names if n != p.state.name]
                if partner:
                    p.set_partner(partner[0])

        self._print(f"[System] Game Initialized. Mafia are: {', '.join(mafia_names)}")
        self.log("Setup", "System", "MafiaReveal", f"Mafia: {', '.join(mafia_names)}", is_secret=True, target_log="Mafia")
        
        cop_names = [p.state.name for p in self.players if p.state.role == "Cop"]
        if cop_names:
             self.log("Setup", "System", "CopReveal", f"Cop: {cop_names[0]}", is_secret=True, target_log="Cop")

    def _get_living_players(self) -> List[Player]:
        return [p for p in self.players if p.state.is_alive]

    def _pause_game(self, listener):
        """Handle pause state - wait for SPACE to resume"""
        self._print("\n[PAUSED] Press SPACE to resume...")
        while True:
            if listener.check_for_space():
                self._print("[RESUMED]")
                return
            time.sleep(0.1)

    def _wait_for_speech_with_pause(self, listener):
        """Wait for TTS to finish while checking for SPACE to pause"""
        while self.tts._current_thread and self.tts._current_thread.is_alive():
            if listener and listener.check_for_space():
                self._pause_game(listener)
            time.sleep(0.1)

    def _wait_for_next(self, listener=None):
        if AUTO_CONTINUE:
            # Poll for SPACE key during sleep
            steps = 20  # 2 seconds / 0.1s
            for _ in range(steps):
                if listener and listener.check_for_space():
                    self._pause_game(listener)
                    return
                time.sleep(0.1)
            return
        input("\n[PRESS ENTER TO CONTINUE NEXT ACTION] >> ")

    def _save_game_stats(self, winner: str):
        """Save game stats to game_stats.json"""
        stats_path = "game_stats.json"

        # Load existing stats or create new
        if os.path.exists(stats_path):
            with open(stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
        else:
            stats = {"games": []}

        # Build game record
        game_id = os.path.basename(self.game_log_path).replace("game_", "").replace(".txt", "")
        mafia_names = [p.state.name for p in self.players if p.state.role == "Mafia"]

        game_record = {
            "id": game_id,
            "winner": winner,
            "turns": self.state.turn,
            "mafia": mafia_names,
            "players": [
                {
                    "name": p.state.name,
                    "role": p.state.role,
                    "survived": p.state.is_alive,
                    "provider": p.state.provider,
                    "model": p.state.model_name
                }
                for p in self.players
            ]
        }

        stats["games"].append(game_record)

        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        self._print(f"[Stats] Game saved to {stats_path}")

    def check_game_over(self) -> bool:
        mafia_count = sum(1 for p in self._get_living_players() if p.state.role == "Mafia")
        town_count = sum(1 for p in self._get_living_players() if p.state.role == "Villager")

        if mafia_count == 0:
            self._print("\n🎉 TOWN WINS! All Mafia eliminated. 🎉")
            self._announce("Town wins! All Mafia have been eliminated")
            self._save_game_stats("Town")
            self._run_reflection("Town")
            return True
        if mafia_count >= town_count:
            self._print("\n💀 MAFIA WINS! They have parity with Town. 💀")
            self._announce("Mafia wins!")
            self._save_game_stats("Mafia")
            self._run_reflection("Mafia")
            return True
        return False

    def run(self):
        with InputListener() as listener:
            self.setup_game()
    
            while True:
                # Wait for any remaining TTS before checking win
                self.tts.wait_for_speech()
    
                # Check Win Condition
                if self.check_game_over():
                    break
    
                # --- DAY PHASE ---
                self.state.phase = "Day"
    
                self.log("Day", "System", "PhaseStart", f"Day {self.state.turn}")
                self._announce(f"Day {self.state.turn} begins")
                
                # Determine Speaking Order (Rotate based on Day)
                # Day 1 start index 0, Day 2 start index 1, etc.
                start_idx = (self.state.turn - 1) % len(self.players)
                rotated_roster = self.players[start_idx:] + self.players[:start_idx]
                ordered_living = [p for p in rotated_roster if p.state.is_alive]
                

    
                # 1. Speaking Round
                living = self._get_living_players()
                # Track nominations (suggestions)
                nominations = {} # PlayerName -> VoteTarget
    
                self.log("Day", "System", "Info", f"Speaking order: {', '.join(p.state.name for p in ordered_living)}")


    
                for player in ordered_living:
                    # Double-check aliveness just in case state drifted
                    if not player.state.is_alive:
                        self._print(f"[DEBUG] Skipping dead player {player.state.name} in speaking order.")
                        continue

                    try:
                        # Generate while previous TTS might still be playing
                        output = player.take_turn(self.state, self.state.turn)
    
                        # Prepare TTS immediately (before waiting for previous to finish)
                        speech = output.speech or ""
                        # Capture nomination if present (no nominations on Day 1)
                        spoken_action = ""
                        if self.state.turn > 1 and output.vote and output.vote not in ("null", "None", "skip", "Skip"):
                             nominations[player.state.name] = output.vote
                             spoken_action = f"{player.state.name} nominates {output.vote}."
                        
                        tts_text = f"{speech} {spoken_action}".strip() if speech else spoken_action
                        audio_path = None
                        if tts_text:
                            # This blocks MAIN thread but runs while PREV TTS thread is playing
                            audio_path = self.tts.prepare_speech(tts_text, player.state.name, announce_name=True)

                        # Wait for previous TTS before displaying new output
                        self._wait_for_speech_with_pause(listener)

                        prefix = "👺 " if player.state.role == "Mafia" else ("👮 " if player.state.role == "Cop" else "")
                        if output.strategy:
                            self._print(f"\n💭 {prefix}{player.state.name} Strategy: {output.strategy}")

                        # Construct content with bracketed action if present
                        action_part = f"[Nominated {output.vote}] " if spoken_action else ""
                        content = f"{action_part}{speech}"
                        self.log("Day", player.state.name, "speak", content)
    
                        # Play pre-generated audio
                        if audio_path:
                            self.tts.play_file(audio_path, background=True)
    
                    except Exception as e:
                        self.log("Day", player.state.name, "error", f"Failed to speak: {e}")
    
                    # User can press Enter while TTS plays to trigger next generation
                    self._wait_for_next(listener)
    
                # 2. Trial Round
                # Identify Nominees (anyone with at least one nomination)
                nominee_counts = {}
                for target in nominations.values():
                     if any(p.state.name == target for p in living):
                         nominee_counts[target] = nominee_counts.get(target, 0) + 1

                if not nominee_counts:
                    self.log("Day", "System", "Info", "No nominations")
                else:
                    # Wait for last speaker to finish
                    self.tts.wait_for_speech()

                    # Sort by nomination count (highest first)
                    sorted_nominees = sorted(nominee_counts.items(), key=lambda x: x[1], reverse=True)

                    nominee_display = [f"{n} ({c})" for n, c in sorted_nominees]
                    self.log("Trial", "System", "PhaseStart", f"Nominees: {', '.join(nominee_display)}")
                    self._announce(f"Nominees for trial: {', '.join(nominee_display)}")

                    someone_died = False
                    for accused_name, nom_count in sorted_nominees:
                        if someone_died:
                            break

                        accused = self.active_players.get(accused_name)
                        if not accused or not accused.state.is_alive:
                            self._print(f"[DEBUG] Skipping trial for {accused_name} (Dead or invalid).")
                            continue

                        # --- TRIAL PHASE ---
                        self.state.phase = "Trial"
                        self.state.on_trial = accused_name
                        self.state.nominees = [accused_name]

                        self._print(f"\n⚖️  TRIAL: {accused_name} ⚖️")
                        self.log("Trial", "System", "Info", f"{accused_name} on trial")
                        self._announce(f"{accused_name}, defend yourself")

                        # 1. Defense - accused speaks
                        try:
                            output = accused.take_turn(self.state, self.state.turn)

                            # Prepare TTS
                            speech = output.speech or ""
                            audio_path = None
                            if speech:
                                audio_path = self.tts.prepare_speech(speech, accused_name, announce_name=True)

                            self._wait_for_speech_with_pause(listener)

                            prefix = "👺 " if accused.state.role == "Mafia" else ""
                            if output.strategy:
                                self._print(f"\n💭 {prefix}{accused_name} Strategy: {output.strategy}")
                            self.log("Trial", accused_name, "speak", f"[Defense] {speech}")

                            if audio_path:
                                self.tts.play_file(audio_path, background=True)
                        except Exception as e:
                            self._print(f"Error in defense: {e}")

                        self._wait_for_next(listener)

                        # 2. Judgment - all living players vote (except accused)
                        self.tts.wait_for_speech()
                        self._print(f"\n🗳️  JUDGMENT: {accused_name} 🗳️")
                        self._announce(f"Vote on {accused_name}: guilty, innocent, or abstain")

                        votes = {"guilty": 0, "innocent": 0, "abstain": 0}
                        voters = [p for p in self._get_living_players() if p.state.name != accused_name]

                        for voter in voters:
                            # Safeguard
                            if not voter.state.is_alive:
                                continue

                            try:
                                output = voter.take_turn(self.state, self.state.turn)
                                self._wait_for_speech_with_pause(listener)

                                prefix = "👺 " if voter.state.role == "Mafia" else ""
                                if output.strategy:
                                    self._print(f"\n💭 {prefix}{voter.state.name} Strategy: {output.strategy}")

                                vote = (output.vote or "").lower().strip()
                                if vote in ["guilty", "innocent"]:
                                    votes[vote] += 1
                                    self.log("Trial", voter.state.name, "vote", f"votes {vote}")
                                    self.tts.speak(f"{voter.state.name} votes {vote}.", voter.state.name, background=True)
                                else:
                                    votes["abstain"] += 1
                                    self.log("Trial", voter.state.name, "vote", "abstains")
                                    self.tts.speak(f"{voter.state.name} abstains.", voter.state.name, background=True)
                            except Exception as e:
                                self._print(f"Error voting: {e}")
                                votes["abstain"] += 1
                                self.log("Trial", voter.state.name, "vote", "abstains")

                            self._wait_for_next(listener)

                        # 3. Verdict
                        self.tts.wait_for_speech()
                        self.log("Trial", "System", "VoteSummary", f"Guilty: {votes['guilty']}, Innocent: {votes['innocent']}, Abstain: {votes['abstain']}")
                        self._announce(f"Guilty: {votes['guilty']}, Innocent: {votes['innocent']}, Abstain: {votes['abstain']}")

                        if votes["guilty"] > votes["innocent"]:
                            # GUILTY - death
                            self._print(f"\n💀💀💀 {accused_name} IS GUILTY 💀💀💀")
                            self.log("Result", "System", "Death", f"{accused_name} HANGED")
                            self._announce(f"{accused_name} has been found guilty and hanged")
                            accused.state.is_alive = False
                            if self.state.reveal_role_on_death:
                                role_emoji = "👺" if accused.state.role == "Mafia" else "👤"
                                self.log("Result", "System", "RoleReveal", f"{role_emoji} {accused_name} was a {accused.state.role}!")
                                self._announce(f"{accused_name} was a {accused.state.role}")
                            someone_died = True
                        else:
                            # INNOCENT - released
                            self._print(f"\n✅ {accused_name} IS RELEASED ✅")
                            self.log("Result", "System", "Released", f"{accused_name} released")
                            self._announce(f"{accused_name} has been found innocent and released")

                        self._wait_for_next(listener)

                    # Clear trial state
                    self.state.on_trial = None

                self._wait_for_next(listener)
    
                # Wait for any remaining TTS before checking win
                self.tts.wait_for_speech()
    
                # Check Win again before Night
                if self.check_game_over():
                    break
    
                # --- NIGHT PHASE ---
                self.state.phase = "Night"
                self.log("Night", "System", "PhaseStart", f"Night {self.state.turn}")
                self._announce(f"Night {self.state.turn} begins")
    
                mafia_alive = [p for p in self._get_living_players() if p.state.role == "Mafia"]
                
                if not mafia_alive:
                    # Game over loop will catch this next iter
                    pass
                else:

                    # --- MAFIA TURN ---
                    self.log("Night", "System", "MafiaWake", "Mafia awake", is_secret=True, target_log="Mafia")
                    
                    night_victim = None
                    mafia_votes = {}
                    for m_player in mafia_alive:
                        if not m_player.state.is_alive:
                             continue
                        try:
                            # Generate while previous TTS might still be playing
                            output = m_player.take_turn(self.state, self.state.turn)

                            target = output.vote
                            # Normalize: strip "kill " prefix if LLM included it
                            if target and target.lower().startswith("kill "):
                                target = target[5:].strip()
                            action_tag = f"[Suggests killing {target}] " if target else ""
                            spoken_action = f"{m_player.state.name} suggests killing {target}." if target else ""
                            
                            # Prepare TTS
                            speech = output.speech or ""
                            tts_text = f"{speech} {spoken_action}".strip() if speech else spoken_action
                            audio_path = None
                            if tts_text:
                                 audio_path = self.tts.prepare_speech(tts_text, m_player.state.name, announce_name=True)

                            # Wait for previous TTS before displaying
                            self._wait_for_speech_with_pause(listener)

                            if output.strategy:
                                self._print(f"\n💭 👺 {m_player.state.name} Strategy: {output.strategy}")
                            
                            content = f"{action_tag}{output.speech or ''}"
                            self.log("Night", m_player.state.name, "whisper", content, is_secret=True, target_log="Mafia")
    
                            # Play pre-generated audio
                            if audio_path:
                                self.tts.play_file(audio_path, background=True)
    
                            if target:
                                mafia_votes[target] = mafia_votes.get(target, 0) + 1
                        except Exception as e:
                            self._print(f"Mafia Error: {e}")
    
                        # User can press Enter while TTS plays
                        if m_player != mafia_alive[-1]:
                             self._wait_for_next(listener)
    
                    # Consensus Summary
                    tally_parts = [f"{k} ({v})" for k,v in mafia_votes.items()]
                    tally_str = ", ".join(tally_parts) if tally_parts else "No votes"
                    self.log("Night", "System", "VoteSummary", f"Votes: {tally_str}", is_secret=True, target_log="Mafia")
    
                    # Do NOT wait for last mafia TTS here. 
                    # We proceed to Cop generation immediately primarily so Cop generates while Mafia talks.
                    # self.tts.wait_for_speech()
    
                    if mafia_votes:
                        max_votes = max(mafia_votes.values())
                        winners = [k for k, v in mafia_votes.items() if v == max_votes]
    
                        if len(winners) > 1:
                            import random
                            kill_target = random.choice(winners)
                            self.log("Night", "System", "TieBreak", f"Tie {winners}. Random: {kill_target}", is_secret=True, target_log="Mafia")
                        else:
                            kill_target = winners[0]
    
                        # Execute Kill (Secretly for now)
                        if kill_target in self.active_players:
                            victim = self.active_players[kill_target]
                            if victim.state.is_alive:
                                victim.state.is_alive = False
                                night_victim = kill_target
                                self.log("Night", "System", "Kill", f"Mafia killed {kill_target}", is_secret=True, target_log="Mafia")
                            else:
                                self.log("Night", "System", "Fail", f"Target {kill_target} already dead", is_secret=True, target_log="Mafia")
                        else:
                             self.log("Night", "System", "Fail", "Invalid target", is_secret=True, target_log="Mafia")
                    else:
                         self.log("Night", "System", "Quiet", "No kill", is_secret=True, target_log="Mafia")

                    # --- COP TURN (After Mafia Kill) ---
                    cop_alive = [p for p in self._get_living_players() if p.state.role == "Cop"]
                    for cop in cop_alive:
                        if not cop.state.is_alive: continue # Safeguard (if died tonight)
                        
                        try:
                            # Cop Turn
                            output = cop.take_turn(self.state, self.state.turn)

                            target_name = output.vote
                            # Normalize
                            if target_name and target_name.lower().startswith("investigate "):
                                target_name = target_name[12:].strip()

                            # Prepare TTS
                            speech = output.speech or ""
                            spoken_action = f"{cop.state.name} investigating {target_name}." if target_name else ""
                            tts_text = f"{speech} {spoken_action}".strip() if speech else spoken_action
                            
                            audio_path = None
                            if tts_text:
                                audio_path = self.tts.prepare_speech(tts_text, cop.state.name, announce_name=True)

                            self._wait_for_speech_with_pause(listener)
                            
                            if output.strategy:
                                self._print(f"\n💭 👮 {cop.state.name} Strategy: {output.strategy}")

                            # Log action
                            action_desc = f"[Investigates {target_name}]" if target_name else "[No investigation]"
                            self.log("Night", cop.state.name, "investigate", f"{action_desc} {output.speech or ''}", is_secret=True, target_log="Cop")
                            
                            # Play Audio
                            if audio_path:
                                self.tts.play_file(audio_path, background=True)

                            # Resolve Investigation
                            if target_name:
                                target = self.active_players.get(target_name)
                                if target:
                                    # Result: "Role: Mafia" or "Role: Villager" (or other)
                                    # For game balance, Cop usually sees "Suspicious" (Mafia) or "Innocent" (Villager/Cop/Doctor)
                                    # But let's give exact role for now as per user request to buff town.
                                    result = "Mafia" if target.state.role == "Mafia" else "Innocent"
                                    
                                    investigation_msg = f"Investigation Result: {target_name} is {result}."
                                    self._print(f"\n🔍 {cop.state.name} checks {target_name}... Result: {result}")
                                    
                                    # Log to Cop's secret log
                                    self.state.cop_logs.append(LogEntry(
                                        turn=self.state.turn,
                                        phase=f"Night {self.state.turn}",
                                        actor="System",
                                        action="Info",
                                        content=investigation_msg
                                    ))
                                else:
                                     self.state.cop_logs.append(LogEntry(
                                        turn=self.state.turn,
                                        phase=f"Night {self.state.turn}",
                                        actor="System",
                                        action="Info",
                                        content=f"Investigation failed: {target_name} not found."
                                    ))

                        except Exception as e:
                            self._print(f"Cop Error: {e}")
                        
                        self._wait_for_next(listener)
    
                        self._wait_for_next(listener)
    
                    # --- DEATH REVEAL (Morning) ---
                    if night_victim:
                         self._print(f"\n🩸 TRAGEDY! {night_victim} was found DEAD in the morning.🩸")
                         self._announce(f"{night_victim} was killed during the night")
                         self.log("Night", "System", "Death", f"{night_victim} found dead")
                         
                         # Role Reveal - DISABLED INITIAL NIGHT REVEAL AS PER RULE
                         # if self.state.reveal_role_on_death:
                         #      victim = self.active_players[night_victim]
                         #      role_emoji = "👺" if victim.state.role == "Mafia" else "👤"
                         #      self._print(f"{role_emoji} {night_victim} was a {victim.state.role}!")
                         #      self.log("Night", "System", "RoleReveal", f"{night_victim} was a {victim.state.role}")

                    self._wait_for_next(listener)
                    self.state.turn += 1

    def _run_reflection(self, winner: str):
        """Allow all players to reflect and update their memories"""
        if not MEMORY_ENABLED:
            self.log("Reflection", "System", "Skip", "Memory system disabled. Skipping reflection.")
            return

        self.state.phase = "Reflection"
        self._print("\n🧠 REFLECTION PHASE 🧠")
        self._print("Players are analyzing their performance...")
        self.log("Reflection", "System", "PhaseStart", "--- REFLECTION PHASE START ---")
        
        # Log Winner and Mafia Reveal
        self.log("Reflection", "System", "Winner", f"The Winner is: {winner}")
        mafia_names = [p.state.name for p in self.players if p.state.role == "Mafia"]
        self.log("Reflection", "System", "MafiaReveal", f"The Mafia were: {', '.join(mafia_names)}")

        self._announce("The game is over. Players are now reflecting on their strategy.")

        # Wait for announcement to finish before starting the reflection loop
        self.tts.wait_for_speech()

        for p in self.players:
            self._print(f"Writing memory for {p.state.name}...")
            try:
                # 1. Generate Reflection (Blocking)
                # While this computes, the PREVIOUS player's TTS might be playing in background.
                new_memory = p.reflect_on_game(self.state, winner)
                
                # 2. Save to file
                with open(f"memories/{p.state.name}.txt", "w", encoding='utf-8') as f:
                    f.write(new_memory)
                
                # 3. Log to Game Log and Console
                self._print(f"\n🧠 {p.state.name} Memory: {new_memory}")
                self.log("Reflection", p.state.name, "reflect", new_memory)

            except Exception as e:
                self._print(f"Error saving memory for {p.state.name}: {e}")
        
        self._print("All memories updated for next game.")

if __name__ == "__main__":
    engine = GameEngine()
    engine.run()
