import random
import asyncio
from typing import List, Dict
from models import Player
from api_clients import UnifiedLLMClient
from schemas import GameState, LogEntry, TurnOutput

# TTS Config
TTS_ENABLED = True   # Set to False to disable text-to-speech
TTS_RATE = "+20%"    # Speech speed: "+20%" = 20% faster, "-10%" = 10% slower
AUTO_CONTINUE = True # Set to True to run without user intervention
MEMORY_ENABLED = False # Set to True to enable distinct memories per player from previous games

# Config for Roster (with TTS voices)
# active: True = participates in game, False = disabled
# use_cli: True = CLI tool, False = API
ROSTER_CONFIG = [
    # OPENAI
    {"active": False, "use_cli": True, "name": "Rick", "provider": "openai", "model": "gpt-5.2", "voice": "en-US-GuyNeural"},
    {"active": False, "use_cli": True, "name": "Morty", "provider": "openai", "model": "gpt-5.1", "voice": "en-US-ChristopherNeural"},

    # ANTHROPIC
    {"active": True, "use_cli": True, "name": "Haiku", "provider": "anthropic", "model": "haiku", "voice": "en-GB-RyanNeural"},
    {"active": False, "use_cli": True, "name": "Sonnet", "provider": "anthropic", "model": "sonnet", "voice": "en-AU-WilliamNeural"},
    {"active": False, "use_cli": True, "name": "Opus", "provider": "anthropic", "model": "opus", "voice": "en-US-GuyNeural"},

    # OPENROUTER (API only)
    {"active": True, "use_cli": False, "name": "Chimera", "provider": "openrouter", "model": "tngtech/deepseek-r1t2-chimera:free", "voice": "en-AU-WilliamNeural"},
    {"active": True, "use_cli": False, "name": "Deepseek", "provider": "openrouter", "model": "nex-agi/deepseek-v3.1-nex-n1:free", "voice": "en-US-ChristopherNeural"},
    {"active": True, "use_cli": False, "name": "Devstral", "provider": "openrouter", "model": "mistralai/devstral-2512:free", "voice": "en-CA-LiamNeural"},

    # GOOGLE
    {"active": True, "use_cli": True, "name": "Pro", "provider": "google", "model": "gemini-2.5-pro", "voice": "en-NZ-MitchellNeural"},
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
        use_voice = voice or self._voice_map.get(player_name, "en-US-AriaNeural")

        if background:
            self.wait_for_speech()
            self._current_thread = threading.Thread(
                target=self._speak_with_name, args=(text, use_voice, player_name if announce_name else None), daemon=True
            )
            self._current_thread.start()
        else:
            self.wait_for_speech()
            self._speak_with_name(text, use_voice, player_name if announce_name else None)

    def _speak_with_name(self, text: str, voice: str, announce_player: str = None):
        """Speak with optional name announcement in narrator voice first."""
        try:
            # Pre-generate main speech audio (strip markdown emphasis)
            clean_text = text.replace("*", "")
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                speech_path = f.name
            asyncio.run(self._generate_audio(clean_text, voice, speech_path))

            # Get name audio and concatenate if needed
            if announce_player:
                name_audio = self._get_cached_name(announce_player)
                if name_audio and os.path.exists(name_audio):
                    # Concatenate name + speech into single file for seamless playback
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                        combined_path = f.name
                    with tempfile.NamedTemporaryFile(mode='w', suffix=".txt", delete=False) as f:
                        f.write(f"file '{name_audio}'\nfile '{speech_path}'\n")
                        list_path = f.name
                    subprocess.run(
                        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", combined_path],
                        check=True, capture_output=True
                    )
                    os.unlink(list_path)
                    subprocess.run(["afplay", combined_path], check=True)
                    os.unlink(combined_path)
                    os.unlink(speech_path)
                    return

            # No name announcement - just play speech
            subprocess.run(["afplay", speech_path], check=True)
            os.unlink(speech_path)
        except Exception as e:
            print(f"[TTS Error] {e}")


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
        self.state = GameState()
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

    def log(self, phase: str, actor: str, action: str, content: str, is_secret: bool = False, vote_target: str = None):
        # Determine display name with number if actor is a player
        actor_display = actor
        player = next((p for p in self.players if p.state.name == actor), None)
        if player:
            idx = self.players.index(player) + 1
            actor_display = f"{idx}. {actor}:"

            # Add Mafia Icon for terminal display only
            if player.state.role == "Mafia":
                actor_display = f"👺 {actor_display}"

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
        elif phase == "Voting": phase_icon = "🗳️ "
        elif phase == "Defense": phase_icon = "🛡️ "
        elif phase == "LastWords": phase_icon = "💀 "
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
        elif phase in ["Voting", "Defense", "LastWords", "Setup", "Reflection"]:
             display_icon = phase_icon
             
        # Rule 4: Day/Night System messages (Info, etc) -> NO ICON
        # This prevents the "sun/moon on every line" issue.
        
        if is_secret:
            self.state.mafia_logs.append(entry)
            display_content = content.replace("[Nominated", "[👉 Nominated").replace("[Voted for", "[🗳️ Voted for").replace("[Suggests killing", "[🔪 Suggests killing").replace("[Defense]", "[🛡️ Defense]").replace("[Last Words]", "[💀 Last Words]")
            self._print(f"\n{display_icon}{actor_display} {vote_str} {display_content}")
        else:
            self.state.public_logs.append(entry)
            display_content = content.replace("[Nominated", "[👉 Nominated").replace("[Voted for", "[🗳️ Voted for").replace("[Suggests killing", "[🔪 Suggests killing").replace("[Defense]", "[🛡️ Defense]").replace("[Last Words]", "[💀 Last Words]")
            self._print(f"\n{display_icon}{actor_display}{vote_str} {display_content}")

    def setup_game(self):
        self._print("Initializing Game...")

        # 1. Filter active players and randomize order
        roster = [p for p in ROSTER_CONFIG if p.get("active", True)]
        random.shuffle(roster)

        player_count = len(roster)
        if player_count < 3:
            raise ValueError(f"Need at least 3 active players, got {player_count}")

        # 2. Assign Roles (2 Mafia, rest Villagers)
        mafia_indices = set(random.sample(range(player_count), 2))

        mafia_names = []
        
        # Create Players
        for i, config in enumerate(roster):
            role = "Mafia" if i in mafia_indices else "Villager"
            
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
        self.log("Setup", "System", "MafiaReveal", f"Mafia: {', '.join(mafia_names)}", is_secret=True)

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

    def check_game_over(self) -> bool:
        mafia_count = sum(1 for p in self._get_living_players() if p.state.role == "Mafia")
        town_count = sum(1 for p in self._get_living_players() if p.state.role == "Villager")

        if mafia_count == 0:
            self._print("\n🎉 TOWN WINS! All Mafia eliminated. 🎉")
            self._announce("Town wins! All Mafia have been eliminated")
            self._run_reflection("Town")
            return True
        if mafia_count >= town_count:
            self._print("\n💀 MAFIA WINS! They have parity with Town. 💀")
            self._announce("Mafia wins!")
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
    
                self.log("Day", "System", "Info", f"Order: {', '.join(p.state.name for p in ordered_living)}")


    
                for player in ordered_living:
                    try:
                        # Generate while previous TTS might still be playing
                        output = player.take_turn(self.state, self.state.turn)
    
                        # Wait for previous TTS before displaying new output
                        self._wait_for_speech_with_pause(listener)

                        prefix = "👺 " if player.state.role == "Mafia" else ""
                        if output.strategy:
                            self._print(f"\n💭 {prefix}{player.state.name} Strategy: {output.strategy}")

                        # Construct content with bracketed action if present
                        speech = output.speech or ""
                        action_part = ""
    
                        # Capture nomination if present (Day 2+)
                        spoken_action = ""
                        if self.state.turn > 1 and output.vote and output.vote not in ("null", "None", "skip", "Skip"):
                             nominations[player.state.name] = output.vote
                             action_part = f"[Nominated {output.vote}] "
                             spoken_action = f"{player.state.name} nominates {output.vote}."
    
                        content = f"{action_part}{speech}"
                        self.log("Day", player.state.name, "speak", content)
    
                        # Start TTS in background with speech + nomination
                        tts_text = f"{speech} {spoken_action}".strip() if speech else spoken_action
                        if tts_text:
                            self.tts.speak(tts_text, player.state.name, background=True, announce_name=True)
    
                    except Exception as e:
                        self.log("Day", player.state.name, "error", f"Failed to speak: {e}")
    
                    # User can press Enter while TTS plays to trigger next generation
                    self._wait_for_next(listener)
    
                # 2. Defense & Voting Round (Skip on Day 1)
                if self.state.turn > 1:
                    
                    # --- DEFENSE PHASE ---
                    # Identify Nominees (anyone with at least one nomination)
                    nominee_counts = {}
                    for target in nominations.values():
                         if any(p.state.name == target for p in living):
                             nominee_counts[target] = nominee_counts.get(target, 0) + 1
                    
                    nominees = list(nominee_counts.keys())
                    self.state.nominees = nominees # Save for prompt generation
    
                    if nominees:
                        # Wait for last speaker to finish
                        self.tts.wait_for_speech()
                        
                        self.state.phase = "Defense"

                        self._print(f"\n🛡️  DEFENSE PHASE 🛡️")
                        
                        nominee_display = [f"{n} ({nominee_counts[n]})" for n in nominees]
                        self.log("Defense", "System", "PhaseStart", f"Nominees: {', '.join(nominee_display)}")
                        self._announce(f"Nominees for elimination are: {', '.join(nominee_display)}")

                        # Sort nominees by day speaking order
                        ordered_nominees = [p.state.name for p in ordered_living if p.state.name in nominees]
                        for nom_name in ordered_nominees:
                            # Find player object
                            nom_player = next((p for p in living if p.state.name == nom_name), None)
                            if not nom_player: continue
    
                            try:
                                # Generate while previous TTS might still be playing
                                output = nom_player.take_turn(self.state, self.state.turn)

                                # Wait for previous TTS before displaying
                                self._wait_for_speech_with_pause(listener)

                                if output.strategy:
                                    self._print(f"\n💭 {nom_player.state.name} Strategy: {output.strategy}")
                                self.log("Defense", nom_player.state.name, "speak", f"[Defense] {output.speech or ''}")
    
                                # TTS for defense speech in background
                                if output.speech:
                                    self.tts.speak(output.speech, nom_player.state.name, background=True, announce_name=True)
                            except Exception as e:
                                self._print(f"Error defending: {e}")
    
                            # User can press Enter while TTS plays
                            self._wait_for_next(listener)
                    else:
                        self.log("Day", "System", "Info", "No nominations")
    
                    # --- VOTING PHASE ---
                    # Wait for any remaining Defense TTS
                    self.tts.wait_for_speech()
    
                    self.state.phase = "Voting"
                    self._print("\n🗳️  VOTING TIME 🗳️")

                    living_count = len([p for p in living])
                    candidates_str = ", ".join(nominees) if nominees else "everyone (since no nominations)"
                    
                    self.log("Voting", "System", "Info", f"{living_count} voters. Candidates: {candidates_str}")

                    self._announce("It is voting time.")

    
                    final_votes = {} # PlayerName -> TargetName
    
                    for player in living:
                        try:
                            # Generate while previous TTS might still be playing
                            output = player.take_turn(self.state, self.state.turn)
                            vote_target = output.vote

                            # Wait for previous TTS before displaying
                            self._wait_for_speech_with_pause(listener)

                            prefix = "👺 " if player.state.role == "Mafia" else ""
                            if output.strategy:
                                self._print(f"\n💭 {prefix}{player.state.name} Strategy: {output.strategy}")

                            # Validate vote
                            # MUST be in nominees list (if nominees exist)
                            if vote_target in [None, "null", "Skip", "skip"]:
                                vote_target = None
                            elif nominees and vote_target not in nominees:
                                self._print(f"[Invalid Vote] {player.state.name} voted for {vote_target} (Not a nominee)")
                                vote_target = None
                            elif not nominees and vote_target not in [p.state.name for p in living]:
                                 # Fallback
                                 vote_target = None
    
                            final_votes[player.state.name] = vote_target
    
                            # Log the vote
                            if vote_target:
                                self.log("Voting", player.state.name, "vote", f"[Vote: {vote_target}]")
                                # TTS for vote announcement
                                self.tts.speak(f"{player.state.name} votes for {vote_target}.", player.state.name, background=True)
                            else:
                                # Log as absent/abstain with square brackets
                                self.log("Voting", player.state.name, "vote", "[Abstain]")
                                self.tts.speak(f"{player.state.name} abstains.", player.state.name, background=True)
    
                        except Exception as e:
                            self._print(f"Error voting: {e}")
                            final_votes[player.state.name] = None
                            self.log("Voting", player.state.name, "vote", "[Abstain]")
    
                        # User can press Enter while TTS plays
                        self._wait_for_next(listener)
    
                    # Aggregate Tally from final_votes
                    votes = {}
                    for target in final_votes.values():
                        if target:
                            votes[target] = votes.get(target, 0) + 1
    
                    # Tally Summary
                    self.tts.wait_for_speech()
                    tally_parts = [f"{k} ({v})" for k,v in votes.items()]
                    tally_str = ", ".join(tally_parts) if tally_parts else "No votes"
                    self.log("Voting", "System", "VoteSummary", f"Votes: {tally_str}")
                    self._announce(f"Votes Cast: {tally_str}")

    
                    if not votes:
                        self.log("Result", "System", "NoLynch", "No votes")
                    else:
                        target, count = max(votes.items(), key=lambda x: x[1])
                        max_votes = count
                        # Identify all players with max votes
                        victims_names = [k for k, v in votes.items() if v == max_votes]
                        
                        if len(victims_names) > 1:
                            self.log("Result", "System", "Tie", f"Tie: {', '.join(victims_names)}. All eliminated.")
                            self._announce("There is a tie. All tied players will be eliminated")
                        
                        # Process deaths
                        for v_name in victims_names:
                            victim = self.active_players.get(v_name)
                            if not victim or not victim.state.is_alive: continue
                            
                            # Execute Kill
                            victim.state.is_alive = False
                            self.log("Result", "System", "Death", f"{v_name} HANGED")
                            self._announce(f"{v_name} has been hanged by the town")
                            self._print(f"💀💀💀 {v_name} IS DEAD 💀💀💀")
                            role_emoji = "👺" if victim.state.role == "Mafia" else "👤"
                            self.log("Result", "System", "RoleReveal", f"{role_emoji} {v_name} was a {victim.state.role}!")

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
                    self.log("Night", "System", "MafiaWake", "Mafia awake", is_secret=True)
                    
                    mafia_votes = {}
                    for m_player in mafia_alive:
                        try:
                            # Generate while previous TTS might still be playing
                            output = m_player.take_turn(self.state, self.state.turn)

                            # Wait for previous TTS before displaying
                            self._wait_for_speech_with_pause(listener)

                            if output.strategy:
                                self._print(f"\n💭 👺 {m_player.state.name} Strategy: {output.strategy}")

                            target = output.vote
                            action_tag = f"[Suggests killing {target}] " if target else ""
                            spoken_action = f"{m_player.state.name} suggests killing {target}." if target else ""
                            content = f"{action_tag}{output.speech or ''}"
                            self.log("Night", m_player.state.name, "whisper", content, is_secret=True)
    
                            # TTS for mafia whisper in background with kill suggestion
                            speech = output.speech or ""
                            tts_text = f"{speech} {spoken_action}".strip() if speech else spoken_action
                            if tts_text:
                                self.tts.speak(tts_text, m_player.state.name, background=True, announce_name=True)
    
                            if target:
                                mafia_votes[target] = mafia_votes.get(target, 0) + 1
                        except Exception as e:
                            self._print(f"Mafia Error: {e}")
    
                        # User can press Enter while TTS plays
                        self._wait_for_next(listener)
    
                    # Consensus Summary
                    tally_parts = [f"{k} ({v})" for k,v in mafia_votes.items()]
                    tally_str = ", ".join(tally_parts) if tally_parts else "No votes"
                    self.log("Night", "System", "VoteSummary", f"Votes: {tally_str}", is_secret=True)
    
                    # Wait for last mafia TTS to finish before summary
                    self.tts.wait_for_speech()
    
                    if mafia_votes:
                        max_votes = max(mafia_votes.values())
                        winners = [k for k, v in mafia_votes.items() if v == max_votes]
    
                        if len(winners) > 1:
                            import random
                            kill_target = random.choice(winners)
                            self.log("Night", "System", "TieBreak", f"Tie {winners}. Random: {kill_target}", is_secret=True)
                        else:
                            kill_target = winners[0]
    
                        # Execute Kill
                        if kill_target in self.active_players and self.active_players[kill_target].state.is_alive:
                            victim = self.active_players[kill_target]
                            victim.state.is_alive = False
                            self.log("Night", "System", "Kill", f"Mafia killed {kill_target}")
                            self._print(f"\n🩸 TRAGEDY! {kill_target} was found DEAD in the morning.🩸")
                            self._announce(f"{kill_target} was killed during the night")
                        else:
                             self.log("Night", "System", "Fail", "Invalid target")
                    else:
                         self.log("Night", "System", "Quiet", "No kill")
    
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
