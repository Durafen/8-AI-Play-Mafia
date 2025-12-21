import random
import json
import os
import shutil
import time
import concurrent.futures
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from models import Player
from api_clients import UnifiedLLMClient
from schemas import GameState, LogEntry, TurnOutput
from config import (
    TTS_ENABLED, AUTO_CONTINUE, MEMORY_ENABLED, REVEAL_ROLE_ON_DEATH,
    NARRATOR_VOICE, ROLE_EMOJIS, PHASE_EMOJIS, ROSTER_CONFIG
)
from tts_engine import TTSEngine
from input_listener import InputListener


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

            # Add role icon for terminal display only
            if player.state.role in ("Mafia", "Cop"):
                actor_display = f"{self._get_role_emoji(player.state.role)} {actor_display}"

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
        phase_icon = PHASE_EMOJIS.get(phase, "")

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

    # --- HELPER METHODS ---

    def _get_role_emoji(self, role: str) -> str:
        """Get emoji for role. Single source of truth."""
        return ROLE_EMOJIS.get(role, "👤")

    def _get_strategy_prefix(self, player: Player) -> str:
        """Get prefix for strategy display (includes role emoji for Mafia/Cop)."""
        if player.state.role == "Mafia":
            return "👺 "
        elif player.state.role == "Cop":
            return "👮 "
        return ""

    def _print_strategy(self, player: Player, output: TurnOutput) -> None:
        """Print strategy with role-appropriate prefix."""
        if output.strategy:
            prefix = self._get_strategy_prefix(player)
            self._print(f"\n💭 {prefix}{player.state.name} Strategy: {output.strategy}")

    def _start_background_turn(self, player: Player) -> Tuple[Optional[concurrent.futures.Future], Optional[concurrent.futures.ThreadPoolExecutor]]:
        """Start a player's turn in background thread. Returns (future, executor)."""
        if not player:
            return None, None
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(player.take_turn, self.state, self.state.turn)
        return future, executor

    def _get_background_result(self, future: Optional[concurrent.futures.Future], executor: Optional[concurrent.futures.ThreadPoolExecutor]) -> Optional[TurnOutput]:
        """Get result from background turn and cleanup executor."""
        if not future:
            return None
        result = future.result()
        if executor:
            executor.shutdown()
        return result

    def _check_game_ends_after_death(self, victim_name: str) -> bool:
        """Check if game would end after this player's death."""
        future_living = [p for p in self._get_living_players() if p.state.name != victim_name]
        future_mafia = sum(1 for p in future_living if p.state.role == "Mafia")
        future_town = sum(1 for p in future_living if p.state.role != "Mafia")
        return (future_mafia == 0) or (future_mafia >= future_town)

    def _collect_votes_concurrently(self, voters: List[Player], all_votes: dict, listener, nominees: List[str]):
        """Collect votes from all voters in up to 4 concurrent processes"""

        def collect_voter_vote(voter: Player):
            """Collect one voter's single vote for a nominee (MANDATORY)"""
            try:
                # Set phase to Trial for voting context (they're voting on nominees)
                output = voter.take_turn(self.state, self.state.turn)

                if output.strategy:
                    prefix = self._get_strategy_prefix(voter)
                    self._print(f"\n💭 {prefix}{voter.state.name} Strategy: {output.strategy}")

                # Get the player name they vote for - MANDATORY, no abstain
                vote = (output.vote or "").lower().strip()
                # Validate: must be a nominee, not just any living player
                valid_targets = [n.lower() for n in nominees]
                if vote and vote in valid_targets:
                    # Return with proper capitalization
                    target = next(n for n in nominees if n.lower() == vote)
                    return voter.state.name, target
                else:
                    # Invalid vote - default to first nominee
                    default_target = nominees[0]
                    self._print(f"[WARN] {voter.state.name} invalid vote '{vote}'. Defaulting to {default_target}")
                    return voter.state.name, default_target
            except Exception as e:
                self._print(f"Error voting for {voter.state.name}: {e}")
                default_target = nominees[0]
                return voter.state.name, default_target

        # Use ThreadPoolExecutor with max 4 workers for concurrent voting
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(collect_voter_vote, voter) for voter in voters]

            for future in concurrent.futures.as_completed(futures):
                voter_name, vote = future.result()
                all_votes[voter_name] = vote
                self.log("Trial", voter_name, "vote", f"votes for {vote}")

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
        town_count = sum(1 for p in self._get_living_players() if p.state.role != "Mafia")

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

                # Log speaking order BEFORE starting first speaker (so they see it)
                phase_flow = " -> Night" if self.state.turn == 1 else " -> Trial -> Night"
                self.log("Day", "System", "Info", f"Speaking order: {', '.join(p.state.name for p in ordered_living)}{phase_flow}")

                # Start first speaker's turn in background while announcement plays
                first_speaker_future, executor = self._start_background_turn(
                    ordered_living[0] if ordered_living else None
                )

                for i, player in enumerate(ordered_living):
                    # Double-check aliveness just in case state drifted
                    if not player.state.is_alive:
                        self._print(f"[DEBUG] Skipping dead player {player.state.name} in speaking order.")
                        continue

                    try:
                        # Use pre-generated output for first speaker if available
                        if i == 0 and first_speaker_future:
                            output = self._get_background_result(first_speaker_future, executor)
                        else:
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

                        self._print_strategy(player, output)

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

                    # Sort by day speaking order (not nomination count)
                    sorted_nominees = sorted(nominee_counts.items(), key=lambda x: next((i for i, p in enumerate(ordered_living) if p.state.name == x[0]), float('inf')))

                    nominee_display = [f"{n} ({c})" for n, c in sorted_nominees]
                    voter_count = len(self._get_living_players())
                    self.log("Trial", "System", "PhaseStart", f"Nominees: {', '.join(nominee_display)} | Voters: {voter_count}")
                    self._announce(f"Nominees for trial: {', '.join(nominee_display)}")

                    # --- DEFENSE PHASE ---
                    # All nominees speak their defense one after another
                    self.state.phase = "Trial"
                    accused_players = []

                    # Start first defendant's turn in background while announcements happen
                    first_future = None
                    executor = None
                    if sorted_nominees:
                        first_name = sorted_nominees[0][0]
                        first_accused = self.active_players.get(first_name)
                        if first_accused and first_accused.state.is_alive:
                            self.state.on_trial = first_name
                            # Announce first defendant BEFORE starting their turn
                            self._print(f"\n⚖️  {first_name} speaks for defense ⚖️")
                            # NOW start their turn in background while announcements play
                            first_future, executor = self._start_background_turn(first_accused)

                    for i, (accused_name, nom_count) in enumerate(sorted_nominees):
                        accused = self.active_players.get(accused_name)
                        if not accused or not accused.state.is_alive:
                            self._print(f"[DEBUG] Skipping defense for {accused_name} (Dead or invalid).")
                            continue

                        accused_players.append(accused)
                        self.state.on_trial = accused_name

                        # For non-first defendants, announce here
                        if i > 0:
                            self._print(f"\n⚖️  {accused_name} speaks for defense ⚖️")

                        # Defense - accused speaks
                        try:
                            # Use pre-generated output for first defendant if available
                            if i == 0 and first_future:
                                output = self._get_background_result(first_future, executor)
                            else:
                                output = accused.take_turn(self.state, self.state.turn)

                            # Prepare TTS immediately (before waiting for previous)
                            speech = output.speech or ""
                            audio_path = None
                            if speech:
                                audio_path = self.tts.prepare_speech(speech, accused_name, announce_name=True)

                            # Wait for previous TTS while current is being prepared
                            self._wait_for_speech_with_pause(listener)

                            self._print_strategy(accused, output)
                            self.log("Trial", accused_name, "speak", f"[Defense] {speech}")

                            # Play current TTS in background (next defendant can prepare while this plays)
                            if audio_path:
                                self.tts.play_file(audio_path, background=True)
                        except Exception as e:
                            self._print(f"Error in defense: {e}")

                        self._wait_for_next(listener)

                    # --- CONCURRENT VOTING PHASE ---
                    self.tts.wait_for_speech()
                    self.state.on_trial = None  # Clear so defendants get voter prompt, not defend prompt
                    self._print(f"\n🗳️  VOTING TIME 🗳️")
                    self._announce(f"Voting time.")

                    # Get all living voters
                    voters = self._get_living_players()

                    # Dictionary to store votes: {voter_name: target_player_name}
                    all_votes = {}

                    # Extract nominee names from sorted_nominees (list of (name, count) tuples)
                    nominee_names = [name for name, count in sorted_nominees]

                    # Use concurrent voting with up to 4 processes
                    self._collect_votes_concurrently(voters, all_votes, listener, nominee_names)

                    self._wait_for_next(listener)

                    # --- RESULTS PHASE ---
                    self.tts.wait_for_speech()

                    # Tally votes by target player (no abstains - all votes count)
                    vote_tally = {}  # target_player: [voters who voted for them]

                    for voter_name, target in all_votes.items():
                        if target not in vote_tally:
                            vote_tally[target] = []
                        vote_tally[target].append(voter_name)

                    # Determine who gets eliminated FIRST (before announcing)
                    kill_targets = []
                    trial_last_words_future = None
                    trial_lw_executor = None
                    trial_victim = None
                    trial_game_ends = False

                    if vote_tally:
                        max_votes = max(len(v) for v in vote_tally.values())
                        most_voted = [k for k, v in vote_tally.items() if len(v) == max_votes]
                        kill_targets = most_voted

                        # For first victim, start last words in background while results TTS plays
                        if kill_targets:
                            first_target = kill_targets[0]
                            trial_victim = self.active_players.get(first_target)
                            if trial_victim and trial_victim.state.is_alive:
                                # Check if game ends
                                future_living = [p for p in self._get_living_players() if p.state.name != first_target]
                                future_mafia = sum(1 for p in future_living if p.state.role == "Mafia")
                                future_town = sum(1 for p in future_living if p.state.role != "Mafia")
                                trial_game_ends = (future_mafia == 0) or (future_mafia >= future_town)

                                if not trial_game_ends:
                                    # Mark dead, set phase, add to logs so victim knows
                                    trial_victim.state.is_alive = False
                                    self.state.phase = "LastWords"

                                    # Determine role emoji
                                    trial_role_emoji = self._get_role_emoji(trial_victim.state.role)

                                    # Add death to public logs so victim sees it (role reveal logged later)
                                    self.state.public_logs.append(LogEntry(
                                        turn=self.state.turn, phase="Trial", actor="System",
                                        action="Death", content=f"{first_target} was voted out"
                                    ))

                                    # Start last words in background
                                    trial_last_words_future, trial_lw_executor = self._start_background_turn(trial_victim)

                    # NOW announce results (TTS plays while last words generates)
                    self._print(f"\n📊 VOTING RESULTS 📊")
                    for accused in accused_players:
                        accused_name = accused.state.name
                        votes_for_this = vote_tally.get(accused_name, [])
                        vote_count = len(votes_for_this)
                        voters_str = ", ".join(votes_for_this) if votes_for_this else "None"
                        result_msg = f"{accused_name} - {vote_count} votes from ({voters_str})"
                        self.log("Trial", "System", "VoteSummary", result_msg)
                        self._announce(f"{accused_name} received {vote_count} votes")

                    if len(kill_targets) > 1:
                        self.log("Trial", "System", "TieBreak", f"Tie between {kill_targets}. All are eliminated!")
                        self._announce(f"Tie! {', '.join(kill_targets)} eliminated.")

                    # Wait for results TTS to finish
                    self.tts.wait_for_speech()

                    # Process eliminations
                    someone_died = False
                    for i, kill_target in enumerate(kill_targets):
                        victim = self.active_players.get(kill_target)
                        if not victim:
                            continue

                        # First victim uses pre-generated last words
                        if i == 0 and trial_last_words_future and not trial_game_ends:
                            try:
                                output = self._get_background_result(trial_last_words_future, trial_lw_executor)

                                speech = output.speech or ""
                                audio_path = None
                                if speech:
                                    audio_path = self.tts.prepare_speech(speech, kill_target, announce_name=True)

                                self._announce(f"{kill_target}, last words.")

                                self._print_strategy(victim, output)

                                self.log("LastWords", kill_target, "speak", f"[Last Words] {speech}")

                                if audio_path:
                                    self.tts.play_file(audio_path, background=True)

                                self._wait_for_next(listener)
                                self.tts.wait_for_speech()

                            except Exception as e:
                                self._print(f"Error in last words: {e}")

                            self.state.phase = "Trial"

                        elif i > 0 and victim.state.is_alive:
                            # Additional victims in tie - generate last words normally
                            future_living = [p for p in self._get_living_players() if p.state.name != kill_target]
                            future_mafia = sum(1 for p in future_living if p.state.role == "Mafia")
                            future_town = sum(1 for p in future_living if p.state.role != "Mafia")
                            game_ends = (future_mafia == 0) or (future_mafia >= future_town)

                            if not game_ends:
                                self.state.phase = "LastWords"
                                victim.state.is_alive = False
                                self._announce(f"{kill_target}, last words.")
                                try:
                                    output = victim.take_turn(self.state, self.state.turn)
                                    speech = output.speech or ""
                                    audio_path = None
                                    if speech:
                                        audio_path = self.tts.prepare_speech(speech, kill_target, announce_name=True)

                                    self._wait_for_speech_with_pause(listener)

                                    self._print_strategy(victim, output)

                                    self.log("LastWords", kill_target, "speak", f"[Last Words] {speech}")

                                    if audio_path:
                                        self.tts.play_file(audio_path, background=True)

                                except Exception as e:
                                    self._print(f"Error in last words: {e}")

                                self._wait_for_next(listener)
                                self.state.phase = "Trial"

                        # Mark victim as dead (always, even if game ends)
                        victim.state.is_alive = False

                        self.log("Result", "System", "Death", f"{kill_target} eliminated.")
                        self._announce(f"{kill_target} eliminated.")
                        if self.state.reveal_role_on_death:
                            role_emoji = self._get_role_emoji(victim.state.role)
                            # Role already in public_logs from earlier - just log for display
                            self.log("Result", "System", "RoleReveal", f"{role_emoji} {kill_target} was a {victim.state.role}!")
                        someone_died = True

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

                    # Start first Mafia's turn in background while announcement plays
                    first_mafia_future, executor = self._start_background_turn(
                        mafia_alive[0] if mafia_alive else None
                    )

                    night_victim = None
                    mafia_votes = {}
                    for i, m_player in enumerate(mafia_alive):
                        if not m_player.state.is_alive:
                             continue
                        try:
                            # Use pre-generated output for first Mafia if available
                            if i == 0 and first_mafia_future:
                                output = self._get_background_result(first_mafia_future, executor)
                            else:
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

                            self._print_strategy(m_player, output)
                            
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
                            kill_target = random.choice(winners)
                            self.log("Night", "System", "TieBreak", f"Tie {winners}. Random: {kill_target}", is_secret=True, target_log="Mafia")
                        else:
                            kill_target = winners[0]
    
                        # Execute Kill (Secretly for now)
                        if kill_target in self.active_players:
                            victim = self.active_players[kill_target]
                            if victim.state.is_alive:
                                # DELAY DEATH: Don't set is_alive=False yet, so Cop doesn't see it
                                # victim.state.is_alive = False 
                                night_victim = kill_target
                                self.log("Night", "System", "Kill", f"Mafia chosen target: {kill_target}", is_secret=True, target_log="Mafia")
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

                            self._print_strategy(cop, output)

                            # Log action
                            action_desc = f"[Investigates {target_name}]" if target_name else "[No investigation]"
                            self.log("Night", cop.state.name, "investigate", f"{action_desc} {output.speech or ''}", is_secret=True, target_log="Cop")
                            
                            # Play Audio
                            if audio_path:
                                self.tts.play_file(audio_path, background=True)

                            # Resolve Investigation (even if cop is killed tonight)
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
                        
                        # Don't wait here - start last words prompt immediately while TTS plays

                    # --- START LAST WORDS PROMPT IN BACKGROUND (while Cop TTS plays) ---
                    night_audio_path = None
                    night_victim_speech = ""
                    night_victim_strategy = None
                    last_words_future = None
                    lw_executor = None
                    death_role_emoji = ""

                    if night_victim and night_victim in self.active_players:
                        victim = self.active_players[night_victim]
                        if victim.state.is_alive:
                            # Mark dead and set phase for prompt context
                            victim.state.is_alive = False
                            self.state.phase = "LastWords"

                            # Determine role emoji
                            death_role_emoji = self._get_role_emoji(victim.state.role)

                            # Add death to public log BEFORE prompt (so victim sees it) - no print yet
                            self.state.public_logs.append(LogEntry(
                                turn=self.state.turn, phase="Night", actor="System",
                                action="Death", content=f"{night_victim} was killed by Mafia"
                            ))
                            if self.state.reveal_role_on_death:
                                self.state.public_logs.append(LogEntry(
                                    turn=self.state.turn, phase="Night", actor="System",
                                    action="RoleReveal", content=f"{death_role_emoji} {night_victim} was a {victim.state.role}"
                                ))

                            # Start last words prompt in background (while Cop TTS still playing)
                            last_words_future, lw_executor = self._start_background_turn(victim)

                    # NOW wait for Cop's TTS to finish before revealing death
                    self.tts.wait_for_speech()

                    # --- APPLY MAFIA KILL (Secret log) ---
                    if night_victim and night_victim in self.active_players:
                        victim = self.active_players[night_victim]
                        self.log("Night", "System", "Kill", f"Mafia killed {night_victim}", is_secret=True, target_log="Mafia")

                        # Get last words result (should be ready or almost ready)
                        if last_words_future:
                            try:
                                output = self._get_background_result(last_words_future, lw_executor)
                                night_victim_speech = output.speech or ""
                                night_victim_strategy = output.strategy
                                if night_victim_speech:
                                    night_audio_path = self.tts.prepare_speech(night_victim_speech, night_victim, announce_name=True)
                            except Exception as e:
                                self._print(f"Error preparing night last words: {e}")

                    # --- DEATH REVEAL (Morning) ---
                    if night_victim:
                         victim = self.active_players[night_victim]
                         self._print(f"\n🩸 TRAGEDY! {night_victim} was found DEAD in the morning.🩸")
                         if self.state.reveal_role_on_death:
                             self._print(f"{death_role_emoji} {night_victim} was a {victim.state.role}!")
                         self._announce(f"{night_victim} was killed during the night")

                         # --- LAST WORDS FOR NIGHT VICTIM (Already prepared) ---
                         self._announce(f"{night_victim}, last words.")

                         if night_victim_strategy:
                             prefix = self._get_strategy_prefix(victim)
                             self._print(f"\n💭 {prefix}{night_victim} Strategy: {night_victim_strategy}")

                         self.log("LastWords", night_victim, "speak", f"[Last Words] {night_victim_speech}")

                         if night_audio_path:
                             self.tts.play_file(night_audio_path, background=True)

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
