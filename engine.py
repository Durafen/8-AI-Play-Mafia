import random
import asyncio
from typing import List, Dict
from models import Player
from api_clients import UnifiedLLMClient
from schemas import GameState, LogEntry, TurnOutput

# Config for Roster
# Config for Roster
ROSTER_CONFIG = [
    # OPENAI
    {"name": "GPT 5.2", "provider": "openai", "model": "gpt-5.2"},
    {"name": "GPT 5.1", "provider": "openai", "model": "gpt-5.1"},
    
    # ANTHROPIC
    {"name": "Haiku", "provider": "anthropic", "model": "haiku"}, 
    {"name": "Sonnet", "provider": "anthropic", "model": "sonnet"},

    # GOOGLE
    {"name": "Gemini 2.5 Pro", "provider": "google", "model": "gemini-2.5-pro"},
    {"name": "Gemini 2.5 Flash", "provider": "google", "model": "gemini-2.5-flash"},
    {"name": "Gemini 3 Flash", "provider": "google", "model": "gemini-3-flash-preview"},

    # GROQ (Qwen)
    {"name": "Qwen Coder", "provider": "groq", "model": "coder-model"}, 
]

import shutil
import os

class GameEngine:
    def __init__(self):
        # Clean Logs
        if os.path.exists("logs"):
            shutil.rmtree("logs")
        os.makedirs("logs", exist_ok=True)
        
        self.client = UnifiedLLMClient(debug=True)
        self.state = GameState()
        self.players: List[Player] = []
        self.active_players: Dict[str, Player] = {} # Name -> Player obj



    def log(self, phase: str, actor: str, action: str, content: str, is_secret: bool = False, vote_target: str = None):
        # Determine display name with number if actor is a player
        actor_display = actor
        player = next((p for p in self.players if p.state.name == actor), None)
        if player:
            idx = self.players.index(player) + 1
            actor_display = f"{idx}. {actor}"
            
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
        elif phase in ["Voting", "Defense", "LastWords", "Setup"]:
             display_icon = phase_icon
             
        # Rule 4: Day/Night System messages (Info, etc) -> NO ICON
        # This prevents the "sun/moon on every line" issue.
        
        if is_secret:
            self.state.mafia_logs.append(entry)
            display_content = content.replace("[Nominated", "[👉 Nominated").replace("[Voted for", "[🗳️ Voted for").replace("[Suggests killing", "[🔪 Suggests killing").replace("[Defense]", "[🛡️ Defense]").replace("[Last Words]", "[💀 Last Words]")
            print(f"\n{display_icon}{actor_display} {vote_str} {display_content}")
        else:
            self.state.public_logs.append(entry)
            display_content = content.replace("[Nominated", "[👉 Nominated").replace("[Voted for", "[🗳️ Voted for").replace("[Suggests killing", "[🔪 Suggests killing").replace("[Defense]", "[🛡️ Defense]").replace("[Last Words]", "[💀 Last Words]")
            print(f"\n{display_icon}{actor_display} {vote_str} {display_content}")

    def setup_game(self):
        print("Initializing Game...")
        
        # 1. Randomize Roster Order
        roster = list(ROSTER_CONFIG)
        random.shuffle(roster)

        # 2. Assign Roles (2 Mafia, 6 Villagers)
        # Since roster is shuffled, we can just pick indices 0 and 1 for Mafia? 
        # Or shuffle roles separately? 
        # Let's shuffle indices to be safe/explicit.
        indices = list(range(8))
        # random.shuffle(indices) -> actually we just need 2 random indices
        mafia_indices = set(random.sample(range(8), 2))

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
                player_index=i+1
            )
            self.players.append(p)
            self.state.players.append(p.state)
            self.active_players[p.state.name] = p
            if role == "Mafia":
                mafia_names.append(p.state.name)

        # Introduce Partners
        for p in self.players:
            if p.state.role == "Mafia":
                partner = [n for n in mafia_names if n != p.state.name]
                if partner:
                    p.set_partner(partner[0])

        print(f"[System] Game Initialized. Mafia are: {', '.join(mafia_names)}")
        self.log("Setup", "System", "GameStart", "Game Initialized. Roles have been assigned.")
        self.log("Setup", "System", "MafiaReveal", f"The Mafia team is: {', '.join(mafia_names)}", is_secret=True)

    def _get_living_players(self) -> List[Player]:
        return [p for p in self.players if p.state.is_alive]

    def _wait_for_next(self):
        input("\n[PRESS ENTER TO CONTINUE NEXT ACTION] >> ")

    def run(self):
        self.setup_game()
        
        while True:
            # Check Win Condition
            mafia_count = sum(1 for p in self._get_living_players() if p.state.role == "Mafia")
            town_count = sum(1 for p in self._get_living_players() if p.state.role == "Villager")
            
            if mafia_count == 0:
                print("\n🎉 TOWN WINS! All Mafia eliminated. 🎉")
                break
            if mafia_count >= town_count:
                print("\n💀 MAFIA WINS! They have parity with Town. 💀")
                break

            # --- DAY PHASE ---
            self.state.phase = "Day"

            self.log("Day", "System", "PhaseStart", f"--- DAY {self.state.turn} START ---")
            
            # Skip speech loop on Day 1? No, user said "Day 1... No voting". implied speech OK.
            # "Day cycle... speak... move to next... then vote."
            
            # 1. Speaking Round
            living = self._get_living_players()
            # Track nominations (suggestions)
            nominations = {} # PlayerName -> VoteTarget

            self.log("Day", "System", "Info", f"Alive: {', '.join(p.state.name for p in living)}")

            for player in living:
                self._wait_for_next()
                try:
                    output = player.take_turn(self.state, self.state.turn)
                    # Print Thought to Terminal
                    prefix = "👺 " if player.state.role == "Mafia" else ""
                    print(f"\n💭 {prefix}{player.state.name} Thinking: {output.thought}")
                    
                    # Construct content with bracketed action if present
                    speech = output.speech or ""
                    action_part = ""

                    # Capture nomination if present (Day 2+)
                    if self.state.turn > 1 and output.vote:
                         nominations[player.state.name] = output.vote
                         action_part = f"[Nominated {output.vote}] "

                    content = f"{action_part}{speech}"
                    self.log("Day", player.state.name, "speak", content)

                except Exception as e:
                    self.log("Day", player.state.name, "error", f"Failed to speak: {e}")

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
                    self.state.phase = "Defense"
                    print(f"\n🛡️  DEFENSE PHASE 🛡️")
                    
                    nominee_display = [f"{n} ({nominee_counts[n]})" for n in nominees]
                    self.log("Defense", "System", "PhaseStart", f"Nominees for elimination: {', '.join(nominee_display)}")
                    
                    for nom_name in nominees:
                        # Find player object
                        nom_player = next((p for p in living if p.state.name == nom_name), None)
                        if not nom_player: continue
                        
                        self._wait_for_next()
                        try:
                            # Defense Turn
                            output = nom_player.take_turn(self.state, self.state.turn)
                            print(f"\n💭 {nom_player.state.name} Defending: {output.thought}")
                            self.log("Defense", nom_player.state.name, "speak", f"[Defense] {output.speech or ''}")
                        except Exception as e:
                            print(f"Error defending: {e}")
                else:
                    self.log("Day", "System", "Info", "No valid nominations. Skipping Defense.")

                # --- VOTING PHASE ---
                self.state.phase = "Voting"
                print("\n🗳️  VOTING TIME 🗳️")
                
                final_votes = {} # PlayerName -> TargetName

                for player in living:
                    # Everyone votes fresh
                    self._wait_for_next()
                    try:
                        output = player.take_turn(self.state, self.state.turn)
                        vote_target = output.vote
                        
                        # Print Thought
                        prefix = "👺 " if player.state.role == "Mafia" else ""
                        print(f"\n💭 {prefix}{player.state.name} Thinking: {output.thought}")
                        
                        # Validate vote
                        # MUST be in nominees list (if nominees exist)
                        if nominees and vote_target not in nominees:
                            print(f"[Invalid Vote] {player.state.name} voted for {vote_target} (Not a nominee)")
                            vote_target = "Skip" 
                        elif not nominees and vote_target not in [p.state.name for p in living]:
                             # Fallback if no nominees (shouldn't happen due to logic above skipping phase, but safety)
                             vote_target = "Skip"

                        final_votes[player.state.name] = vote_target
                        
                        # Force silence in log
                        self.log("Voting", player.state.name, "vote", f"[Voted for {vote_target}]")
                        
                    except Exception as e:
                        print(f"Error voting: {e}")
                        final_votes[player.state.name] = "Skip"

                # Aggregate Tally from final_votes
                votes = {}
                for target in final_votes.values():
                    if target != "Skip":
                        votes[target] = votes.get(target, 0) + 1

                # Tally Summary
                tally_parts = [f"{k} ({v})" for k,v in votes.items()]
                tally_str = ", ".join(tally_parts) if tally_parts else "No votes"
                self.log("Voting", "System", "VoteSummary", f"Votes Cast: {tally_str}")

                if not votes:
                    self.log("Result", "System", "NoLynch", "No votes cast.")
                else:
                    target, count = max(votes.items(), key=lambda x: x[1])
                    max_votes = count
                    # Identify all players with max votes
                    victims_names = [k for k, v in votes.items() if v == max_votes]
                    
                    if len(victims_names) > 1:
                        self.log("Result", "System", "Tie", f"Tie between {', '.join(victims_names)}. ALL will be eliminated!")
                    
                    # Process deaths
                    for v_name in victims_names:
                        victim = self.active_players.get(v_name)
                        if not victim or not victim.state.is_alive: continue
                        
                        # --- LAST WORDS PHASE ---
                        self.state.phase = "LastWords"
                        print(f"\n💀 {v_name} - LAST WORDS 💀")
                        try:
                            self._wait_for_next()
                            output = victim.take_turn(self.state, self.state.turn)
                            print(f"\n💭 {victim.state.name} LastWords: {output.thought}")
                            self.log("LastWords", victim.state.name, "speak", f"[Last Words] {output.speech or ''}")
                        except Exception as e:
                            self.log("LastWords", victim.state.name, "error", f"Failed to speak last words: {e}")

                        # Execute Kill
                        victim.state.is_alive = False
                        self.log("Result", "System", "Death", f"{v_name} was HANGED by the town!")
                        print(f"💀💀💀 {v_name} IS DEAD 💀💀💀")
                        self.log("Result", "System", "Info", f"{v_name} is dead.")

            self._wait_for_next()

            # Check Win again before Night
            # ... (Implicitly handled at loop start)

            # --- NIGHT PHASE ---
            self.state.phase = "Night"
            self.log("Night", "System", "PhaseStart", f"--- NIGHT {self.state.turn} START ---")

            
            mafia_alive = [p for p in self._get_living_players() if p.state.role == "Mafia"]
            
            if not mafia_alive:
                # Game over loop will catch this next iter
                pass
            else:
                self.log("Night", "System", "MafiaWake", "The Mafia wakes up...", is_secret=True)
                
                mafia_votes = {}
                for m_player in mafia_alive:
                    self._wait_for_next() # Admin steps through night too?
                    try:
                        # Mafia player "votes" for kill
                        output = m_player.take_turn(self.state, self.state.turn)
                        
                        print(f"\n💭 👺 {m_player.state.name} (Mafia) Thinking: {output.thought}")

                        target = output.vote
                        action_tag = f"[Suggests killing {target}] " if target else ""
                        content = f"{action_tag}{output.speech or ''}"
                        self.log("Night", m_player.state.name, "whisper", content, is_secret=True)
                        if target:
                            mafia_votes[target] = mafia_votes.get(target, 0) + 1
                    except Exception as e:
                        print(f"Mafia Error: {e}")

                # Consensus Summary
                tally_parts = [f"{k} ({v})" for k,v in mafia_votes.items()]
                tally_str = ", ".join(tally_parts) if tally_parts else "No votes"
                self.log("Night", "System", "VoteSummary", f"Mafia Votes: {tally_str}", is_secret=True)

                if mafia_votes:
                    max_votes = max(mafia_votes.values())
                    winners = [k for k, v in mafia_votes.items() if v == max_votes]
                    
                    if len(winners) > 1:
                        import random
                        kill_target = random.choice(winners)
                        self.log("Night", "System", "TieBreak", f"Mafia Tie {winners}. Randomly chose: {kill_target}", is_secret=True)
                    else:
                        kill_target = winners[0]

                    # Execute Kill
                    if kill_target in self.active_players and self.active_players[kill_target].state.is_alive:
                        victim = self.active_players[kill_target]
                        victim.state.is_alive = False
                        self.log("Night", "System", "Kill", f"The Mafia killed {kill_target}")
                        print(f"\n🩸 TRAGEDY! {kill_target} was found DEAD in the morning.🩸")
                    else:
                         self.log("Night", "System", "Fail", "Mafia targeted invalid player.")
                else:
                     self.log("Night", "System", "Quiet", "Mafia did not kill anyone.")

            self._wait_for_next()
            self.state.turn += 1

if __name__ == "__main__":
    engine = GameEngine()
    engine.run()
