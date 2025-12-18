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

    def log(self, phase: str, actor: str, action: str, content: str, is_secret: bool = False):
        # Determine display name with number if actor is a player
        actor_display = actor
        player = next((p for p in self.players if p.state.name == actor), None)
        if player:
            idx = self.players.index(player) + 1
            actor_display = f"{idx}. {actor}"
            
            # Add Mafia Icon for terminal display only
            if player.state.role == "Mafia":
                actor_display = f"👺 {actor_display}"

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

        if is_secret:
            self.state.mafia_logs.append(entry)
            print(f"\n[SECRET] {actor_display}  {content}")
        else:
            self.state.public_logs.append(entry)
            print(f"\n{actor_display}  {content}")

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
            print(f"\n☀️  DAY {self.state.turn} STARTS ☀️")
            
            # Skip speech loop on Day 1? No, user said "Day 1... No voting". implied speech OK.
            # "Day cycle... speak... move to next... then vote."
            
            # 1. Speaking Round
            living = self._get_living_players()
            # Order stays fixed (ROSTER order) per user request
            
            self.log("Day", "System", "Info", f"Alive: {', '.join(p.state.name for p in living)}")

            for player in living:
                self._wait_for_next()
                try:
                    output = player.take_turn(self.state, self.state.turn)
                    # Print Thought to Terminal
                    prefix = "👺 " if player.state.role == "Mafia" else ""
                    print(f"\n💭 {prefix}{player.state.name} Thinking: {output.thought}")
                    
                    self.log("Day", player.state.name, "speak", output.speech)
                except Exception as e:
                    self.log("Day", player.state.name, "error", f"Failed to speak: {e}")

            # 2. Voting Round (Skip on Day 1)
            if self.state.turn > 1:
                self.state.phase = "Voting"
                print("\n🗳️  VOTING TIME 🗳️")
                votes = {} # target -> count
                
                for player in living:
                    self._wait_for_next()
                    try:
                        output = player.take_turn(self.state, self.state.turn)
                        vote_target = output.vote
                        
                        # Print Thought
                        prefix = "👺 " if player.state.role == "Mafia" else ""
                        print(f"\n💭 {prefix}{player.state.name} Thinking: {output.thought}")
                        
                        # Validate vote
                        if vote_target not in [p.state.name for p in living]:
                            vote_target = "Skip" # Invalid vote
                        
                        self.log("Voting", player.state.name, "vote", f"Voted for {vote_target} | Reason: {output.speech}")
                        
                        if vote_target != "Skip":
                            votes[vote_target] = votes.get(vote_target, 0) + 1
                    except Exception as e:
                        print(f"Error voting: {e}")

                # Tally
                if not votes:
                    self.log("Result", "System", "NoLynch", "No votes cast.")
                else:
                    target, count = max(votes.items(), key=lambda x: x[1])
                    # Check Tie
                    max_votes = count
                    winners = [k for k, v in votes.items() if v == max_votes]
                    
                    if len(winners) > 1:
                        self.log("Result", "System", "Tie", f"Tie between {winners}. No one dies.")
                    else:
                        # KILL
                        eliminated = self.active_players[target]
                        eliminated.state.is_alive = False
                        self.log("Result", "System", "Death", f"{target} was HANGED by the town!")
                        print(f"💀💀💀 {target} IS DEAD 💀💀💀")
                        # Check role reveal?
                        self.log("Result", "System", "Reveal", f"{target} was {eliminated.state.role}")

            # Check Win again before Night
            # ... (Implicitly handled at loop start)

            # --- NIGHT PHASE ---
            self.state.phase = "Night"
            print(f"\n🌙 NIGHT {self.state.turn} FALLS 🌙")
            
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
                        self.log("Night", m_player.state.name, "whisper", f"Suggests killing {target}: {output.speech}", is_secret=True)
                        if target:
                            mafia_votes[target] = mafia_votes.get(target, 0) + 1
                    except Exception as e:
                        print(f"Mafia Error: {e}")

                # Consensus
                if mafia_votes:
                    kill_target, _ = max(mafia_votes.items(), key=lambda x: x[1])
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

            self.state.turn += 1

if __name__ == "__main__":
    engine = GameEngine()
    engine.run()
