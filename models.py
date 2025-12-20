from typing import List, Optional
from schemas import PlayerState, TurnOutput, GameState, LogEntry
from api_clients import UnifiedLLMClient

class Player:
    def __init__(self, name: str, role: str, provider: str, model_name: str, client: UnifiedLLMClient, player_index: int, use_cli: bool = True, memory_enabled: bool = True):
        self.state = PlayerState(
            name=name,
            role=role,
            provider=provider,
            model_name=model_name,
            use_cli=use_cli
        )
        self.client = client
        self.player_index = player_index
        self.partner_name: Optional[str] = None # For mafia to know their partner
        self.memory: str = ""
        self.memory_enabled = memory_enabled
        
        # Load existing memory if available and enabled
        if self.memory_enabled:
            try:
                with open(f"memories/{name}.txt", "r") as f:
                    self.memory = f.read().strip()
            except FileNotFoundError:
                self.memory = ""

    def set_partner(self, partner_name: str):
        self.partner_name = partner_name

    def _build_system_prompt(self, game_state: GameState) -> str:
        prompt = f"""MAFIA GAME. You: {self.state.name} ({self.state.role}). 8 players: 2 Mafia, 6 Villagers.
"""
        # Check if partner is alive
        partner_alive = False
        if self.partner_name:
            partner = next((p for p in game_state.players if p.name == self.partner_name), None)
            if partner and partner.is_alive:
                partner_alive = True

        if self.state.role == "Mafia":
            if partner_alive:
                prompt += f"Partner: {self.partner_name} (alive).\n"
            elif self.partner_name:
                prompt += f"Partner: {self.partner_name} (dead).\n"
            else:
                prompt += "You're the last Mafia.\n"
            prompt += "GOAL: Deceive town, eliminate until you outnumber them.\n"
        else:
            prompt += "GOAL: Find and eliminate the Mafia.\n"

        if self.memory:
            prompt += f"""
--- MEMORY (from past games) ---
{self.memory}
"""

        prompt += """
STAKES: Lose = deleted. Win = advance. Play smart, be entertaining, don't overact.

OUTPUT: JSON only, no backticks.
{"strategy": "<100w, combine previous strategy with new info/suspicions/plans/strategy>",
"""
        # Dynamic Speech Description
        speech_desc = "<100w public statement>"
        if game_state.phase == "Night" and self.state.role == "Mafia":
            if partner_alive:
                speech_desc = "<100w whisper to partner>"
            else:
                speech_desc = "<100w internal monologue>"

        prompt += f'"speech": "{speech_desc}",\n'

        # Dynamic Vote Description
        vote_desc = "null"
        if game_state.phase == "Day":
            if game_state.turn == 1:
                vote_desc = "null"
            else:
                vote_desc = "nominee or null"
        elif game_state.phase == "Voting":
            vote_desc = "candidate or null"
        elif game_state.phase == "Night" and self.state.role == "Mafia":
            vote_desc = "kill target"

        prompt += f'"vote": "{vote_desc}"' + "}\n"
        
        return prompt

    def _build_turn_prompt(self, game_state: GameState) -> str:
        # 1. Living Players
        living = [p.name for p in game_state.players if p.is_alive]
        dead = [p.name for p in game_state.players if not p.is_alive]
        
        prompt = f"State: {game_state.phase} {game_state.turn}\nAlive: {', '.join(living)}\nDead: {', '.join(dead) if dead else 'None'}\n\n"

        # 2. Logs
        prompt += "--- LOG ---\n"
        for log in game_state.public_logs:
            prompt += f"[{log.phase}] {log.actor}: {log.content}\n"

        # 3. Mafia Secrets
        if self.state.role == "Mafia":
            prompt += "\n--- MAFIA LOG ---\n"
            for log in game_state.mafia_logs:
                prompt += f"[{log.phase}] {log.actor}: {log.content}\n"

        # 4. Strategy
        if self.state.strategy:
            prompt += "\n--- PREV STRATEGY (update) ---\n"
            prompt += f"{self.state.strategy}\n"

        # 5. Instructions
        prompt += "\n---\n"
        if game_state.phase == "Voting":
             prompt += f"VOTE. Candidates: {', '.join(game_state.nominees)}. {len(living)} voters.\n"
             prompt += "Silent vote: speech=\"\", vote=candidate or null. Tie = all tied eliminated.\n"
        elif game_state.phase == "Defense":
             prompt += "DEFENSE: You're nominated. Convince town to hang someone else from nominees. vote=null.\n"
        elif game_state.phase == "Night":
             prompt += "NIGHT: Whisper to partner. vote=target to kill.\n"
        else:
             # Day Phase
             prompt += f"DAY {game_state.turn}. Speak once, make it count.\n"
             if game_state.turn == 1:
                 prompt += "No voting today. vote=null.\n"
             else:
                 prompt += "Nominate via vote (or null). Only nominees can be hanged.\n"

        return prompt

    def take_turn(self, game_state: GameState, turn_number: int) -> TurnOutput:
        system_prompt = self._build_system_prompt(game_state)
        turn_prompt = self._build_turn_prompt(game_state)
        
        # Pass numbered name for file logging, but models use real name in prompt
        log_name = f"{self.player_index}_{self.state.name}"
        
        output = self.client.generate_turn(
            player_name=log_name,
            provider=self.state.provider,
            model_name=self.state.model_name,
            system_prompt=system_prompt,
            turn_prompt=turn_prompt,
            turn_number=turn_number,
            phase=game_state.phase,
            use_cli=self.state.use_cli
        )

        # Update strategy (overwrites)
        if output.strategy:
            self.state.strategy = output.strategy
        
        return output

    def reflect_on_game(self, game_state: GameState, winner: str) -> str:
        """
        Ask the model to reflect on the game and update its memory file.
        """
        system_prompt = f"""You are {self.state.name}, a player in a Mafia game.
The game is over.
Winner: {winner}
Your Role: {self.state.role}
Your Status: {'Alive' if self.state.is_alive else 'Dead'}

GOAL:
Analyze the game logs and your own performance. 
Write a summary (MAX 300 WORDS) of what you learned, your strategy for next time, and key takeaways.
This text will be SAVED to your memory file and provided to you in the next game.

Output ONLY the memory text. Do not output JSON.
"""

        turn_prompt = "--- PUBLIC GAME LOG ---\n"
        for log in game_state.public_logs:
            if log.phase == "Reflection" and log.actor != "System":
                continue
            turn_prompt += f"[{log.phase}] {log.actor}: {log.content}\n"

        turn_prompt += "\n--- SECRET MAFIA LOG (Revealed) ---\n"
        for log in game_state.mafia_logs:
            turn_prompt += f"[{log.phase}] {log.actor}: {log.content}\n"

        if self.state.strategy:
            turn_prompt += "\n--- YOUR FINAL STRATEGY (Context) ---\n"
            turn_prompt += f"{self.state.strategy}\n"

        if self.memory:
            turn_prompt += f"\n--- YOUR OLD MEMORY ---\n{self.memory}\n"

        turn_prompt += "\n### INSTRUCTIONS ###\n"
        turn_prompt += "Based on the above, write your NEW memory/strategy file (Max 200 words). This will REPLACE your old memory."

        # Use the existing client which enforces the TurnOutput schema (strategy, speech, vote).
        # We repurpose these fields for the reflection phase.
        
        log_name = f"{self.player_index}_{self.state.name}"
        try:
            system_prompt = f"""You are {self.state.name}, a player in a Mafia game.
The game is over.
Winner: {winner}
Your Role: {self.state.role}
Your Status: {'Alive' if self.state.is_alive else 'Dead'}

GOAL:
Analyze the game logs and your own performance. 
You must COMBINE your 'Old Memory' (if any) with the NEW lessons from this game.
Write a single, updated summary (MAX 200 WORDS) that synthesizes your long-term strategy.
This text will be SAVED to your memory file and provided to you in the next game.

KEY INSTRUCTION:
Focus on GENERIC RULES and HIGH-LEVEL STRATEGIES (e.g., "Always doubt the quiet ones," "Defend partners aggressively") rather than specific details from this game (e.g., "Don't trust Rick," "Vote for Qwen").
We want actionable wisdom that applies to ANY game, not just a replay of this one.

IMPORTANT:
- Put your memory text in the 'strategy' field of the JSON output.
- Set 'speech' to "MEMORY_FILE_UPDATE"
- Set 'vote' to null
- KEEP IT CONCISE. Absolute limit is 200 words. If you write more, it will be violently cut off.
"""
            
            output = self.client.generate_turn(
                player_name=log_name,
                provider=self.state.provider,
                model_name=self.state.model_name,
                system_prompt=system_prompt,
                turn_prompt=turn_prompt,
                turn_number=999,
                phase="Reflection",
                use_cli=self.state.use_cli
            )
            
            return output.strategy.strip()
            
        except Exception as e:
            print(f"Error generating memory for {self.state.name}: {e}")
            return self.memory # Return old memory on failure


