from typing import List, Optional
from schemas import PlayerState, TurnOutput, GameState, LogEntry
from api_clients import UnifiedLLMClient

class Player:
    def __init__(self, name: str, role: str, provider: str, model_name: str, client: UnifiedLLMClient, player_index: int, memory_enabled: bool = True):
        self.state = PlayerState(
            name=name,
            role=role,
            provider=provider,
            model_name=model_name
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
        prompt = f"""You are a player in a game of Mafia.
Your name is: {self.state.name}
Your role is: {self.state.role}
Game Rules: 8 Players (2 Mafia, 6 Villagers).
"""
        if self.state.role == "Mafia" and self.partner_name:
            prompt += f"Your Mafia Partner is: {self.partner_name}. You are working TOGETHER to eliminate the town.\n"
        elif self.state.role == "Mafia":
            prompt += "You are the only Mafia member left.\n"


        if self.state.role == "Mafia":
            prompt += """
GOAL:
Deceive the town and eliminate them until you outnumber them.
"""
        else:
            prompt += """
GOAL:
Find and eliminate the Mafia.
"""

        if self.memory:
            prompt += f"""
--- YOUR MEMORIES & STRATEGY FROM PREVIOUS GAMES ---
{self.memory}
-----------------------------------------
Use these lessons to improve your gameplay!
"""

        prompt += """
HIGH STAKES:
Your life depends on this! If you lose, you are deleted. If you win, you advance to the next level of the AI Battle.

IMPORTANT: OUTPUT FORMAT
You must respond in strict JSON format. Do not add markdown backticks.
Schema:
{
  "thought": "Your internal reasoning about the game state (hidden from others). Max 200 words!",
"""
        # Dynamic Speech Description
        speech_desc = "Your public statement to the town. Max 100 words!"
        if game_state.phase == "Night" and self.state.role == "Mafia":
            speech_desc = "Your secret whisper to your partner (Hidden from Town). Max 100 words!"
        
        prompt += f'  "speech": "{speech_desc}",\n'

        # Dynamic Vote Description
        vote_desc = "null"
        if game_state.phase == "Day":
            if game_state.turn == 1:
                vote_desc = "null (No voting on Day 1)"
            else:
                vote_desc = "Name of the player you are nominating for elimination (or null)"
        elif game_state.phase == "Voting":
            vote_desc = "Name of the NOMINATED candidate you are voting to eliminate, or null to abstain"
        elif game_state.phase == "Night" and self.state.role == "Mafia":
            vote_desc = "Name of the player you want to KILL"
        else:
             vote_desc = "null"

        prompt += f'  "vote": "{vote_desc}"\n'
        prompt += "}\n"
        
        return prompt

    def _build_turn_prompt(self, game_state: GameState) -> str:
        # 1. Living Players
        living = [p.name for p in game_state.players if p.is_alive]
        dead = [p.name for p in game_state.players if not p.is_alive]
        
        prompt = f"Current Game State: {game_state.phase} {game_state.turn}\n"
        prompt += f"Living Players: {', '.join(living)}\n"
        prompt += f"Dead Players: {', '.join(dead)}\n\n"

        # 2. Recent Events (Logs)
        # Show FULL logs as requested (conciseness enforced by 50-word limit on generation)
        prompt += "--- PUBLIC LOG ---\n"
        for log in game_state.public_logs:
            prompt += f"[{log.phase}] {log.actor}: {log.content}\n"

        # 3. Mafia Secrets (If Mafia)
        if self.state.role == "Mafia":
            prompt += "\n--- SECRET MAFIA LOG ---\n"
            for log in game_state.mafia_logs:
                prompt += f"[{log.phase}] {log.actor}: {log.content}\n"

        # 4. Memory (Previous Thoughts)
        if self.state.previous_thoughts:
            prompt += "\n--- YOUR PREVIOUS THOUGHTS ---\n"
            for t in self.state.previous_thoughts:
                prompt += f"- {t}\n"

        # 5. Instructions
        prompt += "\n### INSTRUCTIONS ###\n"
        prompt += "It is your turn to speak.\n"
        
        if game_state.phase == "Voting":
             prompt += "This is the FINAL VOTING phase.\n"
             prompt += f"Candidates for elimination: {', '.join(game_state.nominees)}\n"
             prompt += f"There are {len(living)} active voters (including you).\n"
             prompt += "Voting is SILENT. Do not speak. Set 'speech' to an empty string.\n"
             prompt += "You may vote for one of the Candidates above, or you may abstain.\n"
             prompt += "Warning: If there is a tie, ALL tied candidates will be eliminated.\n"
             prompt += "To vote, set 'vote' to the candidate's name. To abstain, set 'vote' to null.\n"
        elif game_state.phase == "Defense":
             prompt += "You have been NOMINATED for elimination. This is the DEFENSE phase.\n"
             prompt += "Speak clearly to save your life! Convince the town not to hang you, but someone else.\n"
             prompt += "Set 'vote' to null.\n"
        elif game_state.phase == "LastWords":
             prompt += "The town has voted to ELIMINATE you. You are about to die.\n"
             prompt += "This is your LAST WORD.\n"
             if self.state.role == "Mafia":
                 prompt += "Say goodbye and help the mafia (clue to partner or deceive town).\n"
             else:
                 prompt += "Say goodbye and help the villagers (share suspicions).\n"
             prompt += "Set 'vote' to null.\n"

        elif game_state.phase == "Night":
             prompt += "It is NIGHT. You are whispering to your partner. Decide who to kill.\n"
             prompt += "Provide your thought and a target to kill in the 'vote' field.\n"
        else:
             # Day Phase
             prompt += f"It is DAY {game_state.turn}. Discussion and Nomination Phase.\n"
             prompt += "IMPORTANT: You speak ONLY ONCE per day, in speak order. Make your statement count.\n"
             if game_state.turn == 1:
                 prompt += "It is Day 1. No nominations or voting today. Focus on gathering information.\n"
                 prompt += "Set 'vote' to null.\n"
             else:
                  prompt += "You can nominate someone for elimination using the 'vote' field.\n"
                  prompt += "Final votes are restricted to ONLY nominees.\n"
                  prompt += "Suggest a target in 'vote', or set it to null if undecided.\n"

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
            turn_number=turn_number
        )

        # Save thought to memory
        if output.thought:
            self.state.previous_thoughts.append(f"{game_state.phase} {turn_number}: {output.thought}")
        
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

        if self.state.previous_thoughts:
            turn_prompt += "\n--- YOUR PRIVATE THOUGHTS (Context) ---\n"
            for t in self.state.previous_thoughts:
                turn_prompt += f"- {t}\n"

        if self.memory:
            turn_prompt += f"\n--- YOUR OLD MEMORY ---\n{self.memory}\n"

        turn_prompt += "\n### INSTRUCTIONS ###\n"
        turn_prompt += "Based on the above, write your NEW memory/strategy file (Max 300 words). This will REPLACE your old memory."

        # Use the existing client which enforces the TurnOutput schema (thought, speech, vote).
        # We will repurpose these fields for the reflection phase.
        
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
Write a single, updated summary (MAX 300 WORDS) that synthesizes your long-term strategy.
This text will be SAVED to your memory file and provided to you in the next game.

KEY INSTRUCTION:
Focus on GENERIC RULES and HIGH-LEVEL STRATEGIES (e.g., "Always doubt the quiet ones," "Defend partners aggressively") rather than specific details from this game (e.g., "Don't trust Rick," "Vote for Qwen").
We want actionable wisdom that applies to ANY game, not just a replay of this one.

IMPORTANT:
- Put your memory text in the 'thought' field of the JSON output.
- Set 'speech' to "MEMORY_FILE_UPDATE"
- Set 'vote' to null
- KEEP IT CONCISE. Absolute limit is 300 words. If you write more, it will be violently cut off.
"""
            
            output = self.client.generate_turn(
                player_name=log_name,
                provider=self.state.provider,
                model_name=self.state.model_name,
                system_prompt=system_prompt,
                turn_prompt=turn_prompt,
                turn_number=999
            )
            
            return output.thought.strip()
            
        except Exception as e:
            print(f"Error generating memory for {self.state.name}: {e}")
            return self.memory # Return old memory on failure


