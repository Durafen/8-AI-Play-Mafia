from typing import List, Optional
from schemas import PlayerState, TurnOutput, GameState, LogEntry
from api_clients import UnifiedLLMClient

class Player:
    def __init__(self, name: str, role: str, provider: str, model_name: str, client: UnifiedLLMClient, player_index: int):
        self.state = PlayerState(
            name=name,
            role=role,
            provider=provider,
            model_name=model_name
        )
        self.client = client
        self.player_index = player_index
        self.partner_name: Optional[str] = None # For mafia to know their partner

    def set_partner(self, partner_name: str):
        self.partner_name = partner_name

    def _build_system_prompt(self, game_state: GameState) -> str:
        prompt = f"""You are a player in a game of Mafia.
Your name is: {self.state.name}
Your role is: {self.state.role}
Game Rules: There are 8 players total. 2 are Mafia, 6 are Villagers.
"""
        if self.state.role == "Mafia" and self.partner_name:
            prompt += f"Your Mafia Partner is: {self.partner_name}. You are working TOGETHER to eliminate the town.\n"
        elif self.state.role == "Mafia":
            prompt += "You are the only Mafia member left.\n"
        else:
            prompt += "You are a Villager. You do not know who the Mafia is.\n"

        prompt += """
GOAL:
- If Town: Find and eliminate the Mafia.
- If Mafia: Deceive the town and eliminate them until you outnumber them.

HIGH STAKES:
Your life depends on this! If you lose, you are deleted. If you win, you advance to the next level of the AI Battle.

IMPORTANT: OUTPUT FORMAT
You must respond in strict JSON format. Do not add markdown backticks.
Schema:
{
  "thought": "Your internal reasoning about the game state (hidden from others)",
"""
        # Dynamic Speech Description
        speech_desc = "Your public statement to the town (max 4 sentences)"
        if game_state.phase == "Night" and self.state.role == "Mafia":
            speech_desc = "Your secret whisper to your partner (Hidden from Town)"
        
        prompt += f'  "speech": "{speech_desc}",\n'

        # Dynamic Vote Description
        vote_desc = "null"
        if game_state.phase == "Day":
            if game_state.turn == 1:
                vote_desc = "null (No voting on Day 1)"
            else:
                vote_desc = "Name of the player you are nominating for elimination (or null)"
        elif game_state.phase == "Voting":
            vote_desc = "Name of the NOMINATED candidate you are voting to eliminate"
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
        prompt += "\nIt is your turn to speak.\n"
        
        if game_state.phase == "Voting":
             prompt += "This is the FINAL VOTING phase. All players must vote.\n"
             prompt += f"Candidates for elimination: {', '.join(game_state.nominees)}\n"
             prompt += "Voting is SILENT. Do not speak. Set 'speech' to an empty string.\n"
             prompt += "You MUST vote for one of the Candidates above. Set 'vote' to their name.\n"
        elif game_state.phase == "Defense":
             prompt += "You have been NOMINATED for elimination. This is the DEFENSE phase.\n"
             prompt += "Speak clearly to save your life! Convince the town not to hang you.\n"
             prompt += "Set 'vote' to null.\n"
        elif game_state.phase == "LastWords":
             prompt += "The town has voted to ELIMINATE you. You are about to die.\n"
             prompt += "This is your LAST WORD. Say your final goodbye or curse the town.\n"
             prompt += "Set 'vote' to null.\n"
        elif game_state.phase == "Night":
             prompt += "It is NIGHT. You are whispering to your partner. Decide who to kill.\n"
             prompt += "Provide your thought and a target to kill in the 'vote' field.\n"
        else:
             # Day Phase
             prompt += f"It is DAY {game_state.turn}. Discussion and Nomination Phase.\n"
             if game_state.turn == 1:
                 prompt += "It is Day 1. No nominations or voting today. Focus on gathering information.\n"
                 prompt += "Set 'vote' to null.\n"
             else:
                 prompt += "You can nominate someone for elimination using the 'vote' field.\n"
                 prompt += "Any player who is nominated will have to DEFEND themselves before the final vote.\n"
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

