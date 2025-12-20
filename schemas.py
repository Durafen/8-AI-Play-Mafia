from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from uuid import uuid4

class TurnOutput(BaseModel):
    strategy: Optional[str] = Field("", description="Your strategic plan (max 100 words) - overwrites previous")
    speech: Optional[str] = Field("", description="Public statement to the town (max 75 words)")
    vote: Optional[str] = Field(None, description="Name of player to vote for (or None if not voting phase)")

class LogEntry(BaseModel):
    turn: int
    phase: Literal["Day", "Night", "Voting", "Defense", "LastWords", "Setup", "KillReveal", "Result", "Reflection"]
    actor: str
    action: str  # speak, vote, kill, die, system
    content: str

class PlayerState(BaseModel):
    name: str # Version + Model Name
    role: Literal["Mafia", "Villager"]
    is_alive: bool = True
    provider: str
    model_name: str # Technical API model name
    use_cli: bool = True  # True = CLI tool, False = API
    strategy: str = ""  # Living strategic plan, overwritten each turn

class GameState(BaseModel):
    game_id: str = Field(default_factory=lambda: str(uuid4()))
    turn: int = 1
    phase: Literal["Day", "Night", "Voting", "Defense", "LastWords", "Setup", "Result", "Reflection"] = "Setup"
    players: List[PlayerState] = []
    nominees: List[str] = [] # List of player names nominated for elimination
    public_logs: List[LogEntry] = []
    mafia_logs: List[LogEntry] = [] # Secret logs for Mafia eyes only
