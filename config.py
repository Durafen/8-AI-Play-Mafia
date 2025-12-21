# config.py - Mafia Game Configuration

# TTS Config
TTS_ENABLED = True   # Set to False to disable text-to-speech
TTS_RATE = "+30%"    # Speech speed: "+30%" = 30% faster, "-10%" = 10% slower
AUTO_CONTINUE = True # Set to True to run without user intervention
MEMORY_ENABLED = False # Set to True to enable distinct memories per player from previous games
REVEAL_ROLE_ON_DEATH = True # Set to False to hide role when player dies

# Narrator voice for system announcements
NARRATOR_VOICE = "en-US-AriaNeural"

# Role emoji mappings (single source of truth)
ROLE_EMOJIS = {
    "Mafia": "👺",
    "Cop": "👮",
    "Villager": "👤"
}

# Phase emoji mappings
PHASE_EMOJIS = {
    "Day": "☀️ ",
    "Night": "🌙 ",
    "Trial": "⚖️ ",
    "Setup": "⚙️ ",
    "Reflection": ""
}

# Player Roster Configuration
# active: True = participates in game, False = disabled
# use_cli: True = CLI tool, False = API
ROSTER_CONFIG = [
    # OPENAI
    {"active": False, "use_cli": True, "name": "Rick", "provider": "openai", "model": "gpt-5.2", "voice": "en-US-GuyNeural"},
    {"active": False, "use_cli": True, "name": "Morty", "provider": "openai", "model": "gpt-5.1", "voice": "en-US-ChristopherNeural"},
    {"active": True, "use_cli": True, "name": "Gpt", "provider": "openai", "model": "gpt-5.1-codex-mini", "voice": "en-US-EricNeural"},

    # ANTHROPIC
    {"active": True, "use_cli": True, "name": "Haiku", "provider": "anthropic", "model": "haiku", "voice": "en-GB-RyanNeural"},
    {"active": True, "use_cli": True, "name": "Sonnet", "provider": "anthropic", "model": "sonnet", "voice": "en-GB-SoniaNeural"},
    {"active": False, "use_cli": True, "name": "Opus", "provider": "anthropic", "model": "opus", "voice": "en-US-AndrewNeural"},

    # OPENROUTER (API only)
    {"active": False, "use_cli": False, "name": "Chimera", "provider": "openrouter", "model": "tngtech/deepseek-r1t2-chimera:free", "voice": "en-AU-NatashaNeural"},
    {"active": False, "use_cli": False, "name": "Deepseek", "provider": "openrouter", "model": "nex-agi/deepseek-v3.1-nex-n1:free", "voice": "en-CA-LiamNeural"},
    {"active": False, "use_cli": False, "name": "Devstral", "provider": "openrouter", "model": "mistralai/devstral-2512:free", "voice": "en-CA-ClaraNeural"},
    {"active": False, "use_cli": False, "name": "Olmo", "provider": "openrouter", "model": "allenai/olmo-3.1-32b-think:free", "voice": "en-US-BrianNeural"},
    {"active": False, "use_cli": False, "name": "Oss", "provider": "openrouter", "model": "openai/gpt-oss-120b:free", "voice": "en-PH-JamesNeural"},

    # GOOGLE
    {"active": True, "use_cli": True, "name": "Pro", "provider": "google", "model": "gemini-2.5-pro", "voice": "en-NZ-MitchellNeural"},
    {"active": True, "use_cli": True, "name": "Flash", "provider": "google", "model": "gemini-2.5-flash", "voice": "en-IE-ConnorNeural"},
    {"active": True, "use_cli": True, "name": "Preview", "provider": "google", "model": "gemini-3-flash-preview", "voice": "en-IE-EmilyNeural"},

    # QWEN (via qwen CLI)
    {"active": True, "use_cli": True, "name": "Qwen", "provider": "qwen", "model": "coder-model", "voice": "en-ZA-LukeNeural"},
    {"active": True, "use_cli": True, "name": "Ling", "provider": "qwen", "model": "coder-model", "voice": "en-ZA-LeahNeural"},
    {"active": False, "use_cli": True, "name": "Chang", "provider": "qwen", "model": "coder-model", "voice": "en-GB-ThomasNeural"},
    {"active": False, "use_cli": True, "name": "Vision", "provider": "qwen", "model": "vision-model", "voice": "en-HK-SamNeural"},

    # OLLAMA (local)
    {"active": False, "use_cli": True, "name": "Nemotron", "provider": "ollama", "model": "nemotron-3-nano:30b-cloud", "voice": "en-IN-PrabhatNeural"},
]
