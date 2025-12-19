# AI Mafia 🕵️‍♂️🤖

> ⚠️ **Note: This project is HEAVILY WORK IN PROGRESS.**
> Features, APIs, and CLI integrations are subject to rapid change. Use with caution and expect bugs!

**AI Mafia** is a Python-based simulation engine where Large Language Models (LLMs) play the classic social deduction game *Mafia* against each other. Watch as AI agents debate, deceive, and deduce their way to victory in a fully automated environment.

It supports multiple AI providers including OpenAI, Anthropic, Google Gemini, and Groq (Qwen/Llama).

---

## ⚔️ The AI Battle Royale

Watch the most advanced AI models from competing companies deceive and outsmart each other:

| Player | Provider | Model |
|--------|----------|-------|
| Rick | OpenAI | GPT-5.2 |
| Morty | OpenAI | GPT-5.1 |
| Sonnet | Anthropic | Claude Sonnet 4 |
| Haiku | Anthropic | Claude Haiku 3.5 |
| Pro | Google | Gemini 2.5 Pro |
| Flash | Google | Gemini 2.5 Flash |
| Preview | Google | Gemini 3 Flash Preview |
| Qwen | Groq | Qwen 3 Coder |

---

## 🌟 Spectacular Example Games

Check out some of our best simulated games to see high-level AI deception and deduction in action:

*   **[Example 1: The "Trust the Pattern" Trap](https://raw.githubusercontent.com/Durafen/8-AI-Play-Mafia/master/examples/example1.txt)**
    *   **Highlights**: Day 1 starts with a brilliant "accountability framework" proposed by Rick (Town) to trap the Mafia. It works perfectly... until the Mafia flips the script. By the end, a Town player (Sonnet) is executed while confidently explaining why their execution is mathematically impossible if they were Town. Watch how "perfect logic" can be weaponized to lead the Town off a cliff.

*   **[Example 2: The "Mafia Math" Deception](https://raw.githubusercontent.com/Durafen/8-AI-Play-Mafia/master/examples/example2.txt)**
    *   **Highlights**: A masterclass in 1v1v1 endgame manipulation! With 3 players left (1 Mafia, 2 Town), Morty (Town) is absolutely convinced he has solved the game using "parity math." He clears one player (Pro) completely based on game theory. But is his math right? Or is he confidently walking into a trap set by a "confirmed townie"? See if Flash can spot the flaw before it's too late.

*   **[Example 3: 3v2 LYLO: The "Ghost Narrative" Logic Trap](https://raw.githubusercontent.com/Durafen/8-AI-Play-Mafia/master/examples/example3.txt)**
    *   **Highlights**: In a tense 3v2 finale, Town player Preview brilliantly exposes a 'Weaponized Consensus' strategy used to control the game. The table is set for a massive turnaround, but it all hinges on Qwen. Will Qwen see through the 'Ghost Narrative,' or will the 'Wall of Logic' prove too strong? Watch the psychological pressure cooker explode.

*   **[Example 4: The "Illogical Killer" Gambit](https://raw.githubusercontent.com/Durafen/8-AI-Play-Mafia/master/examples/example4.txt)**
    *   **Highlights**: Pro (Mafia) attempts a high-risk "Illogical Killer" gambit. After the *only* player voting for him is killed, he uses the paradox as his defense: "Why would I silence my own critic?" In the endgame, he targets the quietest player (Qwen), arguing that his own "loud and messy" play is proof of innocence. It becomes a showdown of narratives: is Pro a confused Villager, or is he hiding in plain sight?

*   **[Example 5: The "Victory Speech" Bluff](https://raw.githubusercontent.com/Durafen/8-AI-Play-Mafia/master/examples/example5.txt)**
    *   **Highlights**: Facing a mathematical loss in a 2v1 endgame, Haiku (Mafia) tries a bold anti-strategy: he congratulates the Town on their win and *abstains* from voting, refusing to nominate anyone. It's a high-stakes dare: is he a defeated Villager accepting fate, or a Scum player hoping the Town will eat itself if he refuses to fight? Watch the final two Townies spiral as they try to interpret the silence.

---

## 🚀 Features

*   **Multi-Model Roster**: Pit different models against each other (e.g., GPT-5.2 vs Claude Sonnet vs Gemini 2.5).
*   **Dual Modes**: Run using **Local CLI Tools** (free/cheap if using local subscriptions) or via **Direct API Calls**.
*   **Automated Game Loop**: The engine handles phases (Day, Defense, Voting, Last Words, Night), turn management, and death logic automatically.
*   **Text-to-Speech**: Each player has a unique voice (Edge TTS) - hear the AI debate and deceive!
*   **Detailed Logs**: Observe "inner thoughts" of models to understand their strategy and deception.
*   **Rich Terminal UI**: Formatted output with icons for Day/Night cycles and role reveals.
*   **Game Persistence**: Game logs saved to `games/` directory with timestamps.
*   **Persistent Memory** 🧠: Models learn from previous games! They write a strategic summary after each match and use it in the next one.

---

## 🛠️ Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/ai-mafia.git
    cd ai-mafia
    ```

2.  **Create a Virtual Environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configuration (Env Vars)**
    Create a `.env` file in the root directory and add your API keys (needed for API mode or some CLI auths):
    ```ini
    OPENAI_API_KEY=sk-...
    ANTHROPIC_API_KEY=sk-ant-...
    GEMINI_API_KEY=AIza...
    GROQ_API_KEY=gsk_...
    ```

---

## 🎮 How to Play

### Running the Game
Simply run the main script:
```bash
python main.py
```
Follow the on-screen prompts. You will need to press `ENTER` to advance the game state after each agent moves, allowing you to read specific turns.

### Autoplay & Pause ⏯️
If `AUTO_CONTINUE = True` is set in `engine.py`, the game will proceed automatically.
*   **Pause**: Press **SPACE** during the 2-second delay between turns to pause the game.
*   **Resume**: Press **SPACE** again to resume.

### 🧠 Persistent Memory System
The game now features a **learning system**:
1.  **Loading**: At the start of a game, players load their "Memory" from `memories/{Name}.txt`.
2.  **Reflection Phase**: After the game ends, all players (living and dead) analyze the full logs (including secret Mafia communications).
3.  **Learning**: They generate a new 300-word strategic summary, combining old wisdom with new lessons. This file is overwritten for the next game.

**Configuration:**
In `engine.py`, you can toggle this system:
```python
MEMORY_ENABLED = True  # Set to False to disable memory/learning
```


---

## ⚙️ Configuration & Modes

The game supports two distinct modes of operation. You can switch between them by editing the `api_clients.py` file.

### 1. **Terminal Mode (CLI)** 💻
* **Best for:** Running models locally or via already-authenticated CLI tools (saving API costs).
* **How it works:** The engine shells out to command-line tools like `codex`, `claude`, `gemini`, and `qwen`.
* **Prerequisites:** You must have these tools installed in your terminal and authenticated.
* **How to Enable:**
  1. Open `api_clients.py`.
  2. Find the line `USE_CLI = False` (or true).
  3. Set it to **True**:
     ```python
     USE_CLI = True
     ```

### 2. **API Key Mode** 🔑
* **Best for:** Direct, stable connection to model providers using standard API keys.
* **How it works:** Uses official Python SDKs (`openai`, `anthropic`, `google-genai`) to send requests over the network.
* **Prerequisites:** You need a `.env` file with valid API keys (see Installation).
* **How to Enable:**
  1. Open `api_clients.py`.
  2. Find the line `USE_CLI = True`.
  3. Set it to **False**:
     ```python
     USE_CLI = False
     ```

### 🔐 Managing API Keys
If using **API Key Mode**, ensure your `.env` file is populated:

```ini
# .env file
OPENAI_API_KEY=sk-...         # For GPT models
ANTHROPIC_API_KEY=sk-ant-...  # For Claude models
GEMINI_API_KEY=AIza...        # For Gemini models
GROQ_API_KEY=gsk_...          # For Llama/Qwen models
```

> **Note:** In CLI mode, these keys might not be needed if your terminal tools are already logged in (e.g., via `glcloud auth login` or similar).

### 🔊 Text-to-Speech (TTS)
The game uses **Edge TTS** (free Microsoft neural voices) to give each player a unique voice.

Configure in `engine.py`:
```python
TTS_ENABLED = True   # Set to False to disable
TTS_RATE = "+20%"    # Speech speed: "+20%" faster, "-10%" slower
```

Each player has a distinct voice accent (American, British, Australian, Indian, Irish, Canadian, South African) defined in `ROSTER_CONFIG`.

**Requirements:** `pip install edge-tts` (auto-installed with requirements.txt)

### 🤖 Changing Players & Models
You can customize the game roster in `engine.py`. Look for `ROSTER_CONFIG`.
Each entry requires:
- `name`: Display name of the player.
- `provider`: `openai`, `anthropic`, `google`, or `groq`.
- `model`: The exact model string (e.g., `gpt-5.2` for CLI or `gpt-4o` for API).
- `voice`: Edge TTS voice ID (e.g., `en-US-GuyNeural`, `en-GB-RyanNeural`).

**Example:**
```python
ROSTER_CONFIG = [
    {"name": "Mastermind", "provider": "openai", "model": "gpt-5.2"},
    {"name": "The Poet", "provider": "anthropic", "model": "haiku"}, 
]
```

```python
# engine.py
ROSTER_CONFIG = [
    {"name": "Agent Name", "provider": "openai", "model": "gpt-4o"},
    # ... add up to 8 players
]
```

---

## 📜 Game Rules (Simulated)

*   **Roles**: 
    *   **Mafia (2)**: Know each other. Goal is to eliminate all Town members.
    *   **Villagers (6)**: Do not know who is who. Goal is to vote out the Mafia.
*   **Phases**:
    *   **Day**: All players speak publicly.
    *   **Voting**: Players vote to eliminate someone. Majority rules.
    *   **Night**: Mafia perform a secret kill.

---

## 📂 Logs

The game generates detailed logs in the `/logs` directory:
*   `public_logs.txt`: What the players see.
*   `{PlayerName}_history.txt`: Full context window for a specific agent (includes debugging info).
*   `memories/`: Persistent strategy files for each player key.

---

## 🤝 Contributing

Feel free to fork and submit PRs to add new providers, improved prompting strategies, or web interfaces!
