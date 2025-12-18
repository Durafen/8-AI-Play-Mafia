# AI Mafia 🕵️‍♂️🤖

> ⚠️ **Note: This project is HEAVILY WORK IN PROGRESS.**
> Features, APIs, and CLI integrations are subject to rapid change. Use with caution and expect bugs!

**AI Mafia** is a Python-based simulation engine where Large Language Models (LLMs) play the classic social deduction game *Mafia* against each other. Watch as AI agents debate, deceive, and deduce their way to victory in a fully automated environment.

It supports multiple AI providers including OpenAI, Anthropic, Google Gemini, and Groq (Qwen/Llama).

---

## 🚀 Features

*   **Multi-Model Roster**: Pit different models against each other (e.g., GPT-5.2 vs Claude Sonnet vs Gemini 2.5).
*   **Dual Modes**: Run using **Local CLI Tools** (free/cheap if using local subscriptions) or via **Direct API Calls**.
*   **Automated Game Loop**: The engine handles phases (Day, Defense, Voting, Last Words, Night), turn management, and death logic automatically.
*   **Text-to-Speech**: Each player has a unique voice (Edge TTS) - hear the AI debate and deceive!
*   **Detailed Logs**: Observe "inner thoughts" of models to understand their strategy and deception.
*   **Rich Terminal UI**: Formatted output with icons for Day/Night cycles and role reveals.
*   **Game Persistence**: Game logs saved to `games/` directory with timestamps.

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

---

## 🤝 Contributing

Feel free to fork and submit PRs to add new providers, improved prompting strategies, or web interfaces!
