import os
import json
import logging
from typing import Optional, Dict, Any, Type
from pydantic import BaseModel
from openai import OpenAI
from anthropic import Anthropic

from schemas import TurnOutput
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

class UnifiedLLMClient:
    def __init__(self, debug: bool = True):
        self.debug = debug
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        

        
        # xAI (Grok)
        self.xai_client = OpenAI(
            api_key=os.getenv("XAI_API_KEY"),
            base_url="https://api.x.ai/v1",
        )
        
        # Groq (Llama) - Optional, kept if user wants to switch back
        self.groq_client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )

    def _log_debug(self, player_name: str, turn: int, prompt: str, response: str):
        if not self.debug:
            return
        
        # Log Prompt (Disabled per user request)
        # with open(f"logs/{player_name}_prompt_{turn}.txt", "w", encoding="utf-8") as f:
        #     f.write(prompt)
            
        # Log Response (Disabled per user request)
        # with open(f"logs/{player_name}_response_{turn}.txt", "w", encoding="utf-8") as f:
        #     f.write(response)
            
        # Append to History
        separator = "-" * 80
        with open(f"logs/{player_name}_history.txt", "a", encoding="utf-8") as f:
            f.write(f"\n--- Turn {turn} ---\nPROMPT:\n{prompt}\n{separator}\n\nRESPONSE:\n{response}\n{separator}\n")

    def _parse_and_validate(self, response_text: str) -> TurnOutput:
        """Attempts to parse JSON from the response and validate against TurnOutput schema."""
        try:
            # Strip markdown code blocks if present
            clean_text = response_text.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_text)
            return TurnOutput(**data)
        except (json.JSONDecodeError, Exception) as e:
            # Fallback or error handling could go here. For now, re-raise or return a dummy fail.
            print(f"Error parsing JSON: {e}")
            print(f"Raw received: {response_text}")
            # Try to recover partial? No, strictly fail for now to catch issues early.
            raise ValueError(f"Failed to parse model output as JSON: {e}")

    def _call_cli(self, command: str, model: str, prompt: str) -> str:
        """Executes a local terminal command for the model."""
        import subprocess
        
        # Construct command based on tool specific syntax
        cmd = []
        
        if command == "codex":
            # codex exec --model <model> <prompt>
            cmd = ["codex", "exec", "--model", model, prompt]
            
        elif command == "claude":
            # claude --print --model <model> <prompt>
            cmd = ["claude", "--print", "--model", model, prompt]
            
        elif command == "gemini":
            # gemini --model <model> <prompt> (Assumed standard based on usage)
            cmd = ["gemini", "--model", model, prompt]
            
        elif command == "qwen":
            # qwen --model <model> --prompt <prompt>  (OR positional?)
            # Help said: -p, --prompt ... [deprecated: Use the positional prompt instead]
            # So: qwen --model <model> <prompt>
            cmd = ["qwen", "--model", model, prompt]
            
        else:
            # Fallback
            cmd = [command, "--model", model, prompt]
        
        try:
            # Run command
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            # If command not found or fails
            print(f"CLI Error ({command}): {e.stderr}")
            raise e

    def generate_turn(self, player_name: str, provider: str, model_name: str, system_prompt: str, turn_prompt: str, turn_number: int) -> TurnOutput:
        
        # TOGGLE: Set to False to use the real API
        USE_CLI = True 
        
        full_prompt = f"{system_prompt}\n\n{turn_prompt}"
        if USE_CLI:
             full_prompt += "\n\nIMPORTANT: Return ONLY JSON object."
        
        response_text = ""

        try:
            if USE_CLI:
                # --- CLI MODE ---
                cli_command = None
                if provider == "openai":
                    cli_command = "codex"
                elif provider == "anthropic":
                    cli_command = "claude"
                elif provider == "google":
                    cli_command = "gemini"
                elif provider == "groq":
                    cli_command = "qwen" 

                if cli_command:
                    response_text = self._call_cli(cli_command, model_name, full_prompt)
                else:
                    raise ValueError(f"No CLI tool mapped for provider {provider}")

            else:
                # --- API MODE ---
                if provider == "openai":
                    response = self.openai_client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": turn_prompt}
                        ],
                        response_format={"type": "json_object"}
                    )
                    response_text = response.choices[0].message.content

                elif provider == "xai": # Grok
                    model = model_name
                    try:
                        response = self.xai_client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": turn_prompt}
                            ],
                            response_format={"type": "json_object"}
                        )
                    except:
                         response = self.xai_client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": turn_prompt + "\n\nProvide your response in JSON format."}
                            ]
                        )
                    response_text = response.choices[0].message.content

                elif provider == "groq":
                    response = self.groq_client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": turn_prompt},
                        ],
                        response_format={"type": "json_object"}
                    )
                    response_text = response.choices[0].message.content

                elif provider == "anthropic":
                    response = self.anthropic_client.messages.create(
                        model=model_name,
                        max_tokens=1024,
                        system=system_prompt,
                        messages=[
                            {"role": "user", "content": turn_prompt}
                        ]
                    )
                    response_text = response.content[0].text

                elif provider == "google":
                    from google import genai
                    from google.genai import types
                    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
                    response = client.models.generate_content(
                        model=model_name,
                        contents=turn_prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            response_mime_type="application/json"
                        )
                    )
                    response_text = response.text
                else:
                    raise ValueError(f"Unknown provider: {provider}")

            # Debug Log
            self._log_debug(player_name, turn_number, full_prompt, response_text)

            # Parse
            return self._parse_and_validate(response_text)

        except Exception as e:
            # Log the failure too
            self._log_debug(player_name, turn_number, full_prompt, f"ERROR: {str(e)}")
            raise e
