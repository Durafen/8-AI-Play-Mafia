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

    def _log_debug(self, player_name: str, turn_number: int, prompt: str, response: str):
        directory = f"logs"
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        filename = f"{directory}/{player_name}_history.txt"
        
        # Try to clean up the response for the log
        log_response = response
        try:
            # Re-use simple parse logic to get the dict (simplified here to avoid circular dep with _parse_and_validate logic if we needed it, but logic is self contained)
            # 1. Strip markdown
            clean = response.replace("```json", "").replace("```", "").strip()
            
            # 2. Regex find
            import re, json
            match = re.search(r"(\{|\[).+(\}|\])", response, re.DOTALL)
            
            data = None
            try:
                data = json.loads(clean)
            except:
                if match:
                    data = json.loads(match.group(0))

            if data:
                # Handle wrappers (CLI list/dict)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "result" in item and isinstance(item["result"], str):
                            data = json.loads(item["result"].replace("```json", "").replace("```", "").strip())
                            break
                elif isinstance(data, dict) and "result" in data and isinstance(data["result"], str):
                    data = json.loads(data["result"].replace("```json", "").replace("```", "").strip())
                
                # Now data should be the TurnOutput dict
                if "thought" in data: # Basic check
                    log_response = (
                        f"THOUGHT: {data.get('thought')}\n"
                        f"SPEECH:  {data.get('speech')}\n"
                        f"VOTE:    {data.get('vote')}"
                    )
        except:
            pass # Keep raw if any parse error during logging

        with open(filename, "a", encoding="utf-8") as f:
            f.write(f"\n--- Turn {turn_number} ---\nPROMPT:\n{prompt}\n\nRESPONSE:\n{log_response}\n\n" + "-"*80 + "\n")

    def _parse_and_validate(self, response_text: str) -> TurnOutput:
        """Attempts to parse JSON from the response and validate against TurnOutput schema."""
        try:
            # 1. Try generic strip of markdown 
            clean_text = response_text.replace("```json", "").replace("```", "").strip()
            
            # 2. Try Regex to find JSON object (handles headers/logs) or list
            import re
            # Match either object {...} OR list [...]
            json_match = re.search(r"(\{|\[).+(\}|\])", response_text, re.DOTALL)
            
            # Let's try to parse whatever text we have as JSON first
            data = None
            try:
                data = json.loads(clean_text)
            except:
                if json_match:
                    clean_text = json_match.group(0)
                    data = json.loads(clean_text)
                else:
                    raise

            # 3. Handle CLI Wrapper Formats
            
            # Case A: List output (Qwen CLI)
            if isinstance(data, list):
                # Search for the "result" object in the list
                found_result = False
                for item in data:
                    if isinstance(item, dict) and "result" in item and isinstance(item["result"], str):
                        # Found it!
                        clean_text = item["result"]
                        clean_text = clean_text.replace("```json", "").replace("```", "").strip()
                        data = json.loads(clean_text)
                        found_result = True
                        break
                
                if not found_result:
                    print("Debug: JSON List returned but no 'result' field found.")
                    # fallback, maybe the text was just a list? (Unlikely for TurnOutput)

            # Case B: Nested "result" field in dict (Claude CLI)
            elif isinstance(data, dict):
                if "result" in data and isinstance(data["result"], str):
                    inner_text = data["result"]
                    # Clean markdown
                    inner_text = inner_text.replace("```json", "").replace("```", "").strip()
                    
                    # Try regex again on inner text because it might have emojis/prefixes (like 🤖)
                    inner_match = re.search(r"\{.*\}", inner_text, re.DOTALL)
                    if inner_match:
                        inner_text = inner_match.group(0)
                        
                    data = json.loads(inner_text)
            
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
            # claude --print --output-format json --model <model> <prompt>
            cmd = ["claude", "--print", "--output-format", "json", "--model", model, prompt]
            
        elif command == "gemini":
            # gemini --model <model> <prompt>
            cmd = ["gemini", "--model", model, prompt]
            
        elif command == "qwen":
            # qwen --output-format json --model <model> <prompt>
            cmd = ["qwen", "--output-format", "json", "--model", model, prompt]
            
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
