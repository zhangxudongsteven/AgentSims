from typing import List, Dict, Any, Optional
import os
import json
import re
import traceback
import openai
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

class OpenAICaller:
    """OpenAI compatible API caller"""
    def __init__(
        self, 
        model: Optional[str] = None, 
        api_key: Optional[str] = None, 
        base_url: Optional[str] = None
    ) -> None:
        # Use values from .env file if not provided
        self.model = model or os.getenv('OPENAI_MODEL')
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.base_url = base_url or os.getenv('OPENAI_BASE_URL')

        if not self.model:
            self.model = "deepseek-chat"  # Default model if not specified
            
        if not self.api_key:
            raise ValueError("API key not found in .env file. Please set OPENAI_API_KEY environment variable.")
            
        # Initialize OpenAI client with optional base_url
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
            
        self.client = openai.OpenAI(**client_kwargs)
    
    async def ask(self, prompt: str) -> str:
        counter = 0
        result = "{}"
        while counter < 3:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                result = response.choices[0].message.content
                return result
            except Exception as e:
                print(f"Error calling LLM API: {e}")
                counter += 1

        try:
            traceback.print_exc()
        except:
            pass
        return result

class LLMCaller:
    """Main LLM caller interface with JSON response parsing"""
    def __init__(self, model: str) -> None:
        self.model = model
        self.caller = OpenAICaller(model)()
    
    async def ask(self, prompt: str) -> Dict[str, Any]:
        result = await self.caller.ask(prompt)
        try:
            result = json.loads(result)
        except Exception:
            try:
                info = re.findall(r"\{.*\}", result, re.DOTALL)
                if info:
                    info = info[-1]
                    result = json.loads(info)
                else:
                    result = {"response": result}
            except Exception:
                result = {"response": result}
        return result
