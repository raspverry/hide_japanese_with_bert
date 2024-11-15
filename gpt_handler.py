# gpt_handler.py
import os
import requests
from typing import List, Dict

class GPTHandler:
    def __init__(self, api_key: str = None):
        """Initialize ChatGPT client"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def ask(self, messages: List[Dict[str, str]], model: str = "gpt-4o-mini") -> str:
        """Send request to ChatGPT"""
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.7
            }
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"GPT API Error: {response.status_code} - {response.text}")