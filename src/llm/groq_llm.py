

import os
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from groq import Groq

# Correct import for CrewAI 0.28+
from crewai.llm import LLM

load_dotenv()


class GroqLLM(LLM):
    """
    Fully functional Groq LLM wrapper for CrewAI (v0.28+).

    - Uses Groq API directly
    - Bypasses all OpenAI defaults
    - Implements required `call()` method
    - Supports llama-3.1-8b-instant
    """

    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.2,
        max_retries: int = 3,
        timeout: int = 30,
    ):
        super().__init__()  # required by CrewAI

        self._model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is missing in .env!")

        self.client = Groq(api_key=api_key)

    @property
    def model(self):
        """CrewAI queries this property."""
        return self._model

    def call(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """
        CrewAI ALWAYS calls this method.
        It must return only the generated text string.
        """
        attempts = 0

        while attempts < self.max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=1024,
                )

                # CORRECT way to read content from Groq SDK
                return response.choices[0].message.content

            except Exception as e:
                attempts += 1
                print(f"[GroqLLM] Error: {e} (attempt {attempts}/{self.max_retries})")
                time.sleep(1)

        raise RuntimeError("GroqLLM failed after maximum retries.")


