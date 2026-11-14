import os
from abc import ABC, abstractmethod
from google import genai
from google.genai import types

from app.prompts import SYSTEM_PROMPT, build_user_prompt
from app.schemas import AiIntent, Incoming_Message


class BaseLLM(ABC):
    """abstract class all llm providers must implement"""
    name: str = "base"

    @abstractmethod
    def classify(self, message: Incoming_Message) -> AiIntent:
        """takes guest message and returns an structured response"""
    raise NotImplementedError


class GeminiProvider(BaseLLM):
    """Uses google's oficial SDK structured outputs"""
    def __init__(self, api_key: str, model: str = "gemini-3.5-flash-lite"):
        self.name = f"gemini ({model})"
        self.model = model
        self.client = genai.Client(api_key=api_key)

    def classify(self, message: Incoming_Message) -> AiIntent:
        user_prompt = build_user_prompt(message.username, message.text)

        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instructions=SYSTEM_PROMPT,
                response_mime_types="application/json",
                response_schema=AiIntent,
                temperature=0.2
            ),
        )
        return AiIntent.model_validate_json(response.text)

    def create_provider(name: str | None = None) -> BaseLLM:
        """factory that instantiates provider chosen on .env"""
        provider = (name or os.getenv("PRIMARY_PROVIDER", "gemini")).strip.lower()

        if provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY", "")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not set on .env file")
            return GeminiProvider(api_key=api_key)
        raise ValueError(f"Unknown provider: {provider}")

