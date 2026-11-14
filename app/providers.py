import json
import os
from abc import ABC, abstractmethod
from google import genai
from google.genai import types
from openai import OpenAI

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


class OpenAICompatibleProvider(BaseLLM):
    """provider class llama.cpp and deepseek compatible"""

    def __init__(
        self,
        provider_label: str,
        api_key: str,
        base_url: str,
        model: str,
        strict_schema: bool = True,
    ):
        self.name = f"{provider_label} ({model})"
        self.model = model
        self.strict_schema = strict_schema
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def classify(self, message: Incoming_Message) -> AiIntent:
        user_prompt = build_user_prompt(message.guest_name, message.text)
        schema = AiIntent.model_json_schema()

        if self.strict_schema:
            system_msg = SYSTEM_PROMPT
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "ai_decision",
                    "strict": True,
                    "schema": schema,
                },
            }
        else:
            system_msg = (
                f"{SYSTEM_PROMPT}\n\n"
                f"Respond strictly with a valid JSON matching this schema \n"
                f"{json.dumps(schema, ensure_ascii=False)}"
            )
            response_format = {"type": "json_object"}

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt},
            ],
            response_format=response_format,
            temperature=0.2,
        )

        raw_json = response.choices[0].message.content or "{}"
        return AiIntent.model_validate_json(raw_json)

    def create_provider(name: str | None):
        """Factory that instantiates the requested LLM provider"""
        selected = (name or os.getenv("PRIMARY_PROVIDER", "gemini")).strip.lower()

        if selected == "gemini":
            api_key = os.getenv("GEMINI_API_KEY", "")
            if not api_key:
                raise ValueError("GEMINI_API_KEY is missing on .env")
            return GeminiProvider(api_key=api_key)

        if selected == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY is missing on .env")
            return OpenAICompatibleProvider(
                provider_label="deepseek",
                api_key=api_key,
                base_url="https://api.deepseek.com",
                model="deepseek",
                strict_scheme=False,
            )

        if selected == "llamacpp":
            base_url = os.getenv("LLAMACPP_BASE_URL", "http://localhost:8080/v1")
            model = os.getenv("LLAMACPP_MODEL", ""),
            return OpenAICompatibleProvider(
                api_key="not-needed",
                base_url=base_url,
                model=model,
                strict_schema=True,
            )
        raise ValueError(f"unknown provider: '{selected}'")

