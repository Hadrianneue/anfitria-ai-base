import logging
import os
import sys
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

from app.providers import BaseLLM, _PROVIDER_CACHE
from app.resilience import classify_with_resilience
from app.schemas import AiIntent, IncomingMessage

class SimulatedRateLimitError(Exception):
    """Fake HTTP 429 exception to test resilience"""
    status_code = 429

class Broken429Provider(BaseLLM):
    """Mock provider error too many requests"""
    name = "mock-429-provider"

    def classify(self, message: IncomingMessage) -> AiIntent:
        raise SimulatedRateLimitError("429 too many request")

def run_normal_tests(provider_name: str | None = None) -> None:
    samples = [
        IncomingMessage(
            phone="5511988881111",
            guest_name="Mariana",
            text="Oi! Qual o horário do check-in e vocês aceitam cachorro pequeno?",
        ),
        IncomingMessage(
            phone="5511988882222",
            guest_name="Carlos",
            text="Boa tarde, quanto fica do dia 18/12 até 22/12 para 5 adultos?",
        ),
        IncomingMessage(
            phone="5511988883333",
            guest_name="Fernanda",
            text="Socorro, acabamos de chegar e o chuveiro da suíte não está esquentando!",
        ),
    ]

    print("\n=== 1. TESTANDO CLASSIFICAO E STRUCTURED OUTPUTS ===")
    for msg in samples:
        print(f"\n-> Mensagem de {msg.guest_name}: \"{msg.text}\"")
        decision, used_provider = classify_with_resilience(
            msg,
            primary_name=provider_name,
        )
        print(f"    Provedor utilizado: {used_provider}")
        print(decision.model_dump_json(indent=2))

def run_resilience_test(fallback_provider: str) -> None:
    print("\n=== 2. SIMULANDO ERRO 429 +  BACKOFF +  FALLBACK ===")
    _PROVIDER_CACHE["broken_429"] = Broken429Provider()

    msg = IncomingMessage(
        phone="5511988884444",
        guest_name="Lucas",
        text="Tem Wi-Fi bom na casa para trabalhar home office?"
    )

    decision, used_provider = classify_with_resilience(
        msg,
        primary_name="broken_429",
        fallback_name=fallback_provider,
    )
    print(f"\n  Recuperado com sucesso pelo fallback: {used_provider}")
    print(decision.model_dump_json(indent=2))

if __name__ == "__main__":
    chosen = sys.argv[1] if len(sys.argv) > 1 else os.getenv("PRIMARY_PROVIDER", "gemini")

    run_normal_tests(provider_name=chosen)
    run_resilience_test(fallback_provider=chosen)
