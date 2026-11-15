import logging
import os
from pydantic import ValidationError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.providers import BaseLLM, create_provider
from app.schemas import AiIntent, IncomingMessage

logger = logging.getLogger(__name__)

def _is_retryable_error(exc: BaseException) -> bool:
    """Returns True for JSON formatting errors or timeouts"""
    if isinstance(exc, (ValidationError, TimeoutError, ConnectionError)):
        return True

    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status_code in {408, 429, 500, 502, 503, 504}:
        return True

    error_text = str(exc).lower()
    transient_keywords = (
        "429",
        "rate limit",
        "resource exhausted",
        "503",
        "overloaded",
        "unavailable",
        "timeout",
        "connection",
    )
    return any(keyword in error_text for keyword in transient_keywords)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=8, jitter=1),
    retry=retry_if_exception(_is_retryable_error),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)

def _classify_with_retry(provider: BaseLLM, message: IncomingMessage) -> AiIntent:
    """3 retries using exponential retry timing"""
    return provider.classify(message)

def classify_with_resilience(
    message: IncomingMessage,
    primary_name: str | None = None,
    fallback_name: str | None = None,
) -> tuple[AiIntent, str]:
    """switch provider on max timeout, returns llm used and response"""
    primary_key = (primary_name or os.getenv("PRIMARY_PROVIDER", "gemini")).strip().lower()
    fallback_key = (fallback_name or os.getenv("FALLBACK_PROVIDER", "")).strip().lower()

    primary_provider = create_provider(primary_key)

    try:
        decision = _classify_with_retry(primary_provider, message)
        return decision, primary_provider.name
    except Exception as primary_err:
        if not fallback_key or fallback_key == primary_key:
            logger.error("Primary provider '%s' failed and no fallback is set.", primary_key)
            raise

        logger.warning(
            "Primary provider '%s' failed ('%s'). Switching  to fallback '%s'...",
            primary_key,
            primary_err,
            fallback_key,
        )
        fallback_provider = create_provider(fallback_key)
        decision = _classify_with_retry(fallback_provider, message)
        return decision, fallback_provider.name
