"""Shared LLM helpers for ingestion transforms."""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Optional

import config
from langchain_core.messages import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)


def create_ingestion_llm() -> BaseChatModel:
    if config.LLM_PROVIDER == "deepseek":
        from langchain_openai import ChatOpenAI

        if not config.DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_KEY is required when LLM_PROVIDER=deepseek")
        return ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.LLM_BASE_URL,
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)


def create_vision_llm() -> Optional[BaseChatModel]:
    """Create vision LLM for image captioning (independent of text LLM provider)."""
    if not config.VISION_LLM_ENABLED:
        return None

    provider = config.VISION_LLM_PROVIDER
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        logger.info(
            "Vision LLM: Ollama model=%s base_url=%s",
            config.VISION_LLM_MODEL,
            config.VISION_LLM_BASE_URL,
        )
        return ChatOllama(
            model=config.VISION_LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            base_url=config.VISION_LLM_BASE_URL,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = config.VISION_LLM_API_KEY or config.DEEPSEEK_API_KEY
        if not api_key:
            logger.warning("Vision LLM disabled: missing VISION_LLM_API_KEY / DEEPSEEK_API_KEY")
            return None
        return ChatOpenAI(
            model=config.VISION_LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            api_key=api_key,
            base_url=config.VISION_LLM_BASE_URL or None,
        )

    logger.warning("Unsupported VISION_LLM_PROVIDER: %s", provider)
    return None


def invoke_text_llm(llm: BaseChatModel, prompt: str) -> Optional[str]:
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = getattr(response, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
        if content:
            return str(content).strip()
        return None
    except Exception as exc:
        logger.warning("LLM invocation failed: %s", exc)
        return None


def invoke_vision_llm(
    llm: BaseChatModel,
    prompt: str,
    image_path: str,
    ocr_text: Optional[str] = None,
) -> Optional[str]:
    path = Path(image_path)
    if not path.exists():
        logger.warning("Image not found for captioning: %s", image_path)
        return None

    final_prompt = prompt
    if ocr_text and ocr_text.strip() and "{ocr_text}" in prompt:
        final_prompt = prompt.replace("{ocr_text}", ocr_text.strip())

    suffix = path.suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    message = HumanMessage(
        content=[
            {"type": "text", "text": final_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{encoded}"}},
        ]
    )
    try:
        response = llm.invoke([message])
        content = getattr(response, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
        if content:
            return str(content).strip()
        return None
    except Exception as exc:
        logger.warning("Vision LLM invocation failed for %s: %s", image_path, exc)
        return None
