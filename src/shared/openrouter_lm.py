from __future__ import annotations

import dspy

from app.settings import Settings


def configure_dspy_lm(settings: Settings) -> None:
    # LiteLLM reads provider API keys from the environment automatically.
    # The model string determines the provider:
    #   openrouter/<provider>/<model>  → reads OPENROUTER_API_KEY
    #   openai/<model>                 → reads OPENAI_API_KEY / OPENAI_API_BASE
    #   anthropic/<model>              → reads ANTHROPIC_API_KEY
    #   ollama/<model>                 → reads OLLAMA_API_BASE (or use openai/ prefix)
    #   … and so on for all LiteLLM-supported providers
    dspy.configure(
        lm=dspy.LM(settings.lm_model, temperature=settings.lm_temperature, cache=False)
    )
