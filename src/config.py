"""
Centralized configuration for the Metadata Agent.

All configurable settings should be defined here.
You can also override these via environment variables.

Supported LLM providers (set ``LLM_PROVIDER``):

- ``google`` — Gemini via ``langchain_google_genai`` (``GOOGLE_API_KEY``, optional ``LLM_MODEL``).
- ``openai`` — OpenAI Chat Completions (``OPENAI_API_KEY``, optional ``OPENAI_API_BASE`` for proxies/Azure-style base URLs, optional ``LLM_MODEL``).
- ``qwen`` — OpenAI-compatible API (``QWEN_API_BASE``, ``QWEN_API_KEY``, optional ``LLM_MODEL``), e.g. DashScope compatible-mode.
- ``surf`` — Custom OpenAI-compatible endpoint (``SURF_API_BASE``, ``SURF_API_KEY``).

Switching models: set ``LLM_PROVIDER`` and ``LLM_MODEL`` (and the matching API key / base URL), then restart the process or reload the package.
"""
import os
from typing import Optional, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def _env_strip(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read env var and strip whitespace (handles `KEY = value` in .env)."""
    v = os.getenv(name, default)
    if v is None:
        return None
    return v.strip() or None


# =============================================================================
# LLM PROVIDER CONFIGURATION
# =============================================================================

# LLM Provider: "google", "surf", "openai", "qwen"
# Can be overridden by environment variable: LLM_PROVIDER
LLM_PROVIDER = (_env_strip("LLM_PROVIDER", "google") or "google").lower()

# Provider-specific configurations
PROVIDER_CONFIGS = {
    "google": {
        "default_model": "gemini-2.5-flash",
        "api_key_env": "GOOGLE_API_KEY",
        "description": "Google Gemini models",
    },
    "surf": {
        "default_model": "Qwen2.5-Coder-7B-Instruct",
        "api_key_env": "SURF_API_KEY",
        "base_url_env": "SURF_API_BASE",
        "description": "Custom OpenAI-compatible endpoint (e.g., vLLM, TGI)",
    },
    "openai": {
        "default_model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
        "description": "OpenAI models",
    },
    "qwen": {
        "default_model": "Qwen2.5-7B-Instruct",
        "api_key_env": "QWEN_API_KEY",
        "base_url_env": "QWEN_API_BASE",
        "description": "Qwen models via OpenAI-compatible endpoint",
    },
}

# =============================================================================
# LLM MODEL CONFIGURATION
# =============================================================================

# Default model - uses provider's default if not specified
# Can be overridden by environment variable: LLM_MODEL
DEFAULT_MODEL = _env_strip("LLM_MODEL", None)  # None means use provider default

# Default temperature for planning (lower = more deterministic)
# Can be overridden by environment variable: LLM_TEMPERATURE_PLANNING
PLANNING_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE_PLANNING", "0.0"))

# Default temperature for players (higher = more creative)
# Lowering this makes player outputs more deterministic and more likely to
# follow strict JSON formatting instructions in their prompts.
# Can be overridden by environment variable: LLM_TEMPERATURE_PLAYER
PLAYER_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE_PLAYER", "0.0"))


# =============================================================================
# PROVIDER-SPECIFIC API KEYS AND ENDPOINTS
# =============================================================================

# Google (Gemini)
GOOGLE_API_KEY = _env_strip("GOOGLE_API_KEY")

# Surf (custom OpenAI-compatible endpoint)
SURF_API_BASE = _env_strip("SURF_API_BASE")
SURF_API_KEY = _env_strip("SURF_API_KEY")

# Qwen (OpenAI-compatible endpoint, e.g. Alibaba DashScope compatible-mode)
QWEN_API_BASE = _env_strip("QWEN_API_BASE")
QWEN_API_KEY = _env_strip("QWEN_API_KEY")

# OpenAI (optional base URL for Azure-style / corporate proxy endpoints)
OPENAI_API_KEY = _env_strip("OPENAI_API_KEY")
OPENAI_API_BASE = _env_strip("OPENAI_API_BASE")


# =============================================================================
# EXECUTION DEFAULTS
# =============================================================================

# Default execution topology
# Can be overridden by environment variable: DEFAULT_TOPOLOGY
DEFAULT_TOPOLOGY = os.getenv("DEFAULT_TOPOLOGY", "default")

# Default metadata standard
# Can be overridden by environment variable: DEFAULT_METADATA_STANDARD
DEFAULT_METADATA_STANDARD = os.getenv("DEFAULT_METADATA_STANDARD", "basic")


# =============================================================================
# LLM FACTORY
# =============================================================================

# Normalize common alternate spellings to DashScope model id (dot form).
_QWEN_MODEL_ALIASES = {
    "qwen3-5-flash": "qwen3.5-flash",
    "qwen3-5-flash-preview": "qwen3.5-flash",
    "qwen_qwen3-5-flash": "qwen3.5-flash",
    "qwen_qwen3.5-flash": "qwen3.5-flash",
}


def structured_output_kwargs(provider: Optional[str] = None) -> dict:
    """
    Extra kwargs for ``ChatModel.with_structured_output(schema, **kwargs)``.

    Strict schema mode is preferred for Pydantic-backed extraction:
    - method="json_schema"
    - strict=True
    """
    _ = (provider or LLM_PROVIDER).lower()
    return {
        "method": "json_schema",
        "strict": True,
    }


def get_model_name(override: Optional[str] = None) -> str:
    """
    Get the model name to use.
    
    Priority:
    1. Override parameter
    2. LLM_MODEL environment variable
    3. Provider's default model
    """
    if override:
        model = override
    elif DEFAULT_MODEL:
        model = DEFAULT_MODEL
    else:
        model = PROVIDER_CONFIGS.get(LLM_PROVIDER, {}).get("default_model", "gpt-4o-mini")

    if LLM_PROVIDER == "qwen":
        key = str(model).strip().lower()
        return _QWEN_MODEL_ALIASES.get(key, model)

    return model


def create_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    provider: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Factory function to create an LLM instance based on the configured provider.
    
    Args:
        model_name: Model name (uses default if not specified)
        temperature: LLM temperature
        provider: Override the default provider
        **kwargs: Additional arguments passed to the LLM constructor
        
    Returns:
        LangChain chat model instance
        
    Raises:
        ValueError: If provider is not supported or required config is missing
    """
    provider = provider or LLM_PROVIDER
    model = get_model_name(model_name)
    
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        if not GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY not found. Set it in your .env file."
            )
        
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=GOOGLE_API_KEY,
            **kwargs
        )
    
    elif provider == "surf":
        from langchain_openai import ChatOpenAI
        
        if not SURF_API_BASE:
            raise ValueError(
                "SURF_API_BASE not found. Set it in your .env file.\n"
                "Example: SURF_API_BASE=http://localhost:8000/v1"
            )
        
        if not SURF_API_KEY:
            raise ValueError(
                "SURF_API_KEY not found. Set it in your .env file."
            )
        
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=SURF_API_KEY,
            openai_api_base=SURF_API_BASE,
            **kwargs
        )
    
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not found. Set it in your .env file."
            )
        
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=OPENAI_API_BASE,
            **kwargs
        )
    
    elif provider == "qwen":
        from langchain_openai import ChatOpenAI

        if not QWEN_API_BASE:
            raise ValueError(
                "QWEN_API_BASE not found. Set it in your .env file.\n"
                "Example: QWEN_API_BASE=http://localhost:8000/v1"
            )

        if not QWEN_API_KEY:
            raise ValueError(
                "QWEN_API_KEY not found. Set it in your .env file."
            )

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=QWEN_API_KEY,
            openai_api_base=QWEN_API_BASE,
            **kwargs
        )

    else:
        available = list(PROVIDER_CONFIGS.keys())
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. Available: {available}"
        )


# =============================================================================
# CONFIGURATION SUMMARY
# =============================================================================

def get_config_summary() -> str:
    """Return a summary of current configuration."""
    provider_config = PROVIDER_CONFIGS.get(LLM_PROVIDER, {})
    model = get_model_name()
    
    # Check API key status
    api_key_env = provider_config.get("api_key_env", "")
    api_key_set = bool(os.getenv(api_key_env)) if api_key_env else False
    
    return f"""
Configuration Summary:
----------------------
LLM Provider: {LLM_PROVIDER} ({provider_config.get('description', 'Unknown')})
LLM Model: {model}
Planning Temperature: {PLANNING_TEMPERATURE}
Player Temperature: {PLAYER_TEMPERATURE}
Default Topology: {DEFAULT_TOPOLOGY}
Default Metadata Standard: {DEFAULT_METADATA_STANDARD}
API Key ({api_key_env}): {'Set' if api_key_set else 'Not Set'}
"""
