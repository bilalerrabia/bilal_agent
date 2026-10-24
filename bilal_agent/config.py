"""Configuration loading: config.toml (providers, limits, safety) + .env (secrets)."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

# Used when no config.toml can be found anywhere - mirrors config.toml in the repo.
_BUILTIN = {
    "active": "zenmux",
    "providers": {
        "zenmux": {
            "base_url": "https://zenmux.ai/api/v1",
            "model": "z-ai/glm-5.3-flash",
            "api_key_env": "ZENMUX_API_KEY",
        }
    },
    "agent": {"max_iterations": 60, "max_output_tokens": 8192, "llm_timeout": 300, "stream": True},
    "context": {"max_tokens": 60000, "keep_recent": 8},
    "safety": {"confirm_edits": True, "command_timeout": 60},
}

# List prices for cost estimation, $ per 1M tokens: (model substring, input, output).
# Prices drift - override any of this in config.toml [pricing]; /cost shows the source.
_PRICING_TABLE = [
    ("claude-opus", 15.0, 75.0),
    ("claude-3-opus", 15.0, 75.0),
    ("claude-sonnet", 3.0, 15.0),
    ("claude-3-7", 3.0, 15.0),
    ("claude-haiku", 0.8, 4.0),
    ("gpt-4o", 2.5, 10.0),
    ("gpt-4.1", 2.0, 8.0),
    ("gpt-5", 1.25, 10.0),
    ("o3", 2.0, 8.0),
    ("o4-mini", 1.1, 4.4),
    ("gemini-2.5-pro", 1.25, 10.0),
    ("gemini", 0.15, 0.6),
    ("deepseek", 0.27, 1.1),
    ("glm", 0.6, 2.2),
    ("qwen", 0.4, 1.2),
    ("mistral", 2.0, 6.0),
    ("llama", 0.9, 0.9),
]


def resolve_pricing(cfg) -> tuple[float, float, str]:
    """(input $/1M tokens, output $/1M tokens, source note) for cost estimation.

    Order: explicit [pricing] in config.toml wins, then a list-price lookup by
    model-name substring, then a generic fallback. The source is shown by /cost
    so dollar figures are never silently wrong.
    """
    p = getattr(cfg, "pricing", None) or {}
    if p.get("input_per_m") or p.get("output_per_m"):
        return (
            float(p.get("input_per_m") or 0.0),
            float(p.get("output_per_m") or 0.0),
            "config.toml [pricing]",
        )
    model = (getattr(getattr(cfg, "provider", None), "model", "") or "").lower()
    for needle, pin, pout in _PRICING_TABLE:
        if needle in model:
            return pin, pout, f"list price for '{needle}*'"
    return 1.0, 3.0, "generic fallback (set [pricing] in config.toml)"


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    model: str
    api_key_env: str


@dataclass
class Config:
    provider: ProviderConfig
    max_iterations: int = 60
    max_output_tokens: int = 8192
    llm_timeout: int = 300        # read timeout for ONE llm request (seconds)
    stream: bool = True           # stream tokens live as they arrive
    max_context_tokens: int = 60000
    keep_recent: int = 8
    confirm_edits: bool = True
    command_timeout: int = 60
    auto_yes: bool = False  # set by CLI --yes
    temperature: float | None = None  # passed to the API when set in [agent]
    budget_usd: float | None = None   # soft session budget - warns at 80% / 100%
    pricing: dict = field(default_factory=dict)  # {input_per_m, output_per_m} - see resolve_pricing
    config_path: Path | None = None
    raw: dict = field(default_factory=dict)


def load_dotenv(paths: Path | list[Path] | None = None) -> list[tuple[Path, list[str]]]:
    """Load KEY=VALUE from .env files.

    - real environment variables always win (we never override them)
    - files are loaded in order; the first file that defines a key wins
    - accepts a single path or a list; returns [(path, [keys loaded])]
    """
    if paths is None:
        paths = [Path(".env")]
    if isinstance(paths, (str, Path)):
        paths = [Path(paths)]
    loaded: list[tuple[Path, list[str]]] = []
    for env_file in paths:
        env_file = Path(env_file)
        if not env_file.is_file():
            continue
        try:
            text = env_file.read_text(encoding="utf-8")
        except OSError:
            continue
        keys: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and not key.startswith("#") and key not in os.environ:
                os.environ[key] = value
                keys.append(key)
        loaded.append((env_file, keys))
    return loaded


def _find_config(explicit: str | None, root: Path | None = None) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"config file not found: {explicit}")
        return p
    candidates = []
    if root is not None:
        candidates.append(root / "config.toml")
    candidates += [Path.cwd() / "config.toml", Path.home() / ".config" / "bilal" / "config.toml"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_config(
    config_path: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    auto_yes: bool = False,
    root: Path | None = None,
) -> Config:
    path = _find_config(config_path, root)
    if path is not None:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    else:
        raw = dict(_BUILTIN)

    active = provider or raw.get("active")
    providers: dict = raw.get("providers", {})
    if active not in providers:
        known = ", ".join(providers) or "(none defined)"
        raise ValueError(f"provider '{active}' not in config [providers] - known: {known}")

    p = providers[active]
    prov = ProviderConfig(
        name=active,
        base_url=str(p.get("base_url", "")).rstrip("/"),
        model=model or str(p.get("model", "")),
        api_key_env=str(p.get("api_key_env", "")),
    )
    if not prov.base_url or not prov.model:
        raise ValueError(f"provider '{active}' needs base_url and model in config.toml")

    agent_cfg = raw.get("agent", {})
    ctx_cfg = raw.get("context", {})
    safety_cfg = raw.get("safety", {})
    pricing_cfg = raw.get("pricing", {})
    cfg = Config(
        provider=prov,
        max_iterations=int(agent_cfg.get("max_iterations", 60)),
        max_output_tokens=int(agent_cfg.get("max_output_tokens", 8192)),
        llm_timeout=int(agent_cfg.get("llm_timeout", 300)),
        stream=bool(agent_cfg.get("stream", True)),
        max_context_tokens=int(ctx_cfg.get("max_tokens", 60000)),
        keep_recent=int(ctx_cfg.get("keep_recent", 8)),
        confirm_edits=bool(safety_cfg.get("confirm_edits", True)),
        command_timeout=int(safety_cfg.get("command_timeout", 60)),
        auto_yes=auto_yes,
        temperature=(float(agent_cfg["temperature"]) if agent_cfg.get("temperature") is not None else None),
        budget_usd=(float(agent_cfg["budget_usd"]) if agent_cfg.get("budget_usd") else None),
        pricing={k: float(v) for k, v in pricing_cfg.items()},
        config_path=path,
        raw=raw,
    )
    return cfg


def require_api_key(cfg: Config) -> str | None:
    """Return the env var name that must hold the API key, or None if none needed."""
    env_name = cfg.provider.api_key_env
    if not env_name:
        return None
    if os.environ.get(env_name):
        return None
    return env_name


def with_overrides(cfg: Config, **kwargs) -> Config:
    return replace(cfg, **kwargs)
