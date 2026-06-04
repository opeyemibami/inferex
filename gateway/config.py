import os

VLLM_BASE_URL: str = os.getenv("GATEWAY_VLLM_URL", "http://localhost:8001")

VALID_API_KEYS: frozenset[str] = frozenset(
    k.strip()
    for k in os.getenv("GATEWAY_API_KEYS", "dev-key-change-me").split(",")
    if k.strip()
)

RATE_LIMIT_REQUESTS: int = int(os.getenv("GATEWAY_RATE_LIMIT_REQUESTS", "60"))
RATE_LIMIT_WINDOW_SECONDS: int = int(
    os.getenv("GATEWAY_RATE_LIMIT_WINDOW_SECONDS", "60")
)

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT: str = os.getenv("LOG_FORMAT", "console")  # "console" | "json"
