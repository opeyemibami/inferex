FROM python:3.11-slim

WORKDIR /app

# Bring in uv without a separate install step
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install gateway runtime deps only — vLLM runs as a separate service
RUN uv pip install --system \
    "fastapi>=0.136.3" \
    "httpx>=0.28.1" \
    "pydantic>=2.13.4" \
    "structlog>=25.5.0" \
    "uvicorn>=0.48.0"

COPY gateway/ ./gateway/

EXPOSE 8000

CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
