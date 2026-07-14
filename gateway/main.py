import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

import httpx
import structlog
from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from gateway.config import LOG_FORMAT, LOG_LEVEL, VLLM_BASE_URL
from gateway.logging_config import configure_logging
from gateway.rate_limiter import check_rate_limit

configure_logging(LOG_LEVEL, LOG_FORMAT)
logger = structlog.get_logger()

_STRIP_REQUEST_HEADERS = frozenset(
    {"host", "x-api-key", "content-length", "transfer-encoding"}
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage the shared httpx client across the application lifetime."""
    async with httpx.AsyncClient(
        base_url=VLLM_BASE_URL,
        timeout=httpx.Timeout(connect=5.0, read=300.0, write=30.0, pool=5.0),
    ) as client:
        app.state.http_client = client
        logger.info("gateway_started", vllm_url=VLLM_BASE_URL)
        yield
    logger.info("gateway_stopped")


app = FastAPI(title="inferex-gateway", version="0.1.0", lifespan=lifespan)


@app.get("/health", tags=["ops"])
async def health() -> dict:
    """Return gateway liveness status."""
    return {"status": "ok"}


def _forward_headers(request: Request) -> dict[str, str]:
    """Strip hop-by-hop and auth headers before forwarding the request to vLLM."""
    return {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in _STRIP_REQUEST_HEADERS
    }


def _parse_usage(data: dict) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """Extract prompt, completion, and total token counts from a vLLM response body."""
    usage = data.get("usage") or {}
    return (
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
        usage.get("total_tokens"),
    )


def _mask_key(api_key: str) -> str:
    """Return a partially masked API key safe for log output."""
    return api_key[:4] + "..." if len(api_key) > 4 else "***"


@app.api_route(
    "/v1/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE"],
    tags=["proxy"],
)
async def proxy(
    path: str,
    request: Request,
    api_key: str = Depends(check_rate_limit),
) -> Response:
    """Forward authenticated requests to vLLM and log model, tokens, and latency."""
    client: httpx.AsyncClient = request.app.state.http_client
    body = await request.body()

    req_data: dict[str, Any] = {}
    if body:
        try:
            req_data = json.loads(body)
        except json.JSONDecodeError:
            pass

    model: str = req_data.get("model", "unknown")
    is_streaming: bool = bool(req_data.get("stream", False))
    request_id: str = str(uuid.uuid4())
    start: float = time.perf_counter()

    log = logger.bind(
        request_id=request_id,
        path=f"/v1/{path}",
        model=model,
        api_key=_mask_key(api_key),
    )
    log.info("request_received")

    headers = _forward_headers(request)

    if is_streaming:
        return _build_streaming_response(
            client, request, path, body, headers, log, start, model
        )

    try:
        upstream = await client.request(
            method=request.method,
            url=f"/v1/{path}",
            content=body,
            headers=headers,
            params=dict(request.query_params),
        )
    except httpx.ConnectError:
        log.error("vllm_unreachable")
        return Response(
            content=json.dumps({"error": "upstream unavailable"}),
            status_code=503,
            media_type="application/json",
        )
    except httpx.TimeoutException:
        log.error("vllm_timeout")
        return Response(
            content=json.dumps({"error": "upstream timeout"}),
            status_code=504,
            media_type="application/json",
        )

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    prompt_tokens = completion_tokens = total_tokens = None

    if upstream.status_code == 200:
        try:
            prompt_tokens, completion_tokens, total_tokens = _parse_usage(
                upstream.json()
            )
        except Exception:
            pass

    log.info(
        "request_completed",
        status_code=upstream.status_code,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_ms=latency_ms,
        streaming=False,
    )

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=dict(upstream.headers),
        media_type=upstream.headers.get("content-type"),
    )


def _build_streaming_response(
    client: httpx.AsyncClient,
    request: Request,
    path: str,
    body: bytes,
    headers: dict[str, str],
    log: Any,
    start: float,
    model: str,
) -> StreamingResponse:
    """Build a StreamingResponse that proxies SSE chunks and logs usage on completion."""

    async def generate() -> AsyncIterator[bytes]:
        """Yield upstream SSE chunks and log token usage after the stream closes."""
        usage: dict[str, Any] = {}
        # Buffer for reassembling SSE lines across chunk boundaries
        line_buf = ""

        try:
            async with client.stream(
                method=request.method,
                url=f"/v1/{path}",
                content=body,
                headers=headers,
                params=dict(request.query_params),
            ) as upstream:
                async for chunk in upstream.aiter_bytes():
                    line_buf += chunk.decode(errors="replace")
                    # Process all complete lines for usage extraction
                    while "\n" in line_buf:
                        line, line_buf = line_buf.split("\n", 1)
                        line = line.rstrip("\r")
                        if line.startswith("data: ") and line != "data: [DONE]":
                            try:
                                data = json.loads(line[6:])
                                if data.get("usage"):
                                    usage = data["usage"]
                            except json.JSONDecodeError:
                                pass
                    yield chunk

        except httpx.ConnectError:
            log.error("vllm_unreachable")
            yield json.dumps({"error": "upstream unavailable"}).encode()
        except httpx.TimeoutException:
            log.error("vllm_timeout")
            yield json.dumps({"error": "upstream timeout"}).encode()
        finally:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            log.info(
                "request_completed",
                model=model,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
                latency_ms=latency_ms,
                streaming=True,
            )

    return StreamingResponse(generate(), media_type="text/event-stream")
