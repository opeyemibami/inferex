"""Phase 2 quantisation benchmark runner.

Usage: uv run python benchmarks/runner.py --variant all
       uv run python benchmarks/runner.py --variant fp16
"""

from __future__ import annotations

import sys
from pathlib import Path
import os

# Allow direct execution from project root: uv run python benchmarks/runner.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import json
import subprocess
import time
from typing import Any

import httpx
import structlog

from benchmarks.prompts import PROMPTS


# Constants — all tuneable values live here, not inside functions
VLLM_PORT: int = 8001
VLLM_BASE_URL: str = f"http://localhost:{VLLM_PORT}"
HEALTH_ENDPOINT: str = f"{VLLM_BASE_URL}/health"
HEALTH_TIMEOUT_S: int = 900
HEALTH_POLL_INTERVAL_S: int = 5
MAX_TOKENS: int = 200
MAX_MODEL_LEN: int = 4096
REQUEST_TIMEOUT_S: float = 120.0
CONCURRENCY_LEVELS: list[int] = [1, 8, 32]
RESULTS_DIR: Path = Path("benchmarks/results")

VARIANTS: list[dict] = [
    {
        "name": "fp16",
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "extra_args": [],
    },
    {
        "name": "awq",
        "model": "/root/.cache/huggingface/hub/Llama-3.1-8B-AWQ",
        "extra_args": ["--quantization", "awq_marlin", "--dtype", "float16"],
        },
    {
        "name": "gptq",
        "model": "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16",
        "extra_args": ["--quantization", "gptq"],
    },
    {
        "name": "fp8",
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "extra_args": ["--quantization", "fp8"],
    },
]

logger = structlog.get_logger()



# vLLM process management
def start_vllm(variant: dict) -> subprocess.Popen:
    """Start vllm serve in Docker for the given variant."""
    cmd: list[str] = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "--network", "host",
        "-e", f"HUGGING_FACE_HUB_TOKEN={os.environ.get('HUGGING_FACE_HUB_TOKEN', '')}",
        "-v", f"{Path.home()}/.cache/huggingface:/root/.cache/huggingface",
        "-v", f"{Path.home()}/.cache/vllm:/root/.cache/vllm",
        "vllm/vllm-openai:latest",
        "--model", variant["model"],
        "--port", str(VLLM_PORT),
        "--max-model-len", str(MAX_MODEL_LEN),
    ] + variant["extra_args"]

    log = logger.bind(variant=variant["name"], model=variant["model"])
    log.info("starting_vllm", cmd=" ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=None,  # inherit parent stdout
        stderr=None,  # inherit parent stderr
    )

    deadline = time.monotonic() + HEALTH_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(HEALTH_ENDPOINT, timeout=5.0)
            if resp.status_code == 200:
                log.info("vllm_healthy")
                return proc
        except httpx.RequestError:
            pass
        time.sleep(HEALTH_POLL_INTERVAL_S)

    proc.terminate()
    proc.wait()
    raise RuntimeError(
        f"vLLM health check timed out after {HEALTH_TIMEOUT_S}s "
        f"(variant={variant['name']})"
    )


def stop_vllm(proc: subprocess.Popen) -> None:
    """Terminate the vLLM subprocess and wait for it to exit."""
    logger.info("stopping_vllm")
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        logger.warning("vllm_sigterm_timeout_killing")
        proc.kill()
        proc.wait()
    logger.info("vllm_stopped")



# System metrics
def get_vram_mb() -> int | None:
    """Query nvidia-smi for current VRAM usage in MiB on the first GPU.
    
    Returns None if the value is a sentinel/error (e.g. during Docker GPU runs
    where nvidia-smi may return an overflow value instead of a real reading).
    """
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    try:
        value = int(result.stdout.strip().splitlines()[0])
        if value > 40_000:  # RTX 5090 has 32GB; anything higher is a sentinel
            return None
        return value
    except (ValueError, IndexError):
        return None



# Async HTTP request logic
async def run_single(
    prompt: str,
    model_name: str,
    client: httpx.AsyncClient,
) -> dict:
    """Send one non-streaming chat completion and return latency and token metrics."""
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }

    start = time.perf_counter()
    try:
        response = await client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        elapsed_s = time.perf_counter() - start

        usage = response.json().get("usage", {})
        completion_tokens: int = usage.get("completion_tokens", 0)
        total_latency_ms = round(elapsed_s * 1000, 2)
        tokens_per_second = (
            round(completion_tokens / elapsed_s, 2)
            if elapsed_s > 0 and completion_tokens > 0
            else 0.0
        )

        return {
            "total_latency_ms": total_latency_ms,
            "completion_tokens": completion_tokens,
            "tokens_per_second": tokens_per_second,
        }

    except httpx.HTTPStatusError as exc:
        logger.error(
            "request_http_error",
            status_code=exc.response.status_code,
            prompt_prefix=prompt[:60],
        )
        return {
            "total_latency_ms": None,
            "completion_tokens": None,
            "tokens_per_second": None,
            "error": f"HTTP {exc.response.status_code}",
        }
    except httpx.RequestError as exc:
        logger.error("request_network_error", error=str(exc), prompt_prefix=prompt[:60])
        return {
            "total_latency_ms": None,
            "completion_tokens": None,
            "tokens_per_second": None,
            "error": str(exc),
        }


async def run_concurrency_level(
    prompts: list[str],
    concurrency: int,
    model_name: str,
    client: httpx.AsyncClient,
) -> list[dict]:
    """Send all prompts in batches of `concurrency` using asyncio.gather."""
    results: list[dict] = []
    for i in range(0, len(prompts), concurrency):
        batch = prompts[i : i + concurrency]
        batch_results = await asyncio.gather(
            *[run_single(p, model_name, client) for p in batch]
        )
        results.extend(batch_results)
    return results


async def _run_level(
    prompts: list[str],
    concurrency: int,
    model_name: str,
) -> list[dict]:
    """Create a fresh httpx client and run one concurrency level end-to-end."""
    async with httpx.AsyncClient(
        base_url=VLLM_BASE_URL,
        timeout=httpx.Timeout(REQUEST_TIMEOUT_S),
    ) as client:
        return await run_concurrency_level(prompts, concurrency, model_name, client)



# Benchmark orchestration
def benchmark_variant(variant: dict) -> None:
    """Run all concurrency levels for one variant, record VRAM, and save results to JSON."""
    log = logger.bind(variant=variant["name"])
    log.info("benchmark_start", model=variant["model"])

    all_prompts: list[str] = [p["text"] for p in PROMPTS]
    concurrency_results: dict[str, dict] = {}

    proc = start_vllm(variant)
    try:
        for concurrency in CONCURRENCY_LEVELS:
            log.info("running_concurrency_level", concurrency=concurrency)

            level_results = asyncio.run(
                _run_level(all_prompts, concurrency, variant["model"])
            )
            vram_mb = get_vram_mb()

            concurrency_results[str(concurrency)] = {
                "results": level_results,
                "vram_used_mb": vram_mb,
            }
            log.info(
                "concurrency_level_done",
                concurrency=concurrency,
                requests=len(level_results),
                vram_used_mb=vram_mb,
            )
    finally:
        stop_vllm(proc)

    output: dict = {
        "variant": variant["name"],
        "model": variant["model"],
        "concurrency_levels_tested": CONCURRENCY_LEVELS,
        "concurrency_results": concurrency_results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{variant['name']}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    log.info("results_saved", path=str(output_path))



# Entry point
def main() -> None:
    """Parse --variant and run the requested benchmark variants in sequence."""
    parser = argparse.ArgumentParser(
        description="inferex quantisation benchmark runner"
    )
    parser.add_argument(
        "--variant",
        required=True,
        choices=[v["name"] for v in VARIANTS] + ["all"],
        help="Variant to benchmark, or 'all' to run every variant in sequence.",
    )
    args = parser.parse_args()

    variants_to_run = (
        VARIANTS
        if args.variant == "all"
        else [v for v in VARIANTS if v["name"] == args.variant]
    )

    for variant in variants_to_run:
        benchmark_variant(variant)

    logger.info("all_benchmarks_complete")


if __name__ == "__main__":
    main()
