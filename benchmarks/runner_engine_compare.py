"""vLLM vs SGLang benchmark runner.

Usage: uv run python benchmarks/runner_engine_compare.py --engine all
       uv run python benchmarks/runner_engine_compare.py --engine vllm
       uv run python benchmarks/runner_engine_compare.py --engine sglang
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow direct execution from project root: uv run python benchmarks/runner_engine_compare.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import json
import os
import subprocess
import time
from typing import Any

import httpx
import structlog

from benchmarks.prompts import PROMPTS
from benchmarks.runner import get_vram_mb, run_concurrency_level, run_single  # noqa: F401


# Constants
MODEL: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MAX_MODEL_LEN: int = 4096
VLLM_PORT: int = 8001
SGLANG_PORT: int = 8002
HEALTH_TIMEOUT_S: int = 900
HEALTH_POLL_INTERVAL_S: int = 5
REQUEST_TIMEOUT_S: float = 120.0
CONCURRENCY_LEVELS: list[int] = [1, 8, 32]
RESULTS_DIR: Path = Path("benchmarks/results")

logger = structlog.get_logger()



# Engine process management
def start_vllm_engine() -> subprocess.Popen:
    """Start vLLM in Docker on port 8001 and wait until healthy."""
    cmd: list[str] = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "--network", "host",
        "-e", f"HUGGING_FACE_HUB_TOKEN={os.environ.get('HUGGING_FACE_HUB_TOKEN', '')}",
        "-v", f"{Path.home()}/.cache/huggingface:/root/.cache/huggingface",
        "-v", f"{Path.home()}/.cache/vllm:/root/.cache/vllm",
        "vllm/vllm-openai:latest",
        "--model", MODEL,
        "--port", str(VLLM_PORT),
        "--max-model-len", str(MAX_MODEL_LEN),
    ]

    logger.info("starting_vllm_engine", cmd=" ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=None, stderr=None)

    health_url = f"http://localhost:{VLLM_PORT}/health"
    deadline = time.monotonic() + HEALTH_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(health_url, timeout=5.0)
            if resp.status_code == 200:
                logger.info("vllm_engine_healthy")
                return proc
        except httpx.RequestError:
            pass
        time.sleep(HEALTH_POLL_INTERVAL_S)

    proc.terminate()
    proc.wait()
    raise RuntimeError(f"vLLM health check timed out after {HEALTH_TIMEOUT_S}s")


def start_sglang_engine() -> subprocess.Popen:
    """Start SGLang in Docker on port 8002 and wait until healthy."""
    cmd: list[str] = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "--network", "host",
        "-e", f"HUGGING_FACE_HUB_TOKEN={os.environ.get('HUGGING_FACE_HUB_TOKEN', '')}",
        "-v", f"{Path.home()}/.cache/huggingface:/root/.cache/huggingface",
        "lmsysorg/sglang:latest",
        "python", "-m", "sglang.launch_server",
        "--model-path", MODEL,
        "--port", str(SGLANG_PORT),
        "--host", "0.0.0.0",
        "--enable-metrics",
        "--disable-overlap-schedule",
    ]

    logger.info("starting_sglang_engine", cmd=" ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=None, stderr=None)

    health_url = f"http://localhost:{SGLANG_PORT}/health"
    deadline = time.monotonic() + HEALTH_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(health_url, timeout=5.0)
            if resp.status_code == 200:
                logger.info("sglang_engine_healthy")
                return proc
        except httpx.RequestError:
            pass
        time.sleep(HEALTH_POLL_INTERVAL_S)

    proc.terminate()
    proc.wait()
    raise RuntimeError(f"SGLang health check timed out after {HEALTH_TIMEOUT_S}s")


def stop_engine(proc: subprocess.Popen) -> None:
    """Terminate an engine subprocess and wait for it to exit."""
    logger.info("stopping_engine")
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        logger.warning("engine_sigterm_timeout_killing")
        proc.kill()
        proc.wait()
    logger.info("engine_stopped")



# Async level runner
async def _run_level_at(
    prompts: list[str],
    concurrency: int,
    base_url: str,
) -> list[dict]:
    """Create a fresh httpx client at base_url and run one concurrency level."""
    async with httpx.AsyncClient(
        base_url=base_url,
        timeout=httpx.Timeout(REQUEST_TIMEOUT_S),
    ) as client:
        return await run_concurrency_level(prompts, concurrency, MODEL, client)



# Benchmark orchestration
ENGINES: list[dict] = [
    {
        "name": "vllm",
        "base_url": f"http://localhost:{VLLM_PORT}",
        "result_file": "vllm.json",
        "start_fn": start_vllm_engine,
    },
    {
        "name": "sglang",
        "base_url": f"http://localhost:{SGLANG_PORT}",
        "result_file": "sglang.json",
        "start_fn": start_sglang_engine,
    },
]


def benchmark_engine(engine: dict) -> None:
    """Run all concurrency levels for one engine, record VRAM, and save results to JSON."""
    log = logger.bind(engine=engine["name"])
    log.info("benchmark_start", model=MODEL)

    all_prompts: list[str] = [p["text"] for p in PROMPTS]
    concurrency_results: dict[str, dict] = {}

    proc = engine["start_fn"]()
    try:
        for concurrency in CONCURRENCY_LEVELS:
            log.info("running_concurrency_level", concurrency=concurrency)

            level_results = asyncio.run(
                _run_level_at(all_prompts, concurrency, engine["base_url"])
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
        stop_engine(proc)

    output: dict[str, Any] = {
        "engine": engine["name"],
        "model": MODEL,
        "concurrency_levels_tested": CONCURRENCY_LEVELS,
        "concurrency_results": concurrency_results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / engine["result_file"]
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    log.info("results_saved", path=str(output_path))



# Entry point
def main() -> None:
    """Parse --engine and run the requested engine benchmarks in sequence."""
    parser = argparse.ArgumentParser(
        description="inferex phase 4 vLLM vs SGLang benchmark runner"
    )
    parser.add_argument(
        "--engine",
        required=True,
        choices=[e["name"] for e in ENGINES] + ["all"],
        help="Engine to benchmark, or 'all' to run both in sequence.",
    )
    args = parser.parse_args()

    engines_to_run = (
        ENGINES
        if args.engine == "all"
        else [e for e in ENGINES if e["name"] == args.engine]
    )

    for engine in engines_to_run:
        benchmark_engine(engine)

    logger.info("all_benchmarks_complete")


if __name__ == "__main__":
    main()
