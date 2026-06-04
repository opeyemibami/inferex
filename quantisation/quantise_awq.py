"""Quantise Llama-3.1-8B-Instruct to 4-bit AWQ using AutoAWQ."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow direct execution: uv run python quantisation/awq.py from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structlog
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

from quantisation.calibration_prompts import load_calibration_data

MODEL_ID: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SAVE_PATH: Path = Path("~/.cache/huggingface/hub/Llama-3.1-8B-AWQ").expanduser()

# 4-bit GEMM variant: optimised for batch inference (batch size > 1).
# Use version="GEMV" if the production workload is exclusively batch-size-1 decode.
QUANT_CONFIG: dict = {
    "zero_point": True,   # asymmetric quantisation — standard for AWQ
    "q_group_size": 128,  # per-group scaling; smaller = more accurate, larger = more compressible
    "w_bit": 4,
    "version": "GEMM",
}

logger = structlog.get_logger()


def quantise() -> None:
    """Load the model in FP16, calibrate on wikitext-2 plus inference prompts, quantise to 4-bit AWQ, and save."""
    calibration_data: list[str] = load_calibration_data(n_wikitext=128)
    logger.info("calibration_ready", total_samples=len(calibration_data))

    logger.info("loading_tokenizer", model_id=MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    logger.info("loading_model_fp16", model_id=MODEL_ID)
    model = AutoAWQForCausalLM.from_pretrained(
        MODEL_ID,
        low_cpu_mem_usage=True,
        use_cache=False,
    )

    logger.info("quantising", config=QUANT_CONFIG)
    model.quantize(tokenizer, quant_config=QUANT_CONFIG, calib_data=calibration_data)

    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    logger.info("saving", path=str(SAVE_PATH))
    model.save_quantized(str(SAVE_PATH))
    tokenizer.save_pretrained(str(SAVE_PATH))
    logger.info("done", path=str(SAVE_PATH))


if __name__ == "__main__":
    quantise()
