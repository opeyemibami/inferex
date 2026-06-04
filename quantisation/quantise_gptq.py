"""Quantise Llama-3.1-8B-Instruct to 4-bit GPTQ using Hugging Face Optimum."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow direct execution: uv run python quantisation/gptq.py from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structlog
import torch
from optimum.gptq import GPTQQuantizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantisation.calibration_prompts import load_calibration_data

MODEL_ID: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SAVE_PATH: Path = Path("~/.cache/huggingface/hub/Llama-3.1-8B-GPTQ").expanduser()

# desc_act=False: disables activation reordering.
# desc_act=True improves accuracy by grouping weights with similar activation magnitudes,
# but disrupts contiguous memory access during inference, increasing decode latency.
# For throughput-focused benchmarking, False is the correct default.
BITS: int = 4
GROUP_SIZE: int = 128
DESC_ACT: bool = False
MODEL_SEQLEN: int = 2048

logger = structlog.get_logger()


def quantise() -> None:
    """Load the model in FP16, calibrate on wikitext-2 plus inference prompts, quantise to 4-bit GPTQ, and save."""
    calibration_data: list[str] = load_calibration_data(n_wikitext=128)
    logger.info("calibration_ready", total_samples=len(calibration_data))

    logger.info("loading_tokenizer", model_id=MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("loading_model_fp16", model_id=MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    logger.info(
        "quantising",
        bits=BITS,
        group_size=GROUP_SIZE,
        desc_act=DESC_ACT,
        model_seqlen=MODEL_SEQLEN,
    )
    quantizer = GPTQQuantizer(
        bits=BITS,
        dataset=calibration_data,
        group_size=GROUP_SIZE,
        desc_act=DESC_ACT,
        model_seqlen=MODEL_SEQLEN,
    )
    quantized_model = quantizer.quantize_model(model, tokenizer)

    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    logger.info("saving", path=str(SAVE_PATH))
    quantized_model.save_pretrained(str(SAVE_PATH))
    tokenizer.save_pretrained(str(SAVE_PATH))
    logger.info("done", path=str(SAVE_PATH))


if __name__ == "__main__":
    quantise()
