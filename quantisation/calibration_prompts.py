"""Calibration prompts and data loader shared by all quantisation scripts."""

from __future__ import annotations

import structlog

logger = structlog.get_logger()

CALIBRATION_PROMPTS: list[str] = [
    # LLM serving architecture
    (
        "Explain how continuous batching improves GPU utilisation in a production LLM serving "
        "system. What is the difference between static batching, dynamic batching, and continuous "
        "batching, and why does continuous batching matter specifically for variable-length "
        "generation tasks where individual sequences finish at unpredictable steps?"
    ),
    (
        "Describe PagedAttention and how it addresses memory fragmentation in the KV cache. "
        "How does vLLM implement PagedAttention differently from the naive approach of "
        "pre-allocating contiguous KV cache blocks per sequence, and what is the page table "
        "mechanism that maps logical blocks to physical GPU memory?"
    ),
    (
        "What is time to first token (TTFT) and how does it differ from tokens per second? "
        "In a production inference system, describe the workload characteristics that make "
        "TTFT the dominant SLO metric versus when inter-token latency or total throughput "
        "becomes more important for the end-user experience."
    ),
    (
        "Explain the prefill and decode phases of autoregressive language model inference. "
        "Why is the prefill phase compute-bound while the decode phase is typically "
        "memory-bandwidth-bound, and how does this asymmetry affect optimal batch size "
        "selection and GPU utilisation at different request arrival rates?"
    ),
    (
        "Describe how vLLM's scheduler manages requests when KV cache memory is exhausted. "
        "What is preemption in this context, what happens to a running sequence that is "
        "preempted, and how does the scheduler decide which sequences to preempt versus "
        "which to keep running when memory pressure increases?"
    ),
    # Quantisation algorithms
    (
        "Walk through the AWQ (Activation-aware Weight Quantisation) algorithm. What "
        "distinguishes it from naive round-to-nearest quantisation, why does protecting "
        "salient weights identified by activation magnitude improve quantised model accuracy, "
        "and what role does the per-channel scaling factor play in the weight transformation?"
    ),
    (
        "Explain the GPTQ algorithm and its use of the approximate inverse Hessian to "
        "determine optimal weight perturbations during quantisation. What is the role of "
        "the calibration dataset in estimating the Hessian, and how does quantising "
        "weights column-by-column with error compensation differ from independent "
        "per-layer quantisation?"
    ),
    (
        "What makes calibration data selection important for GPTQ and AWQ quantisation? "
        "Explain why domain-specific calibration data — prompts about LLM serving and "
        "GPU memory management rather than generic web text — might improve quantised "
        "model quality on inference engineering tasks while potentially degrading "
        "performance on unrelated domains."
    ),
    (
        "Describe post-training quantisation (PTQ) versus quantisation-aware training (QAT) "
        "for compressing large language models. When would you choose QAT over PTQ for a "
        "production serving use case, what additional infrastructure does QAT require, and "
        "at what model scale does the accuracy difference between PTQ and QAT become "
        "practically significant?"
    ),
    (
        "Explain the FP8 quantisation format and how it differs from INT4 methods like AWQ "
        "and GPTQ. What hardware support is required to execute FP8 matrix multiplications "
        "efficiently, and in what serving scenarios would you prefer FP8 over INT4 despite "
        "FP8 having a larger memory footprint per parameter?"
    ),
    # GPU memory management
    (
        "You are deploying Llama-3.1-8B-Instruct in FP16 on a single 24 GB GPU. Walk through "
        "the VRAM budget: model weights, KV cache at batch size 32 with sequence length 2048, "
        "activations, and CUDA overhead. At what batch size does the KV cache memory begin to "
        "dominate over model weight memory, and how does 4-bit quantisation change this budget?"
    ),
    (
        "Explain how GPU memory fragmentation occurs in an LLM serving system operating over "
        "many hours with variable sequence lengths. What is the difference between internal "
        "and external fragmentation in the KV cache context, and why does PagedAttention "
        "eliminate external fragmentation while accepting a small fixed overhead per page?"
    ),
    (
        "Describe the memory hierarchy relevant to LLM inference: HBM bandwidth, L2 cache "
        "capacity, and SRAM in streaming multiprocessors. In the decode phase of autoregressive "
        "generation at batch size 1, which memory level is the bottleneck, why does increasing "
        "batch size improve hardware efficiency, and at what batch size does the workload "
        "become compute-bound rather than memory-bandwidth-bound?"
    ),
    (
        "What is the relationship between model parallelism degree and optimal batch size in "
        "LLM inference? If you split Llama-3.1-8B across 4 GPUs using tensor parallelism, "
        "explain how all-reduce communication overhead changes the throughput-optimal batch "
        "size compared to single-GPU serving, and when the communication cost dominates."
    ),
    (
        "Describe the token budget problem in long-context LLM inference. At approximately "
        "what context length does KV cache memory per sequence exceed model weight memory "
        "for a 7B parameter model in FP16, and how does this inflection point change the "
        "maximum concurrent batch size you can sustain on a fixed VRAM budget?"
    ),
    # Attention mechanisms and optimisations
    (
        "Explain Flash Attention and why it reduces peak GPU memory usage during attention "
        "computation compared to standard attention. Describe the tiling strategy that allows "
        "the softmax and attention output to be computed without materialising the full "
        "N-by-N attention matrix in HBM, and why this matters most for long sequence lengths."
    ),
    (
        "Describe grouped-query attention (GQA) and multi-query attention (MQA) and how they "
        "differ from standard multi-head attention. How do GQA and MQA reduce KV cache memory "
        "requirements, and what is the trade-off in model capacity or generation quality that "
        "comes with reducing the number of key-value heads?"
    ),
    (
        "Explain chunked prefill and why it allows vLLM to interleave prefill and decode "
        "operations within the same scheduler iteration. What problem does it solve when a "
        "long-context prefill request arrives alongside decode-only requests, and what is "
        "the per-chunk overhead that limits how small the chunks can be made?"
    ),
    # Parallelism and scaling
    (
        "Describe tensor parallelism for transformer inference as implemented in Megatron-LM. "
        "How is the attention mechanism split across multiple GPUs — specifically the Q, K, V "
        "projections and the output projection — and what all-reduce operations are required "
        "at which points in the forward pass to maintain numerical equivalence?"
    ),
    (
        "Explain pipeline parallelism for LLM inference and how micro-batch scheduling reduces "
        "pipeline bubble overhead. In a 4-stage pipeline, what is the theoretical bubble "
        "fraction as a function of micro-batch count, and at what point does increasing "
        "micro-batches stop improving pipeline efficiency?"
    ),
    (
        "Explain disaggregated prefill and decode serving, where prefill and decode run on "
        "separate GPU instances. What are the network bandwidth requirements for transferring "
        "KV cache tensors between prefill and decode nodes for a single 2048-token context "
        "with Llama-3.1-8B, and what workload conditions make disaggregation worth the "
        "infrastructure complexity?"
    ),
    # Advanced serving features
    (
        "Describe speculative decoding in LLM inference. What types of prompts and output "
        "distributions benefit most from speculative decoding, what is the acceptance rate "
        "and how does it determine the actual speedup, and what is the memory overhead of "
        "running a draft model alongside the target model on the same GPU?"
    ),
    (
        "Explain prefix caching in vLLM. Which request patterns benefit most from prefix "
        "caching, how does vLLM track which KV cache blocks correspond to a given prefix "
        "hash, and what happens to cached blocks when memory pressure forces eviction — "
        "can evicted prefix blocks be recovered, or must the prefix be recomputed?"
    ),
    (
        "Describe RadixAttention in SGLang and how the radix tree data structure enables "
        "KV cache sharing across concurrent requests with common prompt prefixes. How does "
        "SGLang's approach differ from vLLM's prefix caching implementation, and what "
        "workloads expose the largest performance gap between the two approaches?"
    ),
    (
        "Explain how CUDA graph capture and replay work in vLLM. Why does vLLM use CUDA "
        "graphs for the decode phase but typically not for the prefill phase, and what "
        "constraint on batch size and sequence length makes graph capture feasible for "
        "decode but not for variable-length prefill?"
    ),
    # Benchmarking and production operations
    (
        "You are benchmarking AWQ and GPTQ quantisation of the same model at 4-bit precision "
        "with group size 128. What metrics would you collect to make a rigorous comparison: "
        "perplexity on a held-out set, TTFT, tokens per second at various batch sizes, "
        "and peak VRAM. Describe what a meaningful result looks like and what would indicate "
        "the calibration data was poorly chosen for the target domain."
    ),
    (
        "Describe the engineering differences between serving Llama-3.1-8B in FP16 versus "
        "4-bit AWQ in terms of memory bandwidth requirements during the decode phase, VRAM "
        "footprint, and expected tokens-per-second at batch size 1 and batch size 32 on a "
        "single A100 80 GB. Under what request volume would you choose AWQ over FP16 in "
        "a cost-sensitive production deployment?"
    ),
    (
        "Explain how AWQ's GEMM and GEMV kernel variants differ and at what batch size the "
        "transition from GEMV-optimal to GEMM-optimal operation typically occurs. How does "
        "the choice of AWQ kernel version at quantisation time affect achievable throughput "
        "when the production workload turns out to be heavier than anticipated?"
    ),
    (
        "You are running a production LLM serving system with a p95 latency SLO of two "
        "seconds for 200-token outputs. Describe the monitoring strategy — metrics, alert "
        "thresholds, and leading indicators — you would use to detect approaching SLO breach "
        "before requests actually fail, including what GPU and serving-layer signals to watch."
    ),
    (
        "Explain how load balancing works across multiple GPU replicas serving the same model. "
        "Beyond simple round-robin, what health signals should a load balancer observe — "
        "KV cache utilisation, queue depth, in-flight sequence count — and why does routing "
        "requests by estimated generation length improve overall cluster throughput compared "
        "to length-agnostic routing?"
    ),
    (
        "Describe the key latency components in a single LLM serving request: network "
        "round-trip, tokenisation, prefill compute, per-token decode latency, and "
        "detokenisation. For a 512-token prompt generating 256 output tokens on a well-"
        "provisioned A100, rank these components by typical contribution to total request "
        "latency and explain which one most benefits from quantisation."
    ),
    (
        "Explain GPTQ quantisation with desc_act=False versus desc_act=True. What is "
        "activation reordering, why does it improve quantisation accuracy by grouping "
        "weights with similar activation magnitudes, and why would you disable it for a "
        "production inference deployment despite the accuracy cost?"
    ),
]


def load_calibration_data(n_wikitext: int = 128) -> list[str]:
    """Load wikitext-2 samples and append the 32 inference-domain prompts."""
    from datasets import load_dataset

    logger.info("loading_wikitext", split="train", n_samples=n_wikitext)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")

    wikitext: list[str] = [
        row["text"]
        for row in ds
        if len(row["text"].strip()) > 100
    ][:n_wikitext]

    combined = wikitext + CALIBRATION_PROMPTS
    logger.info(
        "calibration_data_ready",
        wikitext_samples=len(wikitext),
        inference_prompts=len(CALIBRATION_PROMPTS),
        total=len(combined),
    )
    return combined
