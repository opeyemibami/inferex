"""Fixed benchmark prompt set for the Phase 2 quantisation benchmark suite."""

from __future__ import annotations

PROMPTS: list[dict] = [

    # --- SHORT: 15 prompts — expecting 1-2 sentence answers ---

    {"text": "What is time to first token in LLM serving?", "category": "short"},
    {"text": "What does the vLLM --enforce-eager flag do?", "category": "short"},
    {"text": "What is continuous batching?", "category": "short"},
    {"text": "What is the KV cache in transformer inference?", "category": "short"},
    {"text": "What is AWQ quantisation?", "category": "short"},
    {"text": "What is GPTQ quantisation?", "category": "short"},
    {"text": "What is FP8 quantisation?", "category": "short"},
    {"text": "What is Flash Attention?", "category": "short"},
    {"text": "What is speculative decoding in LLM inference?", "category": "short"},
    {"text": "What is tensor parallelism in the context of LLM serving?", "category": "short"},
    {"text": "What is PagedAttention?", "category": "short"},
    {"text": "What is prefix caching in vLLM?", "category": "short"},
    {"text": "What is the decode phase in autoregressive language model inference?", "category": "short"},
    {"text": "What is a CUDA graph in GPU computing?", "category": "short"},
    {"text": "What is group size in weight quantisation?", "category": "short"},

    # --- MEDIUM: 20 prompts — expecting paragraph answers ---

    {
        "text": (
            "Explain how continuous batching improves GPU utilisation compared to static "
            "batching in an LLM serving system."
        ),
        "category": "medium",
    },
    {
        "text": (
            "Describe the main trade-offs between AWQ and GPTQ at 4-bit precision when "
            "choosing a quantisation method for production serving."
        ),
        "category": "medium",
    },
    {
        "text": (
            "Explain why the decode phase of LLM inference is memory-bandwidth-bound and "
            "what this means for GPU utilisation at low batch sizes."
        ),
        "category": "medium",
    },
    {
        "text": (
            "Describe how PagedAttention solves the memory fragmentation problem in "
            "KV cache management."
        ),
        "category": "medium",
    },
    {
        "text": (
            "Explain what calibration data does in post-training quantisation and why "
            "the choice of calibration dataset matters for downstream quality."
        ),
        "category": "medium",
    },
    {
        "text": (
            "Describe how speculative decoding achieves a throughput improvement and "
            "what conditions must hold for it to be beneficial."
        ),
        "category": "medium",
    },
    {
        "text": (
            "Explain the relationship between batch size and hardware efficiency in the "
            "decode phase of LLM inference."
        ),
        "category": "medium",
    },
    {
        "text": (
            "Describe how tensor parallelism splits attention computation across multiple "
            "GPUs and what communication patterns are required."
        ),
        "category": "medium",
    },
    {
        "text": (
            "Explain how KV cache eviction works in vLLM and what happens to a preempted "
            "sequence that is mid-generation."
        ),
        "category": "medium",
    },
    {
        "text": (
            "Describe the VRAM footprint difference between serving Llama-3.1-8B-Instruct "
            "in FP16 versus 4-bit AWQ on a single GPU."
        ),
        "category": "medium",
    },
    {
        "text": (
            "Explain why prefix caching is particularly valuable for chat applications and "
            "RAG systems with long shared system prompts."
        ),
        "category": "medium",
    },
    {
        "text": (
            "Describe how GPTQ uses an approximation of the inverse Hessian to minimise "
            "quantisation error during weight compression."
        ),
        "category": "medium",
    },
    {
        "text": (
            "Explain how chunked prefill allows vLLM to avoid stalling decode throughput "
            "when a long-context request arrives alongside running decode sequences."
        ),
        "category": "medium",
    },
    {
        "text": (
            "Explain the difference between symmetric and asymmetric quantisation and "
            "which is standard for LLM weight quantisation methods like AWQ."
        ),
        "category": "medium",
    },
    {
        "text": (
            "Describe how Flash Attention avoids materialising the full N-by-N attention "
            "matrix and why this reduces peak GPU memory during prefill."
        ),
        "category": "medium",
    },
    {
        "text": (
            "Explain what happens during vLLM startup when CUDA graph compilation is "
            "enabled, and why it causes a long startup delay."
        ),
        "category": "medium",
    },
    {
        "text": (
            "Describe the key metrics you would monitor in a production LLM serving system "
            "to detect throughput degradation before it breaches an SLO."
        ),
        "category": "medium",
    },
    {
        "text": (
            "Explain how grouped-query attention reduces KV cache memory compared to "
            "standard multi-head attention, and what accuracy trade-off this introduces."
        ),
        "category": "medium",
    },
    {
        "text": (
            "Describe the difference between the prefill and decode phases in terms of "
            "compute intensity and what this implies for scheduling strategy."
        ),
        "category": "medium",
    },
    {
        "text": (
            "Explain how RadixAttention in SGLang enables KV cache sharing across "
            "concurrent requests that share common prompt prefixes."
        ),
        "category": "medium",
    },

    # --- LONG: 15 prompts — expecting detailed technical explanations ---

    {
        "text": (
            "Provide a detailed technical explanation of the AWQ (Activation-aware Weight "
            "Quantisation) algorithm: how salient weights are identified using activation "
            "magnitude, the role of per-channel scaling factors, how calibration data is "
            "used during the search for optimal scales, and why AWQ typically outperforms "
            "naive round-to-nearest quantisation at the same bit width."
        ),
        "category": "long",
    },
    {
        "text": (
            "Give a comprehensive explanation of how to design a benchmark suite for "
            "comparing FP16, AWQ, GPTQ, and FP8 quantisation methods. Cover the metrics "
            "to collect, the prompt categories to test across, the concurrency levels "
            "that expose different performance characteristics, how to isolate quantisation "
            "effects from scheduling effects, and how to interpret results when making a "
            "production deployment decision."
        ),
        "category": "long",
    },
    {
        "text": (
            "Explain in detail how vLLM's PagedAttention manages KV cache memory: the "
            "page table structure, how physical and logical blocks are mapped, how pages "
            "are allocated and freed during request processing, why external memory "
            "fragmentation is eliminated compared to contiguous KV cache allocation, and "
            "what the fixed overhead per page implies for optimal page size selection."
        ),
        "category": "long",
    },
    {
        "text": (
            "Provide a detailed comparison of vLLM and SGLang as LLM inference frameworks, "
            "covering their approaches to KV cache management, scheduling algorithms, "
            "prefix caching implementations, batching strategies, and the workload types "
            "where each framework has a measurable performance advantage."
        ),
        "category": "long",
    },
    {
        "text": (
            "Explain in detail the GPTQ quantisation algorithm: how the inverse Hessian "
            "is approximated from calibration data using the OBQ framework, why quantising "
            "weights column-by-column with error propagation to remaining columns produces "
            "lower quantisation error than independent per-layer quantisation, and how the "
            "desc_act parameter trades inference latency for accuracy."
        ),
        "category": "long",
    },
    {
        "text": (
            "Give a detailed explanation of the GPU memory hierarchy and how it affects "
            "LLM inference performance: the roles of HBM, L2 cache, and SRAM in a modern "
            "GPU, how HBM bandwidth limits decode throughput at low batch sizes, why "
            "increasing batch size improves arithmetic intensity, what the roofline model "
            "reveals about when an LLM workload is compute-bound versus memory-bound, and "
            "how quantisation shifts the roofline crossover point."
        ),
        "category": "long",
    },
    {
        "text": (
            "Explain in detail how continuous batching is implemented in a production LLM "
            "serving system: how the scheduler selects which sequences to include in each "
            "iteration, how it handles sequences that finish mid-batch, how new requests "
            "are inserted into a running batch without padding overhead, and what the "
            "scheduling policy implications are for tail latency across requests of "
            "different lengths."
        ),
        "category": "long",
    },
    {
        "text": (
            "Provide a comprehensive technical explanation of multi-GPU inference strategies "
            "for large language models: tensor parallelism, pipeline parallelism, and "
            "sequence parallelism. For each strategy, explain how the model is partitioned "
            "across GPUs, what collective communication operations are required at each "
            "step, how communication overhead scales with GPU count, and the workload "
            "characteristics where each strategy is preferable."
        ),
        "category": "long",
    },
    {
        "text": (
            "Explain in detail how CUDA graphs work in the context of vLLM's decode phase: "
            "what a CUDA graph captures in terms of kernel launch sequences, why graph "
            "replay reduces CPU-GPU command overhead compared to eager execution, why vLLM "
            "must capture separate graphs for each supported batch size at startup, and the "
            "exact throughput versus startup time trade-off that --enforce-eager makes."
        ),
        "category": "long",
    },
    {
        "text": (
            "Give a detailed explanation of disaggregated prefill and decode serving: why "
            "separating prefill and decode onto different GPU instances can improve "
            "cluster-level throughput utilisation, the KV cache transfer mechanism between "
            "prefill and decode nodes, the network bandwidth requirements for a 2048-token "
            "context with Llama-3.1-8B, and the workload characteristics that make "
            "disaggregation worth the infrastructure and operational complexity."
        ),
        "category": "long",
    },
    {
        "text": (
            "Explain in detail the speculative decoding algorithm: how a draft model "
            "generates k candidate tokens, how the target model verifies all k tokens in "
            "a single forward pass, the token acceptance criterion and what happens to "
            "rejected tokens, how to select a draft model, and the mathematical relationship "
            "between acceptance rate, draft length k, and the realised throughput multiplier."
        ),
        "category": "long",
    },
    {
        "text": (
            "Provide a detailed explanation of how to design SLOs for a production LLM "
            "serving system: how to choose TTFT and inter-token latency targets for "
            "different user-facing applications, how to provision GPU capacity to meet "
            "p95 and p99 SLOs at a given request arrival rate, what observability "
            "infrastructure is required to detect SLO degradation early, and how "
            "quantisation choices affect the hardware budget needed to meet fixed SLOs."
        ),
        "category": "long",
    },
    {
        "text": (
            "Explain in detail the relationship between quantisation bit width, group size, "
            "and calibration data quality in determining the accuracy of a quantised model. "
            "Describe how perplexity on a held-out set relates to downstream task performance, "
            "when perplexity is and is not a reliable proxy for production quality, and how "
            "to detect that a calibration dataset was poorly matched to the production domain."
        ),
        "category": "long",
    },
    {
        "text": (
            "Give a detailed technical explanation of prefix caching in vLLM and "
            "RadixAttention in SGLang: for each system, explain the data structure used "
            "to track and retrieve cached KV blocks, how cache invalidation works when a "
            "prefix is modified, the eviction policy when cache capacity is exhausted, and "
            "the application types — RAG, multi-turn chat, code completion — that benefit "
            "most from each implementation."
        ),
        "category": "long",
    },
    {
        "text": (
            "Explain in detail how to choose between FP16, AWQ, GPTQ, and FP8 quantisation "
            "for a production deployment of Llama-3.1-8B-Instruct. Consider VRAM constraints "
            "and how each format changes peak memory, expected throughput at batch sizes 1 "
            "and 32, acceptable accuracy degradation for the target task, hardware generation "
            "requirements for FP8, calibration data availability and quality, and the "
            "operational cost of maintaining quantised model artefacts over time."
        ),
        "category": "long",
    },
]
