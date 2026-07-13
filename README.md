# inferex

A benchmark-driven LLM inference platform. Serves Llama-3.1-8B-Instruct through a FastAPI gateway backed by vLLM and SGLang, with Prometheus and Grafana observability and Kubernetes deployment on a single RTX 5090.

---

## What this is

An end-to-end inference engineering project covering API gateway design, quantisation benchmarking, engine comparison, and Kubernetes GPU scheduling. Every component answers one question: does this reflect how a production inference platform is actually built?

---

## Architecture

![Architecture](/assets/inferex.svg)

The gateway sits in front of both engines. Prometheus and Grafana run in Docker Compose and scrape vLLM via `172.17.0.1:30001` when running under k3s.

---

## Benchmark results

All results use Llama-3.1-8B-Instruct on RTX 5090 at concurrency 32.

### Quantisation

| Variant | Tokens/sec | p95 latency | KV cache capacity |
|---------|-----------|-------------|-------------------|
| FP16 | 87.4 | 2295 ms | 108,768 tokens |
| AWQ | 182.2 | 1115 ms | 183,968 tokens |
| GPTQ | 172.1 | 1155 ms | 183,792 tokens |
| FP8 | 140.0 | 1436 ms | 158,368 tokens |

VRAM stays within 482 MB across all variants. vLLM reinvests weight memory savings into a larger KV cache rather than reducing total footprint.

![Quantization Benchmark](assets/quantizationbenchmarks.svg)

### Engine comparison

| Engine | Tokens/sec | p95 latency |
|--------|-----------|-------------|
| vLLM | 88.6 | 2232 ms |
| SGLang¹ | 79.0 | 2649 ms |

¹ SGLang ran with `--disable-overlap-schedule` due to a WSL2 CUDA synchronisation constraint. On native Linux with overlap scheduling enabled, SGLang is expected to outperform vLLM at high concurrency.

---

## Stack

| Concern | Tool |
|---------|------|
| Inference engines | vLLM, SGLang |
| Gateway | FastAPI + httpx |
| Observability | Prometheus + Grafana |
| Orchestration | k3s |
| Quantisation | AutoAWQ, Neural Magic GPTQ |
| Model | meta-llama/Meta-Llama-3.1-8B-Instruct |

---

## Running it

### Docker Compose

```bash
git clone <repo>
cd inferex
cp .env.example .env
# set HUGGING_FACE_HUB_TOKEN and GATEWAY_API_KEYS
docker compose up
```

vLLM takes up to 10 minutes on first run due to CUDA graph compilation. Subsequent restarts take ~2 minutes with the cache mounted.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "X-API-Key: dev-key-change-me" \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Meta-Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"Hello"}]}'
```

### Kubernetes (k3s)

```bash
# Build and import images
docker build -f Dockerfile.vllm -t inferex-vllm:latest .
docker save inferex-vllm:latest | k3s ctr images import -
docker build -t inferex-gateway:latest .
docker save inferex-gateway:latest | k3s ctr images import -

# Deploy
kubectl create secret generic inferex-secrets -n inferex \
  --from-literal=HUGGING_FACE_HUB_TOKEN=hf_... \
  --from-literal=GATEWAY_API_KEYS=your-key
kubectl apply -f k8s/
```

Gateway at `http://localhost:30000`, vLLM at `http://localhost:30001`.

---

## Hardware

Tested on RTX 5090 (32 GB VRAM) under WSL2. Minimum 24 GB VRAM for FP16 8B inference.

The RTX 5090 is Blackwell (SM 12.x) and requires CUDA 12.9+. The pip-installed vLLM binary does not support it — all vLLM invocations use `docker run vllm/vllm-openai:latest` which ships the correct toolkit. On Ampere or Ada Lovelace GPUs the local binary works without Docker.

---

## Article
👉 [Read full article](https://opeyemibami.github.io/notes/article-inferex/) 
