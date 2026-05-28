# inferex agent rules
# applies to: Claude Code, Codex CLI, any AI coding assistant

## package management
- This project uses uv — never use pip install directly
- Add dependencies with: uv add <package>
- Dev dependencies with: uv add --dev <package>
- Never create or modify requirements.txt — it does not exist
- Lock file is uv.lock — do not edit manually
- Ask before running uv add for any new dependency

## what you can generate freely
- Dockerfile, docker-compose, .dockerignore
- Kubernetes manifests in k8s/
- Prometheus scrape configs, Grafana dashboard JSON
- FastAPI boilerplate, Pydantic schemas
- OpenAI and Anthropic route handlers and schema models
- Logging setup using structlog
- CLI argument parsing
- Test scaffolding and fixtures in tests/

## what you must never generate
- Any logic inside benchmarks/runner.py — the engineer writes this
- Any content inside docs/FINDINGS.md — the engineer writes this
- Interpretation or commentary on benchmark numbers
- Claims about latency, throughput, or model performance

## code standards
- Python 3.11+, type hints on all functions and return values
- Black formatting, isort for imports
- No print() anywhere — use structlog throughout
- All secrets and config via environment variables, never hardcoded
- Every function needs a one-line docstring minimum
- Pydantic v2 for all schemas

## file placement rules
- New engine integrations → engines/
- All benchmark outputs → benchmarks/results/
- Kubernetes and Docker configs → k8s/
- Observability configs → observability/
- Never create files outside the defined structure without asking

## when to ask vs proceed
- Ask before adding any new dependency
- Ask before touching benchmarks/runner.py for any reason
- Ask before touching docs/FINDINGS.md for any reason
- Proceed freely on infrastructure, boilerplate, schemas, tests
- If a task touches more than 3 files, summarise the plan first and wait