from fastapi import FastAPI

app = FastAPI(title="inferex")

@app.get("/health")
async def health() -> dict:
    """Returns service health status."""
    return {"status": "ok"}