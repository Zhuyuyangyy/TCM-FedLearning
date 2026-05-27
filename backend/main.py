"""面向多中心中医临床数据的隐私保护联邦学习"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.train import router as train_router

app = FastAPI(title="TCM-FedLearning", version="0.2.0", description="面向多中心中医临床数据的隐私保护联邦学习")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ── Register routers ──
app.include_router(train_router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "TCM-FedLearning", "version": "0.2.0"}


@app.get("/api/models")
async def list_models():
    """List available federated learning algorithms."""
    return {
        "algorithms": [
            {"name": "fedavg", "description": "Federated Averaging (McMahan et al.)"},
            {"name": "fedprox", "description": "FedProx — proximal term for non-IID (Li et al., 2020)"},
            {"name": "scaffold", "description": "SCAFFOLD — variance reduction via control variates (Karimireddy et al., 2020)"},
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8027)
