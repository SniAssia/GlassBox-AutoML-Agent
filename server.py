import json  # <-- AJOUTÉ : Nécessaire pour json.loads()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # <-- AJOUTÉ : Pour WASM
from pydantic import BaseModel
from autofit_tool import run_autofit

app = FastAPI()

# --- CONFIGURATION CORS (INDISPENSABLE POUR WASM) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Permet à ton interface WASM de communiquer
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AutoMLRequest(BaseModel):
    csv_base64: str
    target_column: str
    task_type: str = "classification"
@app.get("/run-automl")
async def validate_automl():
    return {
        "status": "ready",
        "message": "GlassBox-AutoML-Agent is ready to process your CSV.",
        "supported_methods": ["POST"]
    }
@app.post("/run-automl")
async def handle_automl(req: AutoMLRequest):
    try:
        # Appelle ton pipeline automatique
        result = run_autofit(req.csv_base64, req.target_column, req.task_type)
        return json.loads(result)
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- BLOC DE LANCEMENT ---
if __name__ == "__main__":
    import uvicorn
    # Lance le serveur sur le port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)