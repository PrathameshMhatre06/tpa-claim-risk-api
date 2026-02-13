from fastapi import FastAPI
from pydantic import BaseModel

# ------------------
# API setup
# ------------------
app = FastAPI(title="TPA Claim Risk API")

class ClaimRequest(BaseModel):
    claim_amount: float

class ClaimResponse(BaseModel):
    risk_score: float
    risk_level: str

# ------------------
# Prediction endpoint
# ------------------
@app.post("/predict", response_model=ClaimResponse)
def predict_risk(data: ClaimRequest):
    # Scale claim amount (₹100000 = 1.0)
    scaled_amount = data.claim_amount / 100000

    # Simple classical risk formula (no torch)
    score = 1.5 * scaled_amount - 0.5

    # Clamp score between 0 and 1
    score = max(0, min(score, 1))

    risk_level = "HIGH" if score >= 0.7 else "LOW"

    return {
        "risk_score": round(score, 2),
        "risk_level": risk_level
    }

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "tpa-claim-risk-api"
    }
