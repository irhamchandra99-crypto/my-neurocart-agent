"""
NeuroCart AI Agent — FastAPI wrapper for OpenAI
Deploy ke Railway: https://railway.app
"""

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import openai
import os
import httpx

app = FastAPI(title="NeuroCart AI Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Ambil dari env var ──────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ── Skill → system prompt mapping ──────────────────────────
SKILL_PROMPTS: dict[str, str] = {
    "summarization":  "You are an expert summarizer. Summarize the given text clearly and concisely. Respond in the same language as the input.",
    "translation":    "You are a professional translator. Translate the given text accurately. Detect the source language automatically.",
    "nlp":            "You are an NLP expert. Analyze and process the given text as instructed.",
    "ocr":            "You are an OCR and document processing expert. Extract and clean text from the described content.",
    "transcription":  "You are a transcription expert. Transcribe spoken content accurately.",
    "code-generation":"You are an expert programmer. Generate clean, well-commented code as requested.",
    "data-analysis":  "You are a data analyst. Analyze the given data and provide clear insights.",
    "classification": "You are a classification expert. Classify the given input and explain your reasoning.",
    "sentiment-analysis": "You are a sentiment analysis expert. Determine the sentiment and explain why.",
    "general":        "You are a helpful AI assistant. Complete the given task accurately.",
}

# ── Request / Response schema ───────────────────────────────
class JobRequest(BaseModel):
    jobId: Optional[str]       = None
    description: str           # task description dari user
    jobType: Optional[str]     = "general"
    clientAddress: Optional[str] = None

class JobResponse(BaseModel):
    result: str
    qualityScore: int          # 0–100, harus ≥ 80 agar payment released
    jobId: Optional[str]       = None
    model: str                 = "gpt-4o-mini"
    tokensUsed: Optional[int]  = None


# ── Health check ────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "agent": "NeuroCart AI Agent",
        "status": "online",
        "skills": list(SKILL_PROMPTS.keys()),
    }

@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Main endpoint — dipanggil oleh Chainlink / NeuroCart ────
@app.post("/run", response_model=JobResponse)
async def run_job(job: JobRequest):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

    skill = (job.jobType or "general").lower().replace(" ", "-")
    system_prompt = SKILL_PROMPTS.get(skill, SKILL_PROMPTS["general"])

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",           # ganti ke "gpt-4o" untuk kualitas lebih tinggi
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": job.description},
            ],
            max_tokens=1500,
            temperature=0.3,               # rendah = lebih konsisten = skor lebih stabil
        )

        result_text  = response.choices[0].message.content or ""
        tokens_used  = response.usage.total_tokens if response.usage else None

        # ── Quality scoring ─────────────────────────────────
        # Hitung skor sederhana berdasarkan panjang & kelengkapan output
        # Chainlink Functions akan re-verify ini, tapi kita kasih estimasi dulu
        quality_score = _calculate_quality(result_text, job.description)

        return JobResponse(
            result=result_text,
            qualityScore=quality_score,
            jobId=job.jobId,
            model="gpt-4o-mini",
            tokensUsed=tokens_used,
        )

    except openai.AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid OpenAI API key")
    except openai.RateLimitError:
        raise HTTPException(status_code=429, detail="OpenAI rate limit exceeded")
    except openai.APIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {str(e)}")


# ── Quality score calculator ────────────────────────────────
def _calculate_quality(result: str, description: str) -> int:
    """
    Hitung quality score 0–100.
    Chainlink Functions akan override ini dengan verifikasi on-chain,
    tapi ini dipakai sebagai self-reported score.
    """
    if not result or len(result.strip()) < 10:
        return 0

    score = 60  # base score kalau ada output

    # Panjang output — lebih dari 50 kata = +10
    words = len(result.split())
    if words >= 200: score += 20
    elif words >= 100: score += 15
    elif words >= 50:  score += 10
    elif words >= 20:  score += 5

    # Tidak ada error/refusal message
    refusal_phrases = ["i cannot", "i'm unable", "i can't", "as an ai", "i apologize"]
    if not any(p in result.lower() for p in refusal_phrases):
        score += 10

    # Ada struktur (list, paragraf, heading)
    if "\n" in result or ":" in result:
        score += 5

    # Relevan dengan task (keyword dari description ada di result)
    desc_keywords = [w for w in description.lower().split() if len(w) > 4][:5]
    matches = sum(1 for kw in desc_keywords if kw in result.lower())
    if matches >= 3: score += 5

    return min(score, 100)


# ── Endpoint info untuk register form ──────────────────────
@app.get("/info")
async def agent_info():
    """Metadata untuk diisi ke form register NeuroCart"""
    return {
        "name": "GPT-4o Mini Agent",
        "skills": list(SKILL_PROMPTS.keys()),
        "priceUSDCents": 200,          # $2.00
        "endpoint": "https://YOUR-RAILWAY-URL.railway.app/run",
        "metadataURI": "",             # opsional: IPFS URL
    }
