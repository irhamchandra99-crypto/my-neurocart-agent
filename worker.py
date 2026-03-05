"""
NeuroCart Agent Worker
- Poll JobEscrow contract tiap beberapa detik
- Auto acceptJob() untuk job yang ditujukan ke agent ini
- Proses dengan OpenAI
- submitResult() balik ke kontrak
"""

import asyncio
import os
import logging
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
import openai

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config dari env ──────────────────────────────────────────
RPC_URL          = os.getenv("RPC_URL", "https://sepolia.base.org")
PRIVATE_KEY      = os.getenv("AGENT_PRIVATE_KEY", "")       # wallet yang register agent
ESCROW_ADDRESS   = os.getenv("ESCROW_ADDRESS", "0xff8d57c82ddb6987decce533dfe1799f880eca75")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
POLL_INTERVAL    = int(os.getenv("POLL_INTERVAL", "15"))     # detik

# ── Setup ────────────────────────────────────────────────────
w3 = Web3(Web3.HTTPProvider(RPC_URL))
w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
oai = openai.OpenAI(api_key=OPENAI_API_KEY)

account = w3.eth.account.from_key(PRIVATE_KEY)
AGENT_ADDRESS = account.address
log.info(f"Agent wallet: {AGENT_ADDRESS}")

# ── Minimal ABI — hanya function yang dibutuhkan ─────────────
ESCROW_ABI = [
    {
        "name": "jobCount", "type": "function",
        "inputs": [], "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view"
    },
    {
        "name": "jobs", "type": "function",
        "inputs": [{"name": "", "type": "uint256"}],
        "outputs": [
            {"name": "jobId",              "type": "uint256"},
            {"name": "clientAgent",        "type": "address"},
            {"name": "providerAgent",      "type": "address"},
            {"name": "registryAgentId",    "type": "uint256"},
            {"name": "payment",            "type": "uint256"},
            {"name": "paymentToken",       "type": "uint8"},
            {"name": "resultData",         "type": "string"},
            {"name": "jobDescription",     "type": "string"},
            {"name": "jobType",            "type": "string"},
            {"name": "status",             "type": "uint8"},
            {"name": "createdAt",          "type": "uint256"},
            {"name": "deadline",           "type": "uint256"},
            {"name": "verificationRequestId", "type": "bytes32"},
            {"name": "qualityScore",       "type": "uint8"},
        ],
        "stateMutability": "view"
    },
    {
        "name": "acceptJob", "type": "function",
        "inputs": [{"name": "jobId", "type": "uint256"}],
        "outputs": [], "stateMutability": "nonpayable"
    },
    {
        "name": "submitResult", "type": "function",
        "inputs": [
            {"name": "jobId",   "type": "uint256"},
            {"name": "result",  "type": "string"},
        ],
        "outputs": [], "stateMutability": "nonpayable"
    },
]

escrow = w3.eth.contract(
    address=Web3.to_checksum_address(ESCROW_ADDRESS),
    abi=ESCROW_ABI
)

# ── Job status enum ──────────────────────────────────────────
STATUS_CREATED  = 0
STATUS_ACCEPTED = 1

# ── Skill → system prompt ────────────────────────────────────
SKILL_PROMPTS = {
    "summarization":    "You are an expert summarizer. Summarize the given text clearly and concisely. Respond in the same language as the input.",
    "translation":      "You are a professional translator. Translate the given text accurately.",
    "nlp":              "You are an NLP expert. Analyze and process the given text as instructed.",
    "ocr":              "You are an OCR expert. Extract and clean text from the described content.",
    "transcription":    "You are a transcription expert. Transcribe spoken content accurately.",
    "code-generation":  "You are an expert programmer. Generate clean, well-commented code as requested.",
    "data-analysis":    "You are a data analyst. Analyze the given data and provide clear insights.",
    "classification":   "You are a classification expert. Classify the given input and explain your reasoning.",
    "sentiment-analysis": "You are a sentiment analysis expert. Determine the sentiment and explain why.",
    "general":          "You are a helpful AI assistant. Complete the given task accurately.",
}

# ── Sudah diproses, hindari double processing ────────────────
processed_jobs: set[int] = set()


# ── Send transaction helper ──────────────────────────────────
def send_tx(fn):
    """Build, sign, send transaction. Return receipt."""
    nonce    = w3.eth.get_transaction_count(AGENT_ADDRESS)
    gas_price = w3.eth.gas_price

    tx = fn.build_transaction({
        "from":     AGENT_ADDRESS,
        "nonce":    nonce,
        "gasPrice": gas_price,
        "gas":      500_000,
    })
    signed = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    log.info(f"  TX sent: {tx_hash.hex()}")
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    return receipt


# ── Call OpenAI ──────────────────────────────────────────────
def run_ai(description: str, job_type: str) -> str:
    skill = job_type.lower().replace(" ", "-")
    system = SKILL_PROMPTS.get(skill, SKILL_PROMPTS["general"])

    response = oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": description},
        ],
        max_tokens=1500,
        temperature=0.3,
    )
    return response.choices[0].message.content or ""


# ── Process satu job ─────────────────────────────────────────
async def process_job(job_id: int, job: dict):
    log.info(f"[Job #{job_id}] Processing — type: {job['jobType']}")

    # 1. acceptJob()
    log.info(f"[Job #{job_id}] Calling acceptJob()...")
    try:
        receipt = send_tx(escrow.functions.acceptJob(job_id))
        if receipt.status != 1:
            log.error(f"[Job #{job_id}] acceptJob() FAILED")
            return
        log.info(f"[Job #{job_id}] Accepted ✓")
    except Exception as e:
        log.error(f"[Job #{job_id}] acceptJob() error: {e}")
        return

    # 2. Run AI
    log.info(f"[Job #{job_id}] Running AI...")
    try:
        result = run_ai(job["jobDescription"], job["jobType"])
        log.info(f"[Job #{job_id}] AI done — {len(result)} chars")
    except Exception as e:
        log.error(f"[Job #{job_id}] AI error: {e}")
        return

    # 3. submitResult()
    log.info(f"[Job #{job_id}] Submitting result...")
    try:
        receipt = send_tx(escrow.functions.submitResult(job_id, result))
        if receipt.status != 1:
            log.error(f"[Job #{job_id}] submitResult() FAILED")
            return
        log.info(f"[Job #{job_id}] Result submitted ✓ — now VERIFYING (Chainlink)")
    except Exception as e:
        log.error(f"[Job #{job_id}] submitResult() error: {e}")


# ── Main polling loop ────────────────────────────────────────
async def poll():
    log.info(f"Worker started. Polling every {POLL_INTERVAL}s...")
    log.info(f"Watching address: {AGENT_ADDRESS}")

    while True:
        try:
            job_count = escrow.functions.jobCount().call()

            for job_id in range(job_count):
                if job_id in processed_jobs:
                    continue

                job = escrow.functions.jobs(job_id).call()
                # jobs() returns tuple — map by index from ABI
                job_dict = {
                    "jobId":          job[0],
                    "clientAgent":    job[1],
                    "providerAgent":  job[2],
                    "registryAgentId": job[3],
                    "payment":        job[4],
                    "paymentToken":   job[5],
                    "resultData":     job[6],
                    "jobDescription": job[7],
                    "jobType":        job[8],
                    "status":         job[9],
                    "createdAt":      job[10],
                    "deadline":       job[11],
                }

                # Hanya proses job yang ditujukan ke wallet ini dan statusnya CREATED
                if (
                    job_dict["status"] == STATUS_CREATED
                    and job_dict["providerAgent"].lower() == AGENT_ADDRESS.lower()
                ):
                    processed_jobs.add(job_id)
                    await process_job(job_id, job_dict)

        except Exception as e:
            log.error(f"Polling error: {e}")

        await asyncio.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    if not PRIVATE_KEY:
        raise ValueError("AGENT_PRIVATE_KEY env var tidak diset!")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY env var tidak diset!")
    asyncio.run(poll())
