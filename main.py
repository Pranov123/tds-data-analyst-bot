import json
import os
import threading

import requests
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, JSONResponse

from agent import run_agent

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
WEBHOOK_SECRET = os.environ["WEBHOOK_SECRET"]
BASE_URL = os.environ["BASE_URL"].rstrip("/")

TG_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to specific origins in production
    allow_credentials=False,      # must be False when allow_origins is "*"
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],          # covers Authorization, A2A-Version, Content-Type, etc.
)

# in-memory per-chat short history: {chat_id: [ {role, content}, ... ]}
CHAT_HISTORY = {}
MAX_HISTORY_TURNS = 6


def send_message(chat_id: int, text: str):
    requests.post(f"{TG_API}/sendMessage", json={"chat_id": chat_id, "text": text})


def handle_message(chat_id: int, text: str):
    history = CHAT_HISTORY.get(chat_id, [])
    log_path = os.path.join(LOG_DIR, f"{chat_id}.jsonl")
    log_url = f"{BASE_URL}/logs/{chat_id}.jsonl"

    try:
        result = run_agent(text, history, log_path, log_url)
    except Exception as e:
        result = {"answer": None, "log_url": log_url, "error": str(e)}

    # update rolling history
    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": json.dumps(result)})
    CHAT_HISTORY[chat_id] = history[-MAX_HISTORY_TURNS * 2:]

    send_message(chat_id, json.dumps(result))


@app.post("/webhook/{secret}")
async def webhook(secret: str, request: Request):
    if secret != WEBHOOK_SECRET:
        return JSONResponse({"ok": False}, status_code=403)
    update = await request.json()
    msg = update.get("message") or update.get("edited_message")
    if not msg or "text" not in msg:
        return {"ok": True}
    chat_id = msg["chat"]["id"]
    text = msg["text"]
    # run in background thread so Telegram doesn't time out the webhook
    threading.Thread(target=handle_message, args=(chat_id, text), daemon=True).start()
    return {"ok": True}


@app.get("/logs/{chat_id}.jsonl")
def get_log(chat_id: str):
    path = os.path.join(LOG_DIR, f"{chat_id}.jsonl")
    if not os.path.exists(path):
        return PlainTextResponse("", status_code=200)
    with open(path) as f:
        content = f.read()
    return PlainTextResponse(content, media_type="application/jsonl")


@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/set_webhook")
def set_webhook():
    url = f"{BASE_URL}/webhook/{WEBHOOK_SECRET}"
    r = requests.get(f"{TG_API}/setWebhook", params={"url": url})
    return r.json()