markdown
# TDS Data Analyst Telegram Bot

FastAPI + Groq (llama-3.3-70b-versatile) agent that answers data-analysis
questions sent over Telegram and replies with `{"answer": ..., "log_url": ...}`.

## Local test

cp .env.example .env # fill in real values
pip install -r requirements.txt
uvicorn main:app --reload


## Deploy: see repo root instructions.