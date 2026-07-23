import json
import os
import re
import time
import uuid

from groq import Groq

from tools import TOOL_SCHEMAS, TOOL_IMPL

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
MODEL = "llama-3.3-70b-versatile"

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are a meticulous data analyst agent.
You will receive a data-analysis question, possibly with inline data or a
reference to a public dataset (e.g. MOSPI). The question specifies the EXACT
JSON shape the final answer must be in, including a "log_url" key.

Rules:
- Use the run_python tool to fetch/download data, compute, and verify numbers.
  Do not guess numeric answers — compute them.
- Use fetch_url to preview a text/HTML/CSV/JSON page before writing parsing code.
- Use fetch_binary_and_extract for any PDF or Excel (.xls/.xlsx) URL. Never use
  pandas.read_csv on a PDF — it will fail. MOSPI data is very often published as
  PDF or Excel, so check the file extension before choosing a tool.
- If a tool call fails, do not give up and guess. Try a different URL, a
  different parsing approach, or fetch_url to find the correct dataset page.
  Only give a final answer once you have real retrieved data backing it.
- Iterate until you are confident in the answer.
- When you are done, reply with ONLY the final JSON object, exactly matching
  the shape requested in the question. Put the literal string "PLACEHOLDER"
  as the value of log_url — it will be replaced automatically. No markdown,
  no explanation, no code fences — just the raw JSON object.
"""

NUDGE_MESSAGE = (
    "You have not successfully retrieved any real data yet — all tool calls so "
    "far have failed. Do not answer with a guess or a value from memory. Try a "
    "different approach: check the file extension and use fetch_binary_and_extract "
    "for PDF/Excel URLs, use fetch_url to inspect the page structure first, or "
    "search for the correct dataset URL. Keep trying before giving a final answer."
)


def _log(logf, obj):
    obj["ts"] = time.time()
    logf.write(json.dumps(obj, default=str) + "\n")
    logf.flush()


def _extract_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return None


def run_agent(question_text: str, history: list, log_path: str, log_url: str):
    """history: list of {'role': 'user'|'assistant', 'content': str} prior turns."""
    run_id = str(uuid.uuid4())
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history:
        messages.append(h)
    messages.append({"role": "user", "content": question_text})

    with open(log_path, "a") as logf:
        _log(logf, {"run_id": run_id, "event": "start", "question": question_text})

        final_json = None
        any_tool_succeeded = False
        max_steps = 12

        for step in range(max_steps):
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0,
                max_tokens=2048,
            )
            msg = resp.choices[0].message
            _log(logf, {
                "run_id": run_id, "event": "llm_response", "step": step,
                "content": msg.content,
                "tool_calls": [tc.function.name for tc in msg.tool_calls] if msg.tool_calls else [],
            })

            if msg.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in msg.tool_calls
                    ],
                })
                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                    except Exception:
                        args = {}
                    _log(logf, {"run_id": run_id, "event": "tool_call", "step": step, "tool": fn_name, "args": args})
                    try:
                        result = TOOL_IMPL[fn_name](**args)
                    except Exception as e:
                        result = f"ERROR running tool: {e}"

                    if not str(result).strip().startswith("ERROR"):
                        any_tool_succeeded = True

                    _log(logf, {"run_id": run_id, "event": "tool_result", "step": step, "tool": fn_name, "result": str(result)[:3000]})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": fn_name,
                        "content": str(result)[:6000],
                    })
                continue

            # no tool calls -> model thinks it's done
            parsed = _extract_json(msg.content or "")

            if not any_tool_succeeded and step < max_steps - 1:
                # Refuse to accept a guessed answer — push back and force retry.
                _log(logf, {"run_id": run_id, "event": "rejected_guess", "step": step, "content": msg.content})
                messages.append({"role": "assistant", "content": msg.content or ""})
                messages.append({"role": "user", "content": NUDGE_MESSAGE})
                continue

            final_json = parsed if parsed is not None else {"answer": (msg.content or "").strip()}
            break

        if final_json is None:
            final_json = {"answer": None}

        final_json["log_url"] = log_url
        _log(logf, {
            "run_id": run_id, "event": "final",
            "any_tool_succeeded": any_tool_succeeded,
            "answer_json": final_json,
        })

    return final_json