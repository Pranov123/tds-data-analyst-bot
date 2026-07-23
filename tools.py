import io
import json
import subprocess
import sys
import tempfile
import textwrap

import requests


def fetch_url(url: str, max_chars: int = 15000) -> str:
    """Fetch a URL and return truncated text content."""
    try:
        r = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        text = r.text
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n...[truncated, {len(r.text)} chars total]"
        return text
    except Exception as e:
        return f"ERROR fetching {url}: {e}"


def run_python(code: str, timeout: int = 60) -> str:
    """Run python code in a subprocess (has pandas/numpy/requests available).
    The code should print() whatever it wants returned."""
    code = textwrap.dedent(code)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = result.stdout[-8000:]
        err = result.stderr[-4000:]
        combined = f"STDOUT:\n{out}"
        if err:
            combined += f"\nSTDERR:\n{err}"
        return combined
    except subprocess.TimeoutExpired:
        return "ERROR: execution timed out"
    except Exception as e:
        return f"ERROR: {e}"


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch raw text/HTML/CSV/JSON content from a public URL. Use this to preview a dataset page or small file before writing code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch"}
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute Python code to download/parse/analyze data. pandas, numpy and requests are pre-installed. Always print() the final result you need.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Full python script to run"}
                },
                "required": ["code"],
            },
        },
    },
]

TOOL_IMPL = {"fetch_url": fetch_url, "run_python": run_python}