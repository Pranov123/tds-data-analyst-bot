import io
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


def fetch_binary_and_extract(url: str, filetype: str = "auto") -> str:
    """Download a PDF or Excel file and extract text/tables.
    filetype: 'pdf', 'excel', or 'auto' (guess from URL)."""
    import pdfplumber
    import pandas as pd

    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
    except Exception as e:
        return f"ERROR downloading {url}: {e}"

    ft = filetype
    if ft == "auto":
        if url.lower().endswith((".xls", ".xlsx")):
            ft = "excel"
        else:
            ft = "pdf"

    try:
        if ft == "excel":
            xls = pd.ExcelFile(io.BytesIO(r.content))
            out = []
            for sheet in xls.sheet_names[:3]:
                df = xls.parse(sheet)
                out.append(f"--- Sheet: {sheet} ---\n{df.head(30).to_string()}")
            return "\n\n".join(out)[:12000]
        else:
            out = []
            with pdfplumber.open(io.BytesIO(r.content)) as pdf:
                for i, page in enumerate(pdf.pages[:8]):
                    text = page.extract_text() or ""
                    tables = page.extract_tables()
                    out.append(f"--- Page {i+1} text ---\n{text}")
                    for t in tables:
                        out.append(f"--- Page {i+1} table ---\n{t}")
            return "\n\n".join(out)[:12000]
    except Exception as e:
        return f"ERROR parsing {url} as {ft}: {e}"


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
            "description": "Fetch raw text/HTML/CSV/JSON content from a public URL. Use this to preview a dataset page or small text/CSV/JSON file before writing code. Do NOT use this for PDF or Excel files — use fetch_binary_and_extract instead.",
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
            "name": "fetch_binary_and_extract",
            "description": "Download a PDF or Excel file (e.g. MOSPI statistical tables, .pdf, .xls, .xlsx) and extract text/tables from it. Use this instead of run_python/pd.read_csv whenever the URL points to a PDF or Excel file — pandas.read_csv cannot read PDFs directly and will error.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "filetype": {
                        "type": "string",
                        "enum": ["pdf", "excel", "auto"],
                        "description": "Defaults to 'auto', which guesses from the URL extension.",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute Python code to parse/compute/analyze data already retrieved (e.g. via fetch_url or fetch_binary_and_extract), or to download/parse CSV/JSON data directly. pandas, numpy and requests are pre-installed. Always print() the final result you need. Do not use pd.read_csv on PDF or Excel URLs.",
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

TOOL_IMPL = {
    "fetch_url": fetch_url,
    "fetch_binary_and_extract": fetch_binary_and_extract,
    "run_python": run_python,
}