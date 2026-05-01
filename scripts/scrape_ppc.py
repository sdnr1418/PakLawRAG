"""
Scrape the Pakistan Penal Code from pakistani.org and save to
output/ppc_sections.json in the same format as the PDF-parsed version:

    {"section_id": "300", "text": "300. Qatl-i-amd. ..."}

Run from the scripts/ directory:
    python scrape_ppc.py
"""

import json
import re
from pathlib import Path

import requests
import urllib3
from bs4 import BeautifulSoup

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

PPC_URL = "https://www.pakistani.org/pakistan/legislation/1860/actXLVof1860.html"


# ── fetch ──────────────────────────────────────────────────────────────────────

def fetch_text(url: str) -> str:
    resp = requests.get(url, verify=False, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    return soup.get_text(separator="\n")


# ── trim to PPC body ───────────────────────────────────────────────────────────

def trim_to_body(raw: str) -> list[str]:
    """
    Drop everything before "Pakistan Penal Code" and after the last Schedule.
    Return a list of non-empty stripped lines.
    """
    low = raw.lower()
    start = low.find("pakistan penal code")
    end = low.rfind("schedule")
    if start == -1:
        start = 0
    body = raw[start: end if end > start else len(raw)]
    return [line.strip() for line in body.splitlines() if line.strip()]


# ── section heading detection ──────────────────────────────────────────────────

# Matches all observed heading formats on pakistani.org:
#   "375."   "375A."   "337-A."   "120-A"   "462G"
#   "Section 375"   "S. 52-A"
_SEC_PATTERN = re.compile(
    r"""
    ^
    (?:(?:Section|S\.)\s*)?       # optional "Section" / "S." prefix
    (\d+)                         # required: base number
    (?:\s*-\s*|\s*)               # optional hyphen (with/without spaces)
    ([A-Z]*)                      # optional letter suffix
    \.?                           # optional trailing dot
    (?:\s|$)                      # must be followed by whitespace or end-of-line
    """,
    re.VERBOSE | re.IGNORECASE,
)

_CHAPTER_PATTERN = re.compile(r"^Chapter\s+[IVXLC\d]+", re.IGNORECASE)

# Guard: ignore lines that are clearly not section headings even if the
# number pattern fires (e.g. "300 people", standalone years like "1860")
_MIN_SECTION = 1
_MAX_SECTION = 600


def parse_section_id(line: str) -> str | None:
    m = _SEC_PATTERN.match(line)
    if not m:
        return None
    num = int(m.group(1))
    if not (_MIN_SECTION <= num <= _MAX_SECTION):
        return None
    suffix = m.group(2).upper()
    return f"{num}{suffix}"


# ── main parser ────────────────────────────────────────────────────────────────

def parse_sections(lines: list[str]) -> list[dict]:
    sections = []
    current_id: str | None = None
    current_lines: list[str] = []

    def flush():
        if current_id and current_lines:
            text = " ".join(current_lines).strip()
            # normalise runs of whitespace
            text = re.sub(r" {2,}", " ", text)
            sections.append({"section_id": current_id, "text": text})

    for line in lines:
        if _CHAPTER_PATTERN.match(line):
            # keep chapter headings attached to the following section text
            # by appending to current buffer so context is not lost
            current_lines.append(line)
            continue

        sec_id = parse_section_id(line)
        if sec_id:
            flush()
            current_id = sec_id
            current_lines = [line]
        else:
            current_lines.append(line)

    flush()
    return sections


# ── deduplication ──────────────────────────────────────────────────────────────

def _sort_key(sec_id: str):
    m = re.fullmatch(r"(\d+)([A-Z]?)", sec_id)
    if m:
        return (int(m.group(1)), m.group(2))
    return (0, sec_id)


def deduplicate(sections: list[dict]) -> list[dict]:
    """Keep the longest text per section_id, then sort numerically."""
    best: dict[str, dict] = {}
    for sec in sections:
        sid = sec["section_id"]
        if sid not in best or len(sec["text"]) > len(best[sid]["text"]):
            best[sid] = sec
    return sorted(best.values(), key=lambda s: _sort_key(s["section_id"]))


# ── save ───────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = Path(__file__).parent


def save(sections: list[dict], path: str = None):
    if path is None:
        path = str(_SCRIPT_DIR / "../output/ppc_sections.json")
    Path(path).parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(sections)} sections → {path}")


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Fetching PPC from pakistani.org ...")
    raw = fetch_text(PPC_URL)

    lines = trim_to_body(raw)
    print(f"  {len(lines)} lines after trimming")

    sections = parse_sections(lines)
    print(f"  {len(sections)} sections before deduplication")

    sections = deduplicate(sections)
    print(f"  {len(sections)} sections after deduplication")

    # quick sanity check on a few known sections
    targets = {"300", "375", "392", "420", "499"}
    found = {s["section_id"] for s in sections}
    for t in sorted(targets):
        status = "✓" if t in found else "✗"
        print(f"  {status} Section {t}")

    save(sections)
