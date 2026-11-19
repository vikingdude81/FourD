#!/usr/bin/env python3
"""arXiv literature sweep for VFD (Vibrational Field Dynamics) topic:
spectral mode selection / graph-Laplacian filtering / coupled oscillators on highly symmetric graphs.
"""
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

# ---------------------------------------------------------------- config
TOPIC = "vfd"
# Improved queries - broader terms to catch relevant work
QUERIES = [
    # Core: Laplacian spectra of highly symmetric graphs
    'all:"graph Laplacian" AND (all:"highly symmetric" OR all:"regular polytope" OR all:"600-cell")',
    'abs:"Laplacian spectrum" AND (abs:"graph" OR abs:"adjacency")',
    # Spectral filtering / mode selection
    'abs:"spectral filtering" AND abs:"graph"',
    'abs:"mode selection" AND (abs:"graph" OR abs:"network")',
    'all:"spectral" AND all:"dimensionality reduction" AND (all:"graph" OR all:"network")',
    # Coupled oscillators / resonance on graphs
    'abs:"coupled oscillators" AND (abs:"graph" OR abs:"network")',
    'abs:"normal modes" AND (abs:"graph Laplacian" OR abs:"vibrational modes")',
    'abs:"resonance" AND abs:"graph spectrum"',
    # Synchronizability / eigenratio
    'abs:synchronizability AND (all:"eigenratio" OR all:"spectral")',
    'abs:"eigenratio" AND (abs:"Laplacian" OR abs:"spectrum")',
    # Spectral entropy / complexity
    'all:"spectral entropy" AND all:"complexity"',
    'all:"spectral complexity" AND all:"network"',
    # Scale-free / complex networks
    'abs:"normal modes" AND (abs:"scale-free" OR all:"Barabasi-Albert")',
    'all:"graph signal processing" AND all:"spectral"',
]
MAX_RESULTS = 100
SELECTED = []  # Will be filled from master.json
MIN_PDF_BYTES = 50_000  # below this it is an error page, not a paper
API = "https://export.arxiv.org/api/query"  # https is mandatory
ATOM = "{http://www.w3.org/2005/Atom}"
ARX = "{http://arxiv.org/schemas/atom}"

# Workspace - write to lit/ dir directly
ROOT = os.path.dirname(os.path.abspath(__file__))
PDFS = os.path.join(ROOT, "pdfs")
TXT = os.path.join(ROOT, "txt")
MASTER = os.path.join(ROOT, "master.json")

# ---------------------------------------------------------------- helpers
def base_id(raw):
    """2609.22300v2 -> 2609.22300  (dedupe key; version suffix must be stripped)."""
    return re.sub(r"v\d+$", "", raw.rsplit("/", 1)[-1].strip())


def fetch(url, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(3 * (i + 1))
    return b""


def query(q, max_results=MAX_RESULTS):
    params = urllib.parse.urlencode(
        {
            "search_query": q,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    raw = fetch(f"{API}?{params}")
    # A throttled feed and a zero-hit feed both look "empty" -- check bytes first.
    if len(raw) < 1000:
        print(f"  !! suspiciously small feed ({len(raw)} bytes) for: {q} -- likely throttled, retry")
        return []
    root = ET.fromstring(raw)
    out = []
    for e in root.findall(f"{ATOM}entry"):
        idel = e.find(f"{ATOM}id")
        if idel is None:
            continue
        bid = base_id(idel.text or "")
        authors = [a.findtext(f"{ATOM}name", "").strip() for a in e.findall(f"{ATOM}author")]
        cats = [c.get("term", "") for c in e.findall(f"{ATOM}category")]
        prim = e.find(f"{ARX}primary_category")
        out.append(
            {
                "id": bid,
                "title": re.sub(r"\s+", " ", e.findtext(f"{ATOM}title", "") or "").strip(),
                "authors": [a for a in authors if a],
                "category": prim.get("term") if prim is not None else (cats[0] if cats else ""),
                "published": (e.findtext(f"{ATOM}published", "") or "")[:10],
                "summary": re.sub(r"\s+", " ", e.findtext(f"{ATOM}summary", "") or "").strip(),
            }
        )
    return out


def slice_around(t, pats, n):
    """Anchor-slice a section. Takes the LAST match -- the first match in a long
    paper is usually the table of contents entry, not the section itself."""
    for p in pats:
        hits = list(re.finditer(p, t, re.I))
        for m in reversed(hits):
            tail = t[m.end() : m.end() + 200]
            # reject TOC entries: dot leaders and bare page numbers
            if re.search(r"\.{5,}", tail) or re.match(r"[\s\d.]{0,60}$", tail):
                continue
            return t[m.start() : m.start() + n]
    return ""


def extract_text(pdf):
    import fitz  # PyMuPDF

    with fitz.open(pdf) as doc:
        return "\n".join(page.get_text() for page in doc)


# ---------------------------------------------------------------- stages
def stage_index():
    os.makedirs(ROOT, exist_ok=True)
    os.makedirs(PDFS, exist_ok=True)
    os.makedirs(TXT, exist_ok=True)
    
    master = {}
    for q in QUERIES:
        rows = query(q)
        new = 0
        for r in rows:
            if r["id"] not in master:
                master[r["id"]] = r
                new += 1
        print(f"  {q!r}: {len(rows)} hits, +{new} new")
        time.sleep(3)  # arXiv API courtesy -- do not remove
    path = MASTER
    json.dump(list(master.values()), open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"INDEX: {len(master)} unique papers -> {path}")
    print(f"SELECTED: {len(SELECTED)} of {len(master)}")
    missing = [s for s in SELECTED if s not in master]
    if missing:
        print(f"  !! selected ids not in index: {missing}")
    return master


def stage_fetch(selected):
    os.makedirs(PDFS, exist_ok=True)
    for bid, title, why in selected:
        out = os.path.join(PDFS, f"{bid}.pdf")
        try:
            raw = fetch(f"https://arxiv.org/pdf/{bid}")
        except Exception as e:  # noqa: BLE001
            print(f"  {bid} FETCH-ERROR {e}")
            continue
        open(out, "wb").write(raw)
        flag = "" if len(raw) >= MIN_PDF_BYTES else "  <-- TOO SMALL, not a real PDF"
        print(f"  {bid} bytes:{len(raw)}{flag}")
        time.sleep(2)


def stage_digest(master):
    os.makedirs(TXT, exist_ok=True)
    for bid, item in master.items():
        pdf = os.path.join(PDFS, f"{bid}.pdf")
        if not os.path.exists(pdf):
            print(f"  !! {bid} missing pdf, skipped")
            continue
        try:
            import fitz  # PyMuPDF
            with fitz.open(pdf) as doc:
                t = "\n".join(page.get_text() for page in doc)
        except ImportError:
            print(f"  !! PyMuPDF not available for {bid}, skipping extraction")
            continue
        
        open(os.path.join(TXT, f"{bid}.txt"), "w", encoding="utf-8").write(t)
        
        intro = slice_around(
            t, [r"\n\s*1[\. ]?\s*Introduction", r"\nIntroduction\s*\n"], 2200
        ) or t[1200:3400]
        concl = slice_around(
            t,
            [r"\n\s*(?:\d+[\. ]+)?\s*(?:Conclusion|Conclusions|Discussion|Final Remarks|Takeaway)"],
            1700,
        )
        print("=" * 100)
        print(
            f"### {bid} | {item.get('published','')} | {item.get('category','')} "
            f"| {len(item.get('authors',[]))} authors"
        )
        print("TITLE:", item.get("title", "?"))
        print("\n--ABSTRACT--\n", item.get("summary", "?"))
        print("\n--INTRO--\n", re.sub(r"\s+", " ", intro).strip()[:500] + "...")
        if concl:
            print("\n--CONCL--\n", re.sub(r"\s+", " ", concl).strip()[:500] + "...")
        else:
            print("\n--CONCL--\n [slice failed -- summarize from abstract + intro only, and say so]")
        print()


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "index"
    master = stage_index()
    if stage in ("fetch", "digest", "all"):
        selected = json.load(open("C:/Users/akbon/OneDrive/Documents/GitHub/FourD/vfd_bridge/lit/selected_papers.json"))
        stage_fetch(selected)
    if stage in ("digest", "all"):
        stage_digest(master)
