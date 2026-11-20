#!/usr/bin/env python3
"""Entity-identity arXiv sweep for E1/E2 claims (Mode B: claim-derived queries).
Adapted from arxiv-research-sweep/scripts/arxiv_sweep.py conventions."""
import json, os, re, sys, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET

QUERIES = [
    ('q01', 'all:"self-aware substructures"'),
    ('q02', 'abs:"diachronic identity"'),
    ('q03', 'abs:"personal identity" AND (abs:uploading OR abs:simulation OR abs:"artificial")'),
    ('q04', 'all:"artificial life" AND abs:individuality'),
    ('q05', 'abs:autopoiesis AND (abs:model OR abs:simulation)'),
    ('q06', 'abs:"substrate independence" AND abs:identity'),
    ('q07', 'abs:"closure of constraints"'),
    ('q08', 'abs:"persistence conditions"'),
    ('q09', 'all:"emergent individuals" OR abs:"emergent individual"'),
    ('q10', 'abs:"reidentification" OR abs:"re-identification"'),
    ('q11', 'all:patternism OR abs:"pattern theory of identity"'),
    ('q12', 'abs:"numerical identity" AND (abs:agent OR abs:machine OR abs:artificial)'),
]
MAX_RESULTS = 60
API = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
ROOT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(ROOT, "raw"), exist_ok=True)

def base_id(raw):
    return re.sub(r"v\d+$", "", raw.rsplit("/", 1)[-1].strip())

def fetch(url, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(3 * (i + 1))
    return b""

papers = {}
for qid, q in QUERIES:
    params = urllib.parse.urlencode({"search_query": q, "start": 0,
                                     "max_results": MAX_RESULTS, "sortBy": "relevance"})
    raw = fetch(f"{API}?{params}")
    open(os.path.join(ROOT, "raw", f"{qid}.xml"), "wb").write(raw)
    ns = {"a": ATOM[1:-1]}
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        print(qid, "PARSE FAIL bytes", len(raw)); continue
    entries = root.findall("a:entry", ns)
    n = 0
    for e in entries:
        eid = base_id(e.findtext("a:id", "", ns))
        title = " ".join(e.findtext("a:title", "", ns).split())
        if eid not in papers:
            papers[eid] = dict(
                id=eid, title=title,
                authors=[a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)],
                published=e.findtext("a:published", "", ns)[:10],
                cat=(e.find(f"a:category", ns) or ET.Element("c")).get("term", ""),
                abstract=" ".join(e.findtext("a:summary", "", ns).split()),
                queries=[])
        papers[eid]["queries"].append(qid)
        n += 1
    print(qid, f"bytes={len(raw)} entries={n}", flush=True)
    time.sleep(3.5)

out = sorted(papers.values(), key=lambda p: (-len(p["queries"]), p["published"]))
json.dump(out, open(os.path.join(ROOT, "master.json"), "w"), indent=1)
print(f"TOTAL UNIQUE {len(out)}")
for p in out[:45]:
    print(f"[x{len(p['queries'])}] {p['id']} {p['published']} {p['cat'][:10]:10s} {p['title'][:90]}")
