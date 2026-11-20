import json, os, re, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
REMAIN = [('q10','abs:"reidentification" OR abs:"re-identification"'),
          ('q11','all:patternism OR abs:"pattern theory of identity"'),
          ('q12','abs:"numerical identity" AND (abs:agent OR abs:machine OR abs:artificial)')]
REMAIN = [(q,s) for q,s in REMAIN if not os.path.exists(f"raw/{q}.xml")]
ATOM = "{http://www.w3.org/2005/Atom}"
API = "https://export.arxiv.org/api/query"
def base_id(raw): return re.sub(r"v\d+$", "", raw.rsplit("/",1)[-1].strip())
def fetch(url, tries=5):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r: return r.read()
        except Exception as e:
            print("retry", i, e, flush=True); time.sleep(30)
    return b""
def parse(raw, qid, papers):
    ns = {"a": ATOM[1:-1]}
    try: root = ET.fromstring(raw)
    except ET.ParseError: return 0
    n = 0
    for e in root.findall("a:entry", ns):
        eid = base_id(e.findtext("a:id","",ns))
        if eid not in papers:
            catel = e.find("a:category", ns)
            papers[eid] = dict(id=eid, title=" ".join(e.findtext("a:title","",ns).split()),
                authors=[a.findtext("a:name","",ns) for a in e.findall("a:author",ns)],
                published=e.findtext("a:published","",ns)[:10],
                cat=catel.get("term","") if catel is not None else "",
                abstract=" ".join(e.findtext("a:summary","",ns).split()), queries=[])
        papers[eid]["queries"].append(qid); n += 1
    return n
papers = {}
for f in sorted(os.listdir("raw")):
    qid = f.split(".")[0]
    parse(open(os.path.join("raw",f),"rb").read(), qid, papers)
for qid, q in REMAIN:
    p = urllib.parse.urlencode({"search_query": q, "start": 0, "max_results": 60, "sortBy": "relevance"})
    raw = fetch(f"{API}?{p}")
    if raw:
        open(os.path.join("raw", f"{qid}.xml"), "wb").write(raw)
        print(qid, "bytes", len(raw), flush=True)
    time.sleep(8)
out = sorted(papers.values(), key=lambda p: (-len(p["queries"]), p["published"]))
json.dump(out, open("master.json","w"), indent=1)
print("TOTAL UNIQUE", len(out))
for p in out[:60]:
    print(f"[x{len(p['queries'])}] {p['id']} {p['published']} {p['cat'][:10]:10s} {p['title'][:88]}")
