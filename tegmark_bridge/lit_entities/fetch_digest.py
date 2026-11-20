import json, os, re, sys, time, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
os.makedirs("pdfs", exist_ok=True); os.makedirs("txt", exist_ok=True)
MIN_PDF_BYTES = 50_000
SELECTED = [
 ("9704009", "Tegmark ultimate ensemble — SAS original definition"),
 ("2607.00220", "category theory account of AI identity — diachronic identity formalism"),
 ("2611.05292", "identity as state not attribute — B1 analog"),
 ("1802.01161", "puzzle about further facts — bearer-null identity philosophy"),
 ("2407.21086", "non-Platonic autopoiesis of CA glider in Lenia — observable glider entity"),
 ("2005.03742", "Lenia expanded universe — glider taxonomy/entity catalogs"),
 ("2402.10236", "sensorimotor agency in CA via diversity search — entity detection"),
 ("2511.17786", "active droplet lifecycle quantitative analysis — lifespan metrics"),
 ("2404.04374", "semantic closure enables open-ended evolution"),
 ("1512.08310", "autopoietic cells with cognition simulations"),
 ("2512.09352", "causal symmetrization signature of operational autonomy — certificate analog"),
 ("2506.03233", "trustworthiness metaphysics of AI systems — persistence conditions"),
 ("1102.1691", "schema redescription in CA — Dretske/Dennett entity extraction"),
 ("2511.05207", "financial market sim heterogeneous preferences — market bridge"),
 ("1504.06320", "uploading gradual-vs-copy identity criteria"),
]
ids = {i for i,_ in SELECTED}
def fetch(url, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r: return r.read()
        except Exception as e:
            print("retry", url[-25:], i, e, flush=True); time.sleep(8*(i+1))
    return b""
for pid, why in SELECTED:
    fn = f"pdfs/{pid.replace('/','_')}.pdf"
    if os.path.exists(fn) and os.path.getsize(fn) > MIN_PDF_BYTES:
        print("have", pid); continue
    raw = fetch(f"https://arxiv.org/pdf/{pid}")
    if len(raw) >= MIN_PDF_BYTES:
        open(fn,"wb").write(raw); print("OK", pid, len(raw), flush=True)
    else:
        print("FAIL", pid, len(raw), flush=True)
    time.sleep(4)
import fitz
for pid,_ in SELECTED:
    fn = f"pdfs/{pid.replace('/','_')}.pdf"
    if not os.path.exists(fn): continue
    try:
        doc = fitz.open(fn); t = "\n".join(p.get_text() for p in doc)
        open(f"txt/{pid.replace('/','_')}.txt","w",encoding="utf-8").write(t)
        print("TXT", pid, len(t), flush=True)
    except Exception as e:
        print("SCANNED?", pid, e, flush=True)
CLAIMS = [
 "persistence", "identity criterion", "diachronic", "self-aware substructure",
 "glider", "lifespan", "reidentification", "numerical identity",
 "persistence condition", "autopoiesis", "individual", "survive", "death",
 "substrate", "pattern", "occupancy", "tracking",
]
for pid, why in SELECTED:
    fn = f"txt/{pid.replace('/','_')}.txt"
    if not os.path.exists(fn):
        print("\n=== NO TEXT:", pid, why); continue
    t = open(fn,encoding="utf-8").read()
    print(f"\n=== {pid}  {why}  ({len(t)} chars)")
    for c in CLAIMS:
        hits = [m.start() for m in re.finditer(re.escape(c), t, re.I)][:2]
        for h in hits:
            lo, hi = max(0,h-260), min(len(t),h+380)
            print(f"  [{c}] …{' '.join(t[lo:hi].split())}…")
