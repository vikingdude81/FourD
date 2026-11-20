import json, os
ROOT = r"C:\Users\akbon\OneDrive\Documents\GitHub\FourD"
D = json.load(open(os.path.join(ROOT, "tegmark_bridge/viz/traces.json")))
D.pop("C4", None)
Q = 48
for sc in D["scenes"]:
    for fr in sc["frames"]:
        fr["u"] = [max(-Q, min(Q, round(v * Q))) for v in fr["u"]]
# one shared centers list at top level
tmpl = open(os.path.join(ROOT, "tegmark_bridge/viz/trace_view.tmpl.html"),
            encoding="utf-8").read()
out = tmpl.replace("__DATA__", json.dumps(D, separators=(",", ":")))
p = os.path.join(ROOT, "tegmark_bridge/viz/trace_view.html")
open(p, "w", encoding="utf-8").write(out)
print("wrote", p, os.path.getsize(p), "bytes")
