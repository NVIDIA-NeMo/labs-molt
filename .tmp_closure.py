import ast, os, sys, collections
ROOT = "nemo_automodel"
mods = {}  # module name -> path
for dp, dn, fn in os.walk(ROOT):
    if "__pycache__" in dp: continue
    for f in fn:
        if f.endswith(".py"):
            p = os.path.join(dp, f)
            m = p[:-3].replace("/", ".")
            if m.endswith(".__init__"): m = m[:-9]
            mods[m] = p
def lines(p):
    with open(p, errors="ignore") as fh: return sum(1 for _ in fh)
L = {m: lines(p) for m, p in mods.items()}
def resolve(name):
    # return module names that exist for dotted name (module or package)
    out = []
    if name in mods: out.append(name)
    return out
def imports_of(m):
    p = mods[m]
    try: tree = ast.parse(open(p, errors="ignore").read())
    except SyntaxError: return set()
    pkg = m if mods[m].endswith("__init__.py") else m.rsplit(".", 1)[0]
    res = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith(ROOT):
                    res.update(resolve(a.name))
                    # parents
                    parts = a.name.split("."); 
                    for i in range(1, len(parts)+1):
                        res.update(resolve(".".join(parts[:i])))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg.split(".")
                base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
                modname = ".".join(base + ([node.module] if node.module else []))
            else:
                modname = node.module or ""
            if not modname.startswith(ROOT): continue
            parts = modname.split(".")
            for i in range(1, len(parts)+1):
                res.update(resolve(".".join(parts[:i])))
            for a in node.names:
                res.update(resolve(modname + "." + a.name))
    return res
KEEP_MODEL_DIRS = sys.argv[1].split(",")
roots = set()
for d in KEEP_MODEL_DIRS:
    for m in mods:
        if m.startswith(f"{ROOT}.components.models.{d}") : roots.add(m)
MOLT_ROOTS = """nemo_automodel
nemo_automodel._transformers.auto_model
nemo_automodel._transformers.mfu
nemo_automodel._transformers.registry
nemo_automodel._transformers.model_init
nemo_automodel.components.moe.router_replay
nemo_automodel.components.moe.layers
nemo_automodel.components.moe.megatron.moe_utils
nemo_automodel.components.distributed.context_parallel
nemo_automodel.components.distributed.mesh_utils
nemo_automodel.components.distributed.mesh
nemo_automodel.components.distributed.config
nemo_automodel.shared.parameter_names
nemo_automodel.components.utils.model_utils
nemo_automodel.components.training.utils
nemo_automodel.components._peft.lora
nemo_automodel.components.optim.dion
nemo_automodel.components.models.common.utils
nemo_automodel.components.models.common
nemo_automodel.components.checkpoint.stateful_wrappers
nemo_automodel.components.checkpoint.checkpointing""".split()
roots.update(m for m in MOLT_ROOTS if m in mods)
missing = [m for m in MOLT_ROOTS if m not in mods]
seen = set(); stack = list(roots)
while stack:
    m = stack.pop()
    if m in seen: continue
    seen.add(m)
    for d in imports_of(m):
        if d not in seen: stack.append(d)
total = sum(L.values()); kept = sum(L[m] for m in seen)
print(f"missing roots: {missing}")
print(f"modules: total {len(mods)} kept {len(seen)} | lines: total {total} kept {kept} ({kept/total:.0%}) deleted {total-kept}")
def bucket(m):
    parts = m.split(".")
    if len(parts) >= 4 and parts[1] == "components" and parts[2] == "models": return "models/" + parts[3]
    if len(parts) >= 3 and parts[1] == "components": return "components/" + parts[2]
    return parts[1] if len(parts) > 1 else "(top)"
kb = collections.Counter(); tb = collections.Counter()
for m in mods:
    tb[bucket(m)] += L[m]
    if m in seen: kb[bucket(m)] += L[m]
print("\nbucket | total | kept | deleted")
for b, t in sorted(tb.items(), key=lambda x: -x[1]):
    if t >= 300 or kb[b]: print(f"{b} | {t} | {kb[b]} | {t-kb[b]}")
# kept model dirs but pulled by closure not in KEEP list
extra = sorted({bucket(m) for m in seen if bucket(m).startswith("models/") and bucket(m)[7:] not in KEEP_MODEL_DIRS})
print("\nmodel dirs pulled in by imports (not in keep list):", extra)
# who pulls the unwanted buckets
UNWANTED = ["components/quantization","components/distributed","components/moe","components/speculative","components/loss","components/datasets","recipes"]
if len(sys.argv) > 2:
    for m in sorted(seen):
        b = bucket(m)
        if b in sys.argv[2].split(","):
            importers = [x for x in seen if m in imports_of(x) and x != m]
            print(f"  {m} ({L[m]}) <- {importers[:4]}")

# ---- second pass: cut edges into droppable modules
import re
DROP_PAT = sys.argv[3].split(",") if len(sys.argv) > 3 else []
cut = {m for m in seen if any(re.search(p, m) for p in DROP_PAT)}
cut_lines = sum(L[m] for m in cut)
edges = collections.Counter()
for x in seen - cut:
    for d in imports_of(x):
        if d in cut: edges[x] += 1
print(f"\nsecond pass: cut {len(cut)} modules / {cut_lines} lines; import sites to edit: {sum(edges.values())} in {len(edges)} files")
for x, n in sorted(edges.items(), key=lambda t: -t[1])[:25]:
    print(f"  {x.replace('nemo_automodel.','')} ({L[x]}): {n} -> {[d.replace('nemo_automodel.','') for d in imports_of(x) if d in cut][:5]}")
kept2 = kept - cut_lines
print(f"\nafter second pass: kept {kept2} lines ({kept2/total:.0%}), deleted {total-kept2}")
kb2 = collections.Counter()
for m in seen - cut: kb2[bucket(m)] += L[m]
print("\nbucket | kept after pass 2")
for b, v in sorted(kb2.items(), key=lambda x: -x[1]):
    if v >= 200: print(f"{b} | {v}")

# ---- dump lists
import json
open("/tmp/am_keep_modules.json","w").write(json.dumps(sorted(m for m in seen - cut)))
open("/tmp/am_cut_modules.json","w").write(json.dumps(sorted(cut)))
open("/tmp/am_delete_files.txt","w").write("\n".join(sorted(mods[m] for m in mods if m not in seen or m in cut)) + "\n")
print("\nfiles to delete:", sum(1 for m in mods if m not in seen or m in cut), "| kept files:", len(seen - cut))
