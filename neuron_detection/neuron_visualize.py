# neuron_visualize.py  (ATTN + FFN version)  ───────────────────────────
from pathlib import Path, PurePath
import re, ast, itertools
import pandas as pd
import matplotlib.pyplot as plt

# 1. where the *.txt results live
# OUT_DIR  = Path("output_all_neurons/google")
# OUT_DIR  = Path("output_neurons/mistralai")
# OUT_DIR  = Path("output_neurons_hrl/google")
OUT_DIR  = Path("output_neurons/google")
# OUT_DIR  = Path("output_sw_neurons/google")


# 2. capture the two-letter language code that sits right before "gsm…"
MODEL_RE = re.compile(r"(?P<lang>[a-z]{2})(?=gsm)")

# ── helper ────────────────────────────────────────────────────────────
def load_lang_file(path: Path):
    """
    Return
        {'ATTN': {layer -> set(neuron_ids)},
         'FFN' : {layer -> set(neuron_ids)}}
    from a 5-line output file:
        0: fwd_up   (FFN-up)
        1: fwd_down (FFN-down)
        2: q        (ATTN-Q)
        3: k        (ATTN-K)
        4: v        (ATTN-V)
    """
    kind_map   = ["FFN",  "FFN",  "ATTN", "ATTN", "ATTN"]
    structures = ["fwd_up","fwd_down","q","k","v"]

    buckets = {"ATTN": {}, "FFN": {}}

    with path.open() as f:
        for kind, struct, line in zip(kind_map, structures, f):
            parsed = ast.literal_eval(line.strip())      # {layer: set}
            for layer, ids in parsed.items():
                buckets[kind].setdefault(int(layer), set()).update(ids)

    return buckets            # two separate dicts


# ── 2. load every file ────────────────────────────────────────────────
lang2attn = {}
lang2ffn  = {}

for p in OUT_DIR.glob("*.txt"):
    m = MODEL_RE.search(p.stem)
    if not m:
        print("skip (no lang):", p.name)
        continue
    lang = m.group("lang")
    try:
        buckets = load_lang_file(p)
    except Exception as e:
        print("could not parse", p.name, ":", e)
        continue
    lang2attn[lang] = buckets["ATTN"]
    lang2ffn [lang] = buckets["FFN"]

if not lang2attn:
    raise RuntimeError("No languages loaded – check OUT_DIR & regex")

langs   = sorted(lang2attn)
layers  = sorted({l for d in lang2attn.values() for l in d})

# ── 3a. heat-map for ATTENTION ───────────────────────────────────────
def make_heat(lang2bucket, title, fname):
    heat = pd.DataFrame(
        {lang: {layer: len(lang2bucket[lang].get(layer, set()))
                for layer in layers}
         for lang in langs}).T

    plt.figure(figsize=(10, max(3, 0.4*len(langs))))
    plt.imshow(heat.values, aspect='auto')
    plt.xticks(range(len(layers)), layers, rotation=90, fontsize=6)
    plt.yticks(range(len(langs)), langs)
    plt.colorbar(label="# neurons")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.show()

make_heat(lang2attn, "Language-specific neurons per layer (ATTN)",
          "attn_layer_heatmap.png")

# ── 3b. heat-map for FFN ─────────────────────────────────────────────
make_heat(lang2ffn,  "Language-specific neurons per layer (FFN)",
          "ffn_layer_heatmap.png")


# ── 3c. line-plot per language & per layer ───────────────────────────
import matplotlib.cm as cm

def line_plot(lang2bucket, title, fname):
    """
    lang2bucket = {'en': {layer -> set() }, ...}
    """
    # DataFrame: rows = layer, cols = lang
    df = pd.DataFrame({
        lang: {layer: len(b.get(layer, set())) for layer in layers}
        for lang, b in lang2bucket.items()
    })
    
    n_lang  = len(df.columns)
    cmap    = cm.get_cmap('tab20', n_lang)      
    ax      = df.plot(
        figsize=(10, 4),
        lw=2,
        colormap=cmap,
        marker='o',  
    )
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("# language-specific neurons")
    ax.set_title(title)
    ax.set_xticks(layers)
    ax.set_xticklabels(layers, rotation=0, fontsize=7)
    ax.grid(True, ls='--', alpha=.3)
    ax.legend(title="language", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.show()
line_plot(lang2ffn,
          "Per-language language-specific neurons (FFN)",
          "ffn_layer_lineplot.png")

line_plot(lang2attn,
          "Per-language language-specific neurons (ATTN)",
          "attn_layer_lineplot.png")





# ── 4. overlap matrix ────────────────────────────────────────────────
def make_overlap(lang2bucket, title, fname):
    ov = pd.DataFrame(index=langs, columns=langs, dtype=float)

    for l1, l2 in itertools.product(langs, repeat=2):
        b1, b2 = lang2bucket[l1], lang2bucket[l2]
        shared = sum(len(b1.get(k, set()) & b2.get(k, set())) for k in layers)
        total  = sum(len(b2.get(k, set()))                   for k in layers)
        ov.loc[l1, l2] = shared / total if total else 0.0

    plt.figure(figsize=(6, 5))
    plt.imshow(ov.values, vmin=0, vmax=1, aspect='equal')
    plt.xticks(range(len(langs)), langs, rotation=45)
    plt.yticks(range(len(langs)), langs)
    plt.colorbar(label="fraction of L2 neurons shared with L1")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.show()

# (1) Attention 専用
make_overlap(lang2attn,
             "Neuron overlap matrix (ATTN only)",
             "neuron_overlap_attn.png")

# (2) FFN 専用
make_overlap(lang2ffn,
             "Neuron overlap matrix (FFN only)",
             "neuron_overlap_ffn.png")

def combined(lang):
    comb = {}
    for bucket in (lang2attn[lang], lang2ffn[lang]):
        for layer, ids in bucket.items():
            comb.setdefault(layer, set()).update(ids)
    return comb


overlap = pd.DataFrame(index=langs, columns=langs, dtype=float)
for l1, l2 in itertools.product(langs, repeat=2):
    n1, n2 = combined(l1), combined(l2)
    shared = sum(len(n1.get(k, set()) & n2.get(k, set())) for k in layers)
    total  = sum(len(n2.get(k, set())) for k in layers)
    overlap.loc[l1, l2] = shared / total if total else 0.0

plt.figure(figsize=(6, 5))
plt.imshow(overlap.values, vmin=0, vmax=1, aspect='equal')
plt.xticks(range(len(langs)), langs, rotation=45)
plt.yticks(range(len(langs)), langs)
plt.colorbar(label="fraction of L2 neurons shared with L1")
plt.title("Neuron overlap matrix (ATTN ∪ FFN)")
plt.tight_layout()
plt.savefig("neuron_overlap_matrix.png", dpi=300, bbox_inches="tight")
plt.show()


p = next(Path(OUT_DIR).glob("*.txt"))  # 1ファイルだけ見る
with p.open() as f:
    for i, line in enumerate(f):
        d = ast.literal_eval(line)
        total = sum(len(s) for s in d.values())
        print(f"line {i}:  total neurons = {total}")
        
with open(p) as f:                    # どのファイルでも OK
    for i, line in enumerate(f):
        ids = ast.literal_eval(line)
        mx  = max(max(s) for s in ids.values())
        print(i, mx)