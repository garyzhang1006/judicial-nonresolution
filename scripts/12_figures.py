"""Step 12: figures."""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, FED_APP, COURT_LABEL

FIG = os.path.join(os.path.dirname(OUT), "paper", "generated")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight"})


def have(fn):
    p = os.path.join(OUT, fn)
    return p if os.path.exists(p) else None


# ---------------------------------------------------------------- fig 1: time
p = have("05_prev_year.csv")
if p:
    y = pd.read_csv(p)
    y = y[(y["year"] >= 1891) & (y["year"] <= 2025) & (y["n"] >= 150)]
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    ax.plot(y["year"], 100 * y["rate"], lw=1.0, color="#1b3a5c")
    ax.fill_between(y["year"], 100 * y["lo"], 100 * y["hi"], alpha=.22,
                    color="#1b3a5c", lw=0)
    ax.set_xlabel("year of decision")
    ax.set_ylabel("\\% of opinions")
    ax.set_title("Opinions with explicit non-resolution language", fontsize=8)
    ax.grid(axis="y", lw=.3, alpha=.5)
    fig.savefig(os.path.join(FIG, "fig_time.pdf"))
    plt.close(fig)
    print("fig_time.pdf")

# ------------------------------------------------------- fig 2: court x period
p = have("05_prev_court_period.csv")
if p:
    cp = pd.read_csv(p)
    piv = cp.pivot_table(index="court", columns="period", values="rate")
    piv = piv.reindex([c for c in FED_APP if c in piv.index])
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    im = ax.imshow(100 * piv.values, cmap="BuPu", aspect="auto")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=35, ha="right", fontsize=6)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([COURT_LABEL.get(c, c) for c in piv.index], fontsize=6)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{100*v:.1f}", ha="center", va="center",
                        fontsize=5,
                        color="white" if 100 * v > np.nanpercentile(100*piv.values, 65) else "black")
    fig.colorbar(im, ax=ax, label="\\% of opinions", shrink=.85)
    ax.set_title("Non-resolution language by court and era", fontsize=8)
    fig.savefig(os.path.join(FIG, "fig_heat.pdf"))
    plt.close(fig)
    print("fig_heat.pdf")

# ---------------------------------------------------------- fig 3: context curve
p = have("09_by_stratum.csv")
if p:
    st = pd.read_csv(p)
    order = ["SENT", "W256", "W1024", "W4096"]
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    marks = {"PLAIN": "o", "H1": "s", "H2": "^", "CTRL_HOLD": "D", "CTRL_RAND": "v"}
    for s, g in st.groupby("stratum"):
        g = g.set_index("context").reindex(order).reset_index()
        ax.plot(range(len(order)), 100 * g["tfidf_acc"], marker=marks.get(s, "o"),
                ms=3, lw=1, label=s)
    lex = st.groupby("context")["lexical_acc"].mean().reindex(order)
    ax.plot(range(len(order)), 100 * lex.values, ls="--", color="grey", lw=1,
            label="lexical rule (mean)")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=6)
    ax.set_xlabel("context window")
    ax.set_ylabel("accuracy (\\%)")
    ax.legend(fontsize=5.2, frameon=False, ncol=2)
    ax.grid(axis="y", lw=.3, alpha=.5)
    ax.set_title("Accuracy by stratum and context width", fontsize=8)
    fig.savefig(os.path.join(FIG, "fig_context.pdf"))
    plt.close(fig)
    print("fig_context.pdf")

# ------------------------------------------------------------ fig 4: layer two
p = have("10b_annotated.parquet")
if p:
    l2 = pd.read_parquet(p)
    u = l2[l2["usable"]]
    if len(u):
        fig, ax = plt.subplots(figsize=(3.3, 2.0))
        labs = ["UNRESOLVED", "DECIDED"]
        w, x = 0.38, np.arange(4)
        for k, lab in enumerate(labs):
            g = u[u["origin_label"] == lab]
            if not len(g):
                continue
            vals = [100 * (g["status"] == s).mean() for s in range(4)]
            ax.bar(x + (k - .5) * w, vals, w,
                   label=f"origin {lab.lower()} ($n$={len(g)})",
                   color=["#1b3a5c", "#c8794b"][k])
        ax.set_xticks(x)
        ax.set_xticklabels(["$L_0$ open", "$L_1$ assumed", "$L_2$ supported",
                            "$L_3$ established"], fontsize=6)
        ax.set_ylabel("\\% of citing passages")
        ax.legend(fontsize=6, frameon=False)
        ax.grid(axis="y", lw=.3, alpha=.5)
        ax.set_title("Status ascribed by later opinions", fontsize=8)
        fig.savefig(os.path.join(FIG, "fig_layer2.pdf"))
        plt.close(fig)
        print("fig_layer2.pdf")
print("figures done")
