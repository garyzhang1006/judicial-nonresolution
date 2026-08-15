"""Step 22: additional figures, each chart form chosen for its quantity."""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT
FIG = os.path.join(os.path.dirname(OUT), "paper", "generated")
plt.rcParams.update({"font.size": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200,
                     "savefig.bbox": "tight"})
BLUE, ORANGE, GREY = "#2b6cb0", "#c05621", "#a0aec0"

# 1. eligibility funnel -> waterfall, since each step is a subtraction
pv = json.load(open(os.path.join(OUT, "05_prevalence.json")))
val = [v for _, v in pv["audit"]]
lab = ["text", "date", "$\\geq$1000\nchars", "not\nblocked", "dedup"]
fig, ax = plt.subplots(figsize=(3.2, 1.45))
run = val[0]
ax.bar(0, val[0] / 1e6, color=BLUE, width=.62)
for i in range(1, len(val)):
    drop = (run - val[i]) / 1e6
    ax.bar(i, max(drop, .012), bottom=val[i] / 1e6, color=ORANGE, width=.62)
    ax.text(i, run / 1e6 + .04, f"-{run-val[i]:,}".replace(",", ","),
            ha="center", fontsize=5.2, color=ORANGE)
    ax.plot([i - 1 + .31, i + .31], [run / 1e6] * 2, color=GREY, lw=.6, ls=":")
    run = val[i]
ax.bar(len(val), val[-1] / 1e6, color=BLUE, width=.62)
ax.set_xticks(range(len(val) + 1)); ax.set_xticklabels(lab + ["eligible"], fontsize=6)
ax.set_ylabel("opinions (M)"); ax.set_ylim(0, 2.35)
ax.text(len(val), val[-1] / 1e6 + .06, f"{val[-1]/1e6:.2f}M", ha="center", fontsize=6)
fig.savefig(os.path.join(FIG, "fig_funnel.pdf")); plt.close(fig); print("fig_funnel waterfall")

# 2. per-expression counts -> horizontal bars
t = pd.read_csv(os.path.join(OUT, "05_prev_trigger.csv")).sort_values("n_opinions")
# drawn near its printed size so the expressions stay legible; the bar ends are
# not labelled because the axis already carries the value
fig, ax = plt.subplots(figsize=(2.75, 1.95))
y = np.arange(len(t))
ax.barh(y, t["n_opinions"] / 1e3, color=BLUE, height=.7)
ax.set_yticks(y); ax.set_yticklabels(t["expression"], fontsize=6.6)
ax.set_ylim(-.65, len(t) - .35)
ax.set_xlabel("opinions containing the expression (thousands)", fontsize=6.6)
ax.set_xlim(0, t["n_opinions"].max() / 1e3 * 1.03)
ax.tick_params(axis="x", labelsize=6.6)
fig.savefig(os.path.join(FIG, "fig_trigger.pdf")); plt.close(fig); print("fig_trigger bars")

# 3. stratum accuracy -> dumbbell, since the quantity of interest is the gap
s = pd.read_csv(os.path.join(OUT, "09_by_stratum.csv"))
s = s[s.context == "SENT"].set_index("stratum")
order = [x for x in ["CTRL_RAND", "CTRL_HOLD", "H2", "PLAIN", "H1"] if x in s.index]
y = np.arange(len(order))
fig, ax = plt.subplots(figsize=(3.3, 1.6))
for i, o in enumerate(order):
    a, b = s.loc[o, "lexical_acc"], s.loc[o, "tfidf_acc"]
    ax.plot([a, b], [i, i], color=GREY, lw=1.4, zorder=1)
    ax.scatter(a, i, s=26, color=GREY, zorder=2, label="dictionary" if i == 0 else "")
    ax.scatter(b, i, s=26, color=BLUE, zorder=3, label="TF-IDF" if i == 0 else "")
ax.set_yticks(y); ax.set_yticklabels([o.replace("CTRL_", "ctrl ").lower() for o in order],
                                     fontsize=6)
ax.set_xlabel("accuracy"); ax.set_xlim(0, 1.08)
ax.set_ylim(-0.6, len(order) - 0.15)
ax.legend(fontsize=6, frameon=False, ncol=2, loc="lower center",
          bbox_to_anchor=(0.5, 1.0), handletextpad=.3, columnspacing=1.2)
fig.savefig(os.path.join(FIG, "fig_strata.pdf")); plt.close(fig); print("fig_strata dumbbell")

# 4. learning curve -> line with the individual resamples shown
lc = pd.read_csv(os.path.join(OUT, "20_learning_curve.csv"))
fig, ax = plt.subplots(figsize=(3.2, 1.4))
ax.fill_between(lc.n_train, lc.h1_acc_mean - lc.h1_acc_sd,
                lc.h1_acc_mean + lc.h1_acc_sd, color=BLUE, alpha=.16, lw=0)
ax.plot(lc.n_train, lc.h1_acc_mean, marker="o", ms=3.4, lw=1.3, color=BLUE,
        label="TF-IDF")
ax.axhline(lc.lexical.iloc[0], ls="--", lw=1, color=ORANGE,
           label="frozen dictionary")
ax.set_xlabel("annotated training items"); ax.set_ylabel("accuracy on H1")
lo = min(lc.h1_acc_mean.min() - lc.h1_acc_sd.max(), lc.lexical.iloc[0]) - .02
ax.set_ylim(lo, lc.h1_acc_mean.max() + .045)
ax.legend(fontsize=6, frameon=False, ncol=2, loc="lower center",
          bbox_to_anchor=(0.5, 1.0), handletextpad=.4, columnspacing=1.2)
fig.savefig(os.path.join(FIG, "fig_lc.pdf")); plt.close(fig); print("fig_lc band")

# 5. the two matching strategies -> tapering ribbons, so the attrition each one
# suffers is read off the taper rather than compared across separate bars
v = json.load(open(os.path.join(OUT, "21_validation.json")))
pvd = json.load(open(os.path.join(OUT, "19_proxy_validation.json")))


def ribbon(ax, xs, vals, y0, scale, color, tail=None):
    """Half-height proportional to the count, smoothstepped between waypoints."""
    for i in range(len(xs) - 1):
        seg = np.linspace(xs[i], xs[i + 1], 60)
        u = (seg - xs[i]) / (xs[i + 1] - xs[i])
        h = (vals[i] + (vals[i + 1] - vals[i]) * u * u * (3 - 2 * u)) * scale / 2
        c = tail if (tail and i == len(xs) - 2) else color
        ax.fill_between(seg, y0 - h, y0 + h, color=c, lw=0)


prox = [pvd["n_checked"], pvd["n_genuine"]]
prop = [v["n_matched"], v["n_genuine"], v["n_distinct"], v["esc_dedup"]]
SCALE = .50 / max(prop)
YP, YQ = 1.06, 0.45
fig, ax = plt.subplots(figsize=(3.4, 1.30))
ribbon(ax, [0, 1], prox, YP, SCALE, GREY)
ribbon(ax, [0, 1, 2, 3], prop, YQ, SCALE, BLUE, tail=ORANGE)
# counts above each ribbon, the verified share below it, so neither runs into
# the other or into the stage labels under the axis
for x, n_ in zip([0, 1], prox):
    ax.text(x, YP + n_ * SCALE / 2 + .04, str(n_), ha="center", fontsize=6, color=GREY)
for x, n_ in zip([0, 1, 2, 3], prop):
    ax.text(x, YQ + n_ * SCALE / 2 + .04, str(n_), ha="center", fontsize=6,
            color=ORANGE if x == 3 else BLUE)
ax.text(-.10, YP, "proximity", ha="right", va="center", fontsize=6, color=GREY)
ax.text(-.10, YQ, "proposition", ha="right", va="center", fontsize=6, color=BLUE)
ax.text(1, YP - prox[1] * SCALE / 2 - .05, f"{pvd['precision']:.0%} genuine",
        ha="center", va="top", fontsize=5.6, color=GREY)
ax.text(1, YQ - prop[1] * SCALE / 2 - .05, f"{v['precision']:.0%} genuine",
        ha="center", va="top", fontsize=5.6, color=BLUE)
ax.set_xticks([0, 1, 2, 3]); ax.set_xlim(-.78, 3.32); ax.set_ylim(.06, 1.32)
ax.set_xticklabels(["hand-checked\nflags", "genuine", "distinct\npropositions",
                    "later called\na holding"], fontsize=6)
ax.set_yticks([]); ax.tick_params(length=0)
for sp in ("left", "bottom"):
    ax.spines[sp].set_visible(False)
fig.savefig(os.path.join(FIG, "fig_escal.pdf")); plt.close(fig); print("fig_escal ribbons")
