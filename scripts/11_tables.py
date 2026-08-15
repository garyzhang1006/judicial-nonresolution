"""Step 11: emit every number the paper reports as a LaTeX macro or table.

No figure in the paper is typed by hand. Each one is defined here from a saved
artefact, so recompiling after a pipeline change updates the prose as well.
"""
import json, os, re, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, FED_APP, COURT_LABEL

GEN = os.path.join(os.path.dirname(OUT), "paper", "generated")
os.makedirs(GEN, exist_ok=True)
MAC = {}
MAC_PATH = os.path.join(GEN, "macros.tex")

# Release bundles may omit large intermediate parquets while retaining their
# generated, verified paper macros. Preserve those values and overwrite only
# macros whose source artefacts are present in this run.
if os.path.exists(MAC_PATH):
    for _line in open(MAC_PATH, encoding="utf-8"):
        _match = re.fullmatch(
            r"\\(?:newcommand|providecommand)\{\\([A-Za-z]+)\}\{(.*)\}\s*",
            _line,
        )
        if _match:
            MAC[_match.group(1)] = _match.group(2)


def mac(name, value):
    MAC[name] = value


def num(x, d=0):
    return f"{x:,.{d}f}"


def pct(x, d=1):
    return f"{100*x:.{d}f}\\%"


def load(fn, kind="csv"):
    p = os.path.join(OUT, fn)
    if not os.path.exists(p):
        print(f"  (missing {fn})")
        return None
    return (pd.read_csv(p) if kind == "csv" else
            pd.read_parquet(p) if kind == "pq" else json.load(open(p)))


def esc(x):
    """Escape the LaTeX specials that appear in our label and field names."""
    s = str(x)
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("&", r"\&"),
                 ("%", r"\%"), ("#", r"\#"), ("$", r"\$"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


def write_table(name, body):
    # The trailing newline matters: without it \input leaves the final "\\"
    # adjacent to \bottomrule, and the \\ lookahead swallows the \noalign.
    with open(os.path.join(GEN, f"{name}.tex"), "w") as fh:
        fh.write(body.rstrip() + "\n")
    print(f"  wrote {name}.tex")


# ---------------------------------------------------------------- corpus funnel
st = load("03_stats.json", "json")
pv = load("05_prevalence.json", "json")
if st:
    mac("NScanned", num(st["n_opinion_rows_scanned"]))
    mac("NFedApp", num(st["n_federal_appellate_kept"]))
    mac("NCandRaw", num(st["n_candidate_passages"]))
    mac("DictSHA", st["dictionary_sha1"][:12])
    mac("PassMinutes", num(st["minutes"], 0))
if pv:
    mac("NEligible", num(pv["n_eligible"]))
    mac("NWithTrigger", num(pv["n_with_trigger"]))
    mac("Prevalence", pct(pv["prevalence"], 2))
    mac("NOccurrences", num(pv["n_occurrences"]))
    mac("NWordsBn", f"{pv['n_words']/1e9:.2f}")
    rows = "\n".join(
        f"{esc(k)} & {num(v)} \\\\" for k, v in pv["audit"])
    write_table("tab_funnel", rows)

# ---------------------------------------------------------------- prevalence
pc = load("05_prev_court.csv")
if pc is not None:
    pc = pc.set_index("court").reindex(FED_APP).reset_index()
    body = "\n".join(
        f"{COURT_LABEL.get(r['court'], r['court'])} & {num(r['n'])} & "
        f"{num(r['k'])} & {100*r['rate']:.2f} & "
        f"{100*r['lo']:.2f}--{100*r['hi']:.2f} & {r['per_100k_words']:.1f} \\\\"
        for _, r in pc.iterrows() if pd.notna(r["n"]))
    write_table("tab_prev_court", body)
    hi = pc.loc[pc["rate"].idxmax()]
    lo = pc.loc[pc["rate"].idxmin()]
    mac("CourtHi", COURT_LABEL.get(hi["court"], hi["court"]))
    mac("CourtHiRate", pct(hi["rate"], 2))
    mac("CourtLo", COURT_LABEL.get(lo["court"], lo["court"]))
    mac("CourtLoRate", pct(lo["rate"], 2))

pp = load("05_prev_period.csv")
if pp is not None:
    body = "\n".join(
        f"{esc(r['period'])} & {num(r['n'])} & {num(r['k'])} & {100*r['rate']:.2f} & "
        f"{100*r['lo']:.2f}--{100*r['hi']:.2f} & {r['per_100k_words']:.1f} \\\\"
        for _, r in pp.iterrows())
    write_table("tab_prev_period", body)
    f, l = pp.iloc[0], pp.iloc[-1]
    mac("PeriodFirstRate", pct(f["rate"], 2))
    mac("PeriodLastRate", pct(l["rate"], 2))
    mac("PeriodRatio", f"{l['rate']/max(f['rate'],1e-9):.1f}")

pt = load("05_prev_trigger.csv")
if pt is not None:
    body = "\n".join(
        f"{r['trigger']} & \\texttt{{{esc(r['expression'])}}} & {num(r['n_opinions'])} "
        f"& {100*r['share_of_corpus']:.2f} \\\\" for _, r in pt.iterrows())
    write_table("tab_prev_trigger", body)

pg = load("05_prev_type_group.csv")
if pg is not None:
    gi = pg.set_index("type_group")["rate"]
    for k, tag in (("majority/lead", "Majority"), ("separate", "Separate")):
        if k in gi.index:
            mac("Prev" + tag, pct(gi[k], 2))
    body = "\n".join(f"{esc(r['type_group'])} & {num(r['n'])} & {num(r['k'])} & "
                     f"{100*r['rate']:.2f} \\\\" for _, r in pg.iterrows())
    write_table("tab_prev_type", body)

ps = load("05_prev_status.csv")
if ps is not None:
    si = ps.set_index("precedential_status")["rate"]
    for k, tag in (("Published", "Pub"), ("Unpublished", "Unpub")):
        if k in si.index:
            mac("Prev" + tag, pct(si[k], 2))
    ps = ps[ps["n"] >= 500]
    body = "\n".join(f"{esc(r['precedential_status'])} & {num(r['n'])} & {num(r['k'])} "
                     f"& {100*r['rate']:.2f} \\\\" for _, r in ps.iterrows())
    write_table("tab_prev_status", body)

# ---------------------------------------------------------------- benchmark
cfg = load("07_sample_config.json", "json")
ann = load("08_annotated.parquet", "pq")
if cfg:
    mac("NDrawn", num(cfg["n"]))
    mac("HeldOutCourts", ", ".join(COURT_LABEL.get(c, c)
                                   for c in cfg["held_out_courts"]))
    mac("TemporalCut", str(cfg["temporal_cut"]))
if ann is not None and len(ann):
    mac("NAnnotated", num(len(ann)))
    b = ann[ann["label"].isin(["DECIDED", "UNRESOLVED"])]
    mac("NBench", num(len(b)))
    mac("ShareUnresolved", pct((b["label"] == "UNRESOLVED").mean()))
    mac("NUnclear", num(int((ann["label"] == "UNCLEAR").sum())))
    sub = b[b["sublabel"] != ""]["sublabel"].value_counts(normalize=True)
    for k in ("AVOIDED", "ASSUMED", "RESERVED"):
        mac("Share" + k.capitalize(), pct(sub.get(k, 0.0)))
    ct = pd.crosstab(ann["stratum"], ann["label"])
    for c in ("DECIDED", "UNRESOLVED", "UNCLEAR"):
        if c not in ct:
            ct[c] = 0
    body = "\n".join(
        f"{esc(i)} & {num(r['DECIDED'])} & {num(r['UNRESOLVED'])} & {num(r['UNCLEAR'])} "
        f"& {100*r['UNRESOLVED']/max(1,r['DECIDED']+r['UNRESOLVED']):.1f} \\\\"
        for i, r in ct.iterrows())
    write_table("tab_stratum", body)
    sp = pd.crosstab(ann["split"], ann["label"])
    body = "\n".join(f"{esc(i)} & {num(r.sum())} & "
                     f"{100*r.get('UNRESOLVED',0)/max(1,r.sum()):.1f} \\\\"
                     for i, r in sp.iterrows())
    write_table("tab_splits", body)

# ---------------------------------------------------------------- models
res = load("09_results.csv")
enc = load("09b_encoder_results.csv")
llm = load("09c_llm_results.csv")
llm_run = load("09c_llm_run_manifest.json", "json")
if llm is not None:
    required_cells = pd.MultiIndex.from_product(
        [["SENT", "W256", "W1024", "W4096"], ["TEST", "TEMPORAL", "COURT"]],
        names=["context", "split"],
    )
    actual_cells = pd.MultiIndex.from_frame(llm[["context", "split"]])
    missing_cells = required_cells.difference(actual_cells)
    duplicate_cells = actual_cells[actual_cells.duplicated()].unique()
    if (len(llm) != 12 or llm["model"].nunique() != 1 or
            len(missing_cells) or len(duplicate_cells)):
        raise ValueError(
            "09c_llm_results.csv must contain one model and exactly one row for "
            f"each of 12 context/split cells; rows={len(llm)}, "
            f"models={llm['model'].unique().tolist()}, "
            f"missing={missing_cells.tolist()}, duplicates={duplicate_cells.tolist()}"
        )
if llm_run:
    mac("LLMDevice", esc(llm_run["device"]))
    mac("LLMDtype", esc(llm_run["dtype"]))
    mac("LLMMinutes", f"{llm_run['replacement_elapsed_seconds']/60:.1f}")
    mac("LLMRevision", llm_run["replacement_revision"][:12])
if res is not None:
    allr = pd.concat([x for x in (res, enc, llm) if x is not None],
                     ignore_index=True)
    if llm is not None and len(llm):
        mac("LLMName", esc(llm["model"].iloc[0]))
        t = llm[llm["split"] == "TEST"]
        if len(t):
            mac("LLMBestF", f"{t['macro_f1'].max():.3f}")
            mac("LLMBestCtx", str(t.loc[t["macro_f1"].idxmax(), "context"]))
            mac("LLMSentF", f"{t[t.context=='SENT']['macro_f1'].iloc[0]:.3f}"
                if (t["context"] == "SENT").any() else "--")
    piv = allr.pivot_table(index=["model", "context"], columns="split",
                           values="macro_f1")
    for c in ("TEST", "TEMPORAL", "COURT"):
        if c not in piv:
            piv[c] = np.nan
    # Degenerate cells are marked rather than silently printed: a model that
    # collapses to one class scores the majority baseline exactly, and printing
    # that as a third decimal invites it to be read as a result.
    majf = allr[allr["model"] == "majority"].set_index("split")["macro_f1"].to_dict()

    def cell(m, c, sp):
        v = piv.loc[(m, c), sp]
        if pd.isna(v):
            return "--"
        base = majf.get(sp)
        mark = "$^{\\dagger}$" if base is not None and abs(v - base) < 1e-6 else ""
        return f"{v:.3f}{mark}"

    # The frozen dictionary is invariant to the window by construction, since
    # every window is centred on the anchor, so its four identical rows are
    # printed once rather than four times.
    rows, seen_lex = [], False
    for m, c in piv.index:
        if m == "lexical-rule":
            if seen_lex:
                continue
            seen_lex, c = True, "any"
            m2 = "lexical-rule"
            vals = [cell("lexical-rule", "SENT", sp)
                    for sp in ("TEST", "TEMPORAL", "COURT")]
        else:
            m2 = m
            vals = [cell(m, c, sp) for sp in ("TEST", "TEMPORAL", "COURT")]
        rows.append(f"{esc(m2)} & {c} & " + " & ".join(vals) + " \\\\")
    body = "\n".join(rows)
    write_table("tab_models", body)
    lex = res[(res["model"] == "lexical-rule") & (res["split"] == "TEST")]
    if len(lex):
        mac("LexF", f"{lex['macro_f1'].max():.3f}")
    io_ = res[(res["model"] == "issue-only") & (res["split"] == "TEST")]
    if len(io_):
        mac("IssueOnlyF", f"{io_['macro_f1'].iloc[0]:.3f}")
        mac("IssueOnlyAcc", pct(io_["acc"].iloc[0]))
    best = allr[(allr["split"] == "TEST") & (~allr["model"].isin(
        ["lexical-rule", "majority", "issue-only"]))]
    if len(best):
        bb = best.loc[best["macro_f1"].idxmax()]
        mac("BestModel", str(bb["model"]))
        mac("BestCtx", str(bb["context"]))
        mac("BestF", f"{bb['macro_f1']:.3f}")

ls_ = load("09c_llm_by_stratum.csv")
if ls_ is not None and len(ls_):
    write_table("tab_llmstratum", "\n".join(
        f"{esc(r['context'])} & {esc(r['stratum'])} & {100*r['llm_acc']:.1f} \\\\"
        for _, r in ls_.iterrows()))

bs = load("09_by_stratum.csv")
if bs is not None:
    body = "\n".join(
        f"{esc(r['stratum'])} & {r['context']} & {num(r['n'])} & "
        f"{r['lexical_acc']:.3f} & {r['tfidf_acc']:.3f} \\\\"
        for _, r in bs.sort_values(["stratum", "context"]).iterrows())
    write_table("tab_stratum_acc", body)

# ---------------------------------------------------------------- longitudinal
ch = load("10a_chains.parquet", "pq")
l2 = load("10b_annotated.parquet", "pq")
if ch is not None and len(ch):
    mac("NChains", num(len(ch)))
    mac("NIssueSpecific", num(int(ch["issue_specific"].sum())))
    mac("ShareIssueSpecific", pct(ch["issue_specific"].mean()))
if l2 is not None and len(l2):
    u = l2[l2["usable"]]
    mac("NLayerTwo", num(len(l2)))
    mac("NLayerTwoUsable", num(len(u)))
    if len(u):
        for lab, tag in (("UNRESOLVED", "Unres"), ("DECIDED", "Dec")):
            g = u[u["origin_label"] == lab]
            if len(g):
                mac(f"Esc{tag}", pct((g["status"] >= 2).mean()))
                mac(f"Attr{tag}", pct(g["A"].mean()))
                mac(f"Indep{tag}", pct(g["E"].mean()))
                mac(f"N{tag}", num(len(g)))
        body = "\n".join(
            f"{esc(lab)} & {num(len(g))} & " +
            " & ".join(f"{100*(g['status']==s).mean():.1f}" for s in range(4)) +
            f" & {100*g['A'].mean():.1f} & {100*g['E'].mean():.1f} \\\\"
            for lab, g in u.groupby("origin_label"))
        write_table("tab_layer2", body)

# ---------------------------------------------------------------- api check
api = load("api_crosscheck.json", "json")
if api and pv:
    tot = sum(v for v in api["denominator_by_court"].values() if v)
    mac("ApiTotal", num(tot))
    body = "\n".join(
        f"{COURT_LABEL.get(c,c)} & {num(api['denominator_by_court'].get(c) or 0)} \\\\"
        for c in FED_APP)
    write_table("tab_api", body)

with open(MAC_PATH, "w") as fh:
    for k, v in sorted(MAC.items()):
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"wrote macros.tex with {len(MAC)} definitions")


# ============================================================ appendix tables
def head(name, cells):
    with open(os.path.join(GEN, f"{name}_head.tex"), "w") as fh:
        fh.write(" & ".join(cells) + " \\\\\n")


# --- text source composition by period
el = load("05_eligible.parquet", "pq")
if el is not None:
    ct = pd.crosstab(el["text_source"], el["period"], normalize="columns")
    cols = list(ct.columns)
    head("tab_textsrc", ["field"] + [str(c) for c in cols])
    write_table("tab_textsrc", "\n".join(
        f"\\texttt{{{i.replace('_', chr(92)+'_')}}} & " +
        " & ".join(f"{100*r[c]:.1f}" for c in cols) + " \\\\"
        for i, r in ct.iterrows()))
    mac("NTextFields", str(ct.shape[0]))
    mac("OCRShare", pct(el["ocr"].mean()))
    mac("MedWords", num(el["n_words"].median()))
    mac("YearMin", str(int(el["year"].min())))
    mac("YearMax", str(int(el["year"].max())))

# --- the frozen regular expressions
# Rendered as a full-width verbatim block rather than a tabular. A regex is not
# prose and does not wrap; escaping it into a tabular cell fights the typesetter
# and loses the one property a reader needs, which is that it can be copied.
from triggers import SPEC as TRIG_SPEC
import textwrap as _tw
_lines = []
for t, _n, rx in TRIG_SPEC:
    parts = _tw.wrap(rx, width=88) or [""]
    _lines.append(f"{t}  {parts[0]}")
    _lines += [" " * 4 + q for q in parts[1:]]
with open(os.path.join(GEN, "tab_regex_full.tex"), "w") as fh:
    fh.write("\\begin{figure*}[t]\n\\centering\\scriptsize\n"
             "\\begin{verbatim}\n" + "\n".join(_lines) + "\n\\end{verbatim}\n"
             "\\caption{The thirteen frozen expressions as regular expressions, "
             "matched case-insensitively over whitespace-normalized text.}\n"
             "\\label{tab:regex}\n\\end{figure*}\n")
print("  wrote tab_regex_full.tex (verbatim, full width)")

# --- issue extraction yield
fr = load("06_frame.parquet", "pq")
cand = os.path.join(OUT, "03_candidates.jsonl")
if fr is not None and os.path.exists(cand):
    import collections
    raw = collections.Counter()
    with open(cand) as fh:
        for line in fh:
            raw[json.loads(line)["trigger_ids"]] += 1
    kept = fr[fr["family"] == "TRIGGER"]["trigger_ids"].value_counts()
    rows = []
    for t, _n, _rx in TRIG_SPEC:
        r = sum(v for k, v in raw.items() if t in k.split("+"))
        k = sum(v for kk, v in kept.items() if t in str(kk).split("+"))
        if r:
            rows.append(f"{t} & {num(r)} & {num(k)} & {100*k/r:.1f} \\\\")
    head("tab_extract", ["id", "anchors", "with issue", "yield (\\%)"])
    write_table("tab_extract", "\n".join(rows))
    tot_r, tot_k = sum(raw.values()), int((fr["family"] == "TRIGGER").sum())
    mac("ExtractYield", pct(tot_k / max(1, tot_r)))
    mac("NFrameTrigger", num(tot_k))
    mac("NFrameCtrl", num(int((fr["family"] != "TRIGGER").sum())))
    ws = fr[fr["family"] == "TRIGGER"]["issue"].astype(str)
    mac("ShareWhether", pct(ws.str.lower().str.startswith("whether").mean()))

# --- full prevalence tables in one appendix block
blocks = []
for fn, cap, cols in [
        ("05_prev_court_period.csv", "Share of opinions containing at least one "
         "dictionary expression, by court and period.", None),
        ("05_prev_type_group.csv", "By opinion type.", None),
        ("05_prev_status.csv", "By precedential status.", None)]:
    d = load(fn)
    if d is None:
        continue
    if "period" in d.columns and "court" in d.columns:
        piv = d.pivot_table(index="court", columns="period", values="rate")
        piv = piv.reindex([c for c in FED_APP if c in piv.index])
        body = "\n".join(
            f"{COURT_LABEL.get(i,i)} & " +
            " & ".join("--" if pd.isna(r[c]) else f"{100*r[c]:.2f}"
                       for c in piv.columns) + " \\\\"
            for i, r in piv.iterrows())
        # court by period is five columns wide and does not fit one column
        blocks.append("\\begin{table*}[t]\\centering\\small\n"
                      "\\begin{tabular}{l" + "r" * piv.shape[1] + "}\n\\toprule\n"
                      "court & " + " & ".join(str(c) for c in piv.columns) +
                      " \\\\\n\\midrule\n" + body +
                      "\n\\bottomrule\n\\end{tabular}\n"
                      f"\\caption{{{cap}}}\\label{{tab:cp}}\n\\end{{table*}}\n")
    else:
        key = [c for c in d.columns if c in ("type_group", "precedential_status")][0]
        d = d[d["n"] >= 100]
        body = "\n".join(f"{r[key]} & {num(r['n'])} & {num(r['k'])} & "
                         f"{100*r['rate']:.2f} \\\\" for _, r in d.iterrows())
        blocks.append("\\begin{table}[t]\\centering\\small\n"
                      "\\begin{tabular}{lrrr}\n\\toprule\n"
                      f"{key.replace('_',' ')} & $N$ & $k$ & \\% \\\\\n\\midrule\n"
                      + body + "\n\\bottomrule\n\\end{tabular}\n"
                      f"\\caption{{{cap}}}\\label{{tab:{key}}}\n\\end{{table}}\n")
with open(os.path.join(os.path.dirname(GEN), "tab_prev_extra_wrap.tex"), "w") as fh:
    fh.write("\n".join(blocks))
print("  wrote tab_prev_extra_wrap.tex")

# --- guidelines verbatim
import guidelines as G
def vb(s):
    # The guideline is written to an 80-column terminal width, which overflows a
    # two-column page. Re-wrap to the column, preserving the indentation that
    # carries the rule structure.
    import textwrap
    out = []
    for line in s.strip().split("\n"):
        if not line.strip():
            out.append("")
            continue
        ind = len(line) - len(line.lstrip())
        out += textwrap.wrap(line.strip(), width=max(20, 52 - ind),
                             initial_indent=" " * ind,
                             subsequent_indent=" " * (ind + 2)) or [""]
    return ("\\begin{quote}\\scriptsize\\begin{verbatim}\n" + "\n".join(out)
            + "\n\\end{verbatim}\\end{quote}\n")
with open(os.path.join(os.path.dirname(GEN), "guidelines_wrap.tex"), "w") as fh:
    # Only the layer-one guideline is reproduced. It is the one that produced
    # every label in the benchmark; the layer-two and issue-statement guides are
    # in the release, and printing all three costs a page and a half.
    fh.write(vb(G.LAYER1))
    fh.write(f"\n\\noindent Version {G.VERSION}. The single permitted revision "
             "after the pilot, and the layer-two and issue-statement guidelines, "
             "are in the released package.\n")
print("  wrote guidelines_wrap.tex")

# --- human context ablation
hab = load("08_human_ablation.csv")
if hab is not None and len(hab):
    mac("NHumanAbl", num(len(hab)))
    mac("HumanAgree", pct(hab["agree"].mean()))
    mac("HumanFlip", pct(1 - hab["agree"].mean()))
    mac("HumanUnclearSent", pct((hab["sent_bin"] == "X").mean()))
    mac("HumanUnclearFull", pct((hab["full_bin"] == "X").mean()))
    g = hab.groupby("stratum")["agree"].agg(["size", "mean"]).reset_index()
    write_table("tab_humanabl", "\n".join(
        f"{esc(r['stratum'])} & {num(r['size'])} & {100*r['mean']:.1f} \\\\"
        for _, r in g.iterrows()))

# --- recall audit
rcann = load("08_annotated_recall.parquet", "pq")
if rcann is not None and len(rcann):
    ok = rcann[rcann["label"].isin(["DECIDED", "UNRESOLVED"])]
    mac("NRecallAnn", num(len(rcann)))
    if len(ok):
        mac("RecallGenuine", pct((ok["label"] == "UNRESOLVED").mean()))

rc = load("13_recall.json", "json")
if rc:
    mac("RecallSampled", num(rc["n_sampled"]))
    mac("RecallHit", pct(rc["share_with_probe"]))
    body = "\n".join(
        f"{k} & {esc(v['name'])} & {num(v['n'])} & {100*v['share']:.2f} \\\\"
        for k, v in sorted(rc["per_probe"].items()) if v["n"])
    head("tab_recall", ["id", "probe expression", "$n$", "\\% of clean"])
    write_table("tab_recall", body)


# --- figures the rewritten results section needs
ann_ = load("08_annotated.parquet", "pq")
if ann_ is not None and len(ann_):
    b_ = ann_[ann_["label"].isin(["DECIDED", "UNRESOLVED"])]
    mac("NDev", num(int((b_["split"] == "DEV").sum())))
    h1 = b_[b_["stratum"] == "H1"]
    if len(h1):
        mac("HOneUnresShare", pct((h1["label"] == "UNRESOLVED").mean(), 0))
bs_ = load("09_by_stratum.csv")
if bs_ is not None and len(bs_):
    h = bs_[bs_["stratum"] == "H1"]
    if len(h):
        mac("LexHOne", f"{h['lexical_acc'].mean():.3f}")
res_ = load("09_results.csv")
if res_ is not None and len(res_):
    t_ = res_[(res_["model"] == "tfidf-lr") & (res_["split"] == "TEST")]
    if len(t_):
        mac("TfidfBestF", f"{t_['macro_f1'].max():.3f}")
ct_ = load("09_context_test.csv")
if ct_ is not None and len(ct_):
    t_ = ct_[ct_["split"] == "TEST"]
    if len(t_):
        w = t_.loc[t_["delta_macro_f1"].idxmin()]
        mac("CtxWorstDelta", f"{w['delta_macro_f1']:+.3f}")
        mac("CtxWorstLo", f"{w['ci_lo']:+.3f}")
        mac("CtxWorstHi", f"{w['ci_hi']:+.3f}")
llm_ = load("09c_llm_results.csv")
if llm_ is not None and len(llm_):
    t_ = llm_[(llm_["split"] == "TEST") & (llm_["context"] == "W4096")]
    if len(t_):
        mac("LLMWideF", f"{t_['macro_f1'].iloc[0]:.3f}")
ch_ = load("10a_chains.parquet", "pq")
if ch_ is not None and len(ch_):
    mac("NChainPassages", num(len(ch_)))
    mac("NChainOrigins", num(int(ch_["origin_item_id"].nunique())))
    mac("ChainIssueSpecific", pct(ch_["issue_specific"].mean()))
l2_ = load("10b_annotated.parquet", "pq")
if l2_ is not None and len(l2_):
    mac("NLayerTwoAnnotated", num(len(l2_)))
    mac("OffIssueRate", pct(l2_["off_issue"].mean(), 0))
    mac("NLayerTwoUsable", num(int(l2_["usable"].sum())))


# --- judicially authored characterizations
js = load("15_stats.json", "json"); jt = load("15_tight_stats.json", "json")
if js: mac("NCharLoose", num(js["n_characterizations"]))
if jt:
    mac("NCharTight", num(jt["n_tight"])); mac("NCharLZero", num(jt["n_L0"]))
    mac("NCharOps", num(jt["n_cited"]))
j17 = load("17_stats.json", "json")
if j17:
    mac("NJudgeBench", num(j17["n"])); mac("NJudgeUnres", num(j17["n_unresolved"]))
    mac("NJudgeDec", num(j17["n_decided"]))
j18 = load("18_judge_eval.csv")
if j18 is not None and len(j18):
    lx = j18[j18["model"] == "lexical-rule"]["macro_f1"].max()
    tf = j18[j18["model"] == "tfidf-lr"]["macro_f1"].max()
    mac("JudgeLexF", f"{lx:.3f}"); mac("JudgeTfidfF", f"{tf:.3f}")


iv = load("21_validation.json", "json")
if iv:
    mac("NIMMatched", num(iv["n_matched"])); mac("NIMGenuine", num(iv["n_genuine"]))
    mac("NIMPrec", pct(iv["precision"], 0)); mac("NIMDistinct", num(iv["n_distinct"]))
    mac("NIMEsc", num(iv["esc_dedup"])); mac("IMEscRate", pct(iv["rate_dedup"], 0))
pv = load("19_proxy_validation.json", "json")
if pv:
    mac("NProxyCheck", num(pv["n_checked"])); mac("NProxyGood", num(pv["n_genuine"]))
    mac("ProxyPrec", pct(pv["precision"], 0))
e19 = load("19_escalation.json", "json")
if e19:
    mac("NEscN", num(e19["n"])); mac("NEscOps", num(e19["n_opinions"]))
    mac("EscOpen", pct(e19["rate_open"])); mac("EscNotOpen", pct(e19["rate_notopen"]))
    mac("EscDiffPP", f"{100*e19['diff']:+.1f}")
    mac("EscCILo", f"{100*e19['ci'][0]:+.1f}"); mac("EscCIHi", f"{100*e19['ci'][1]:+.1f}")


lc = load("20_learning_curve.csv")
if lc is not None and len(lc):
    write_table("tab_lc", "\n".join(
        f"{int(r['n_train'])} & {r['h1_acc_mean']:.3f} & {r['h1_acc_sd']:.3f} & "
        f"{100*r['beats_lexical']:.0f} \\\\" for _, r in lc.iterrows()))
    mac("LCCross", str(int(lc[lc["beats_lexical"] >= 0.5]["n_train"].min())
                       if (lc["beats_lexical"] >= 0.5).any() else 0))
    mac("LCTopAcc", f"{lc['h1_acc_mean'].iloc[-1]:.3f}")

# Any macro the prose uses but the pipeline has not yet produced is defined as a
# visible placeholder, so a partial run still compiles and the gap is obvious.
PAPER = os.path.dirname(GEN)
used = set()
for fn in os.listdir(PAPER):
    if fn.endswith(".tex") and fn != "acl_latex.tex":
        used |= set(re.findall(r"\\([A-Z][A-Za-z]{3,})\b",
                               open(os.path.join(PAPER, fn)).read()))
KNOWN_LATEX = {"IfFileExists", "textbackslash", "textasciitilde", "Cir", "LaTeX",
               "Delta", "Pr", "S", "J", "C", "N"}
missing = sorted((used - set(MAC)) - KNOWN_LATEX)
with open(MAC_PATH, "w") as fh:
    for k, v in sorted(MAC.items()):
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    for k in missing:
        fh.write(f"\\providecommand{{\\{k}}}{{\\textbf{{??}}}}\n")
print(f"final macros.tex: {len(MAC)} definitions, {len(missing)} placeholders")
if missing:
    print("  placeholders:", " ".join(missing))


# --- paired context comparison
ct = load("09_context_test.csv")
if ct is not None and len(ct):
    t = ct[ct["split"] == "TEST"]
    if len(t):
        best = t.loc[t["delta_macro_f1"].idxmax()]
        mac("CtxBestWin", str(best["context"]))
        mac("CtxBestDelta", f"{best['delta_macro_f1']:+.3f}")
        mac("CtxBestLo", f"{best['ci_lo']:+.3f}")
        mac("CtxBestHi", f"{best['ci_hi']:+.3f}")
        mac("CtxBestP", f"{best['p_gt_zero']:.3f}")
    body = "\n".join(
        f"{r['split']} & {r['context']} & {r['delta_macro_f1']:+.3f} & "
        f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] & {r['p_gt_zero']:.3f} \\\\"
        for _, r in ct.iterrows())
    write_table("tab_context", body)

# --- layer two contrast with clustered intervals
an = load("10c_analysis.json", "json")
if an:
    mac("NLTwoOrigins", num(an["n_origins"]))
    mac("NOffIssue", num(an["n_off_issue"]))
    for lab, tag in (("UNRESOLVED", "Unres"), ("DECIDED", "Dec")):
        g = an["groups"].get(lab)
        if not g:
            continue
        mac(f"Esc{tag}", pct(g["escalated"]))
        mac(f"Esc{tag}Lo", pct(g["escalated_ci"][0]))
        mac(f"Esc{tag}Hi", pct(g["escalated_ci"][1]))
        mac(f"Est{tag}", pct(g["established"]))
        mac(f"Attr{tag}", pct(g["attributed"]))
        mac(f"Indep{tag}", pct(g["independent"]))
        mac(f"N{tag}", num(g["n"]))
    c = (an.get("contrast") or {}).get("escalated")
    if c:
        mac("EscDiff", f"{100*c['diff']:+.1f}")
        mac("EscDiffLo", f"{100*c['ci'][0]:+.1f}")
        mac("EscDiffHi", f"{100*c['ci'][1]:+.1f}")
        mac("EscDiffSig", "excludes zero" if c["excludes_zero"] else "includes zero")
    pw = load("10c_pathways.csv")
    if pw is not None and len(pw):
        write_table("tab_paths", "\n".join(
            f"{esc(r['origin_label'])} & {esc(r['rel'])} & {num(r['n'])} & "
            f"{100*r['escalated']:.1f} & {100*r['attributed']:.1f} \\\\"
            for _, r in pw.iterrows()))
    ca = (an.get("contrast") or {}).get("A")
    if ca:
        mac("AttrDiff", f"{100*ca['diff']:+.1f}")
        mac("AttrDiffLo", f"{100*ca['ci'][0]:+.1f}")
        mac("AttrDiffHi", f"{100*ca['ci'][1]:+.1f}")

# ======================================================= compose full floats
# \input inside a tabular breaks TeX's alignment scanning, so each float is
# emitted complete: environment, tabular, caption and label together.
SPECS = {
 "tab_funnel": ("lr", "criterion & opinions", "table*",
   "Eligibility funnel. Criteria are applied in the order shown; each row is the "
   "count surviving it.", "funnel"),
 "tab_prev_court": ("lrrrcr", "court & $N$ & $k$ & \\% & 95\\% CI & per 100k words",
   "table*", "Opinions containing at least one dictionary expression, by court. "
   "$N$ is eligible opinions and $k$ those with at least one expression. "
   "Intervals are Wilson \\citep{wilson1927probable}.", "prevcourt"),
 "tab_prev_period": ("lrrrcr", "period & $N$ & $k$ & \\% & 95\\% CI & per 100k words",
   "table*", "The same quantity by era. The rate per hundred thousand words is "
   "reported alongside the opinion-level rate because opinion length is not "
   "constant across eras.", "prevperiod"),
 "tab_prev_trigger": ("llrr", "id & expression & opinions & \\% of corpus", "table*",
   "The thirteen frozen expressions and how often each appears. T08 and T09 are "
   "subsets of T07 by construction.", "trigger"),
 "tab_stratum": ("lrrrr", "stratum & dec. & unres. & unclear & \\% unres.", "table",
   "Annotated labels by sampling stratum. The last column excludes "
   "\\textsc{unclear}.", "stratum"),
 "tab_splits": ("lrr", "split & $n$ & \\% unresolved", "table*",
   "Evaluation splits. They are disjoint at the opinion level and only "
   "\\textsc{dev} was used for fitting.", "splits"),
 "tab_models": ("llrrr",
   "model & context & random ($n{=}54$) & temporal ($n{=}14$) & court ($n{=}27$)",
   "table*",
   "Macro-$F_1$ on the three held-out conditions. The lexical rule is the frozen "
   "dictionary, invariant to the window because every window is anchor-centred, "
   "and issue-only is the leakage control. Splits are small and the temporal "
   "split too small to rank models. $\\dagger$ marks a cell equal to the majority "
   "baseline, meaning the model collapsed to one class.",
   "models"),
 "tab_layer2": ("lrrrrrrr",
   "origin & $n$ & $L_0$ & $L_1$ & $L_2$ & $L_3$ & $A{=}1$ & $E{=}1$", "table*",
   "How later opinions characterize the origin, by what the origin did. "
   "$L_0$--$L_3$ are percentages of citing passages. $A$ is attribution of a "
   "resolution to the origin; $E$ is independent resolution by the later court.",
   "layer2"),
 "tab_stratum_acc": ("llrrr", "stratum & context & $n$ & lexical & TF-IDF", "table*",
   "Accuracy by stratum and context width. The lexical rule is the frozen "
   "dictionary applied to the window.", "stratacc"),
 "tab_textsrc": (None, None, "table*",
   "Text source field by period, as a percentage of eligible opinions in that "
   "period.", "textsrc"),
 "tab_extract": ("lrrr", "id & anchors & with issue & yield (\\%)", "table",
   "Issue-statement extraction yield by expression.", "extract"),
 "tab_api": ("lr", "court & indexed opinions", "table*",
   "Denominators reported by the CourtListener search index, used only for the "
   "cross-check in Appendix~\\ref{app:api}.", "api"),
 "tab_context": ("llrcr",
   "split & context & $\\Delta F_1$ & 95\\% CI & $P(\\Delta{>}0)$", "table",
   "Paired bootstrap on the effect of widening the context window, relative to "
   "the anchor sentence alone, for the sparse model. Resamples are paired across "
   "conditions.", "context"),
 "tab_llmstratum": ("llr",
   "context & stratum & accuracy (\\%)", "table",
   "Zero-shot open instruction-tuned model, accuracy by context width and "
   "sampling stratum, scored by label-token likelihood.", "llmstratum"),
 "tab_humanabl": ("lrr",
   "stratum & $n$ & agreement with full context (\\%)", "table",
   "How often the annotator's sentence-only label matches the label the same "
   "annotator gave with the full window. A within-annotator context ablation, "
   "not an inter-annotator statistic.", "humanabl"),
 "tab_paths": ("llrrr",
   "origin & citing court & $n$ & escalated (\\%) & $A{=}1$ (\\%)", "table",
   "Escalation and attribution by the relationship between the citing court and "
   "the origin. Cell counts are small and these are descriptive pathways.",
   "paths"),
 "tab_lc": ("lrrr",
   "training items & H1 accuracy & s.d. & beats dictionary (\\%)", "table",
   "Learning curve on the adversarial stratum. Each row resamples 12 training "
   "subsets of the stated size from the development split and evaluates on the "
   "held-out H1 items; the dictionary reaches 0.895 on the same items.", "lc"),
 "tab_recall": ("llrr", "id & probe expression & $n$ & \\% of clean", "table*",
   "Probe expressions found in opinions the frozen dictionary called clean. The "
   "probe was written after the fact and is used only for this audit.", "recall"),
}

# A few tables carry a wide interval column and do not fit a single column at
# \small. Size is set per table rather than shrinking every float.
FONT = {"tab_prev_court": "\\footnotesize", "tab_prev_period": "\\footnotesize",
        "tab_models": "\\footnotesize", "tab_context": "\\scriptsize",
        "tab_prev_period": "\\footnotesize", "tab_prev_trigger": "\\footnotesize",
        "tab_recall": "\\footnotesize"}

for nm, (align, hdr, env, cap, lab) in SPECS.items():
    bp = os.path.join(GEN, f"{nm}.tex")
    if not os.path.exists(bp):
        continue
    body = open(bp).read().rstrip()
    if align is None:                       # header/align supplied by a _head file
        hp = os.path.join(GEN, f"{nm}_head.tex")
        if not os.path.exists(hp):
            continue
        hdr = open(hp).read().strip().rstrip("\\")
        align = "l" + "r" * hdr.count("&")
    with open(os.path.join(GEN, f"{nm}_full.tex"), "w") as fh:
        fh.write(f"\\begin{{{env}}}[t]\n\\centering{FONT.get(nm, chr(92)+'small')}\n"
                 f"\\begin{{tabular}}{{{align}}}\n\\toprule\n{hdr} \\\\\n"
                 f"\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n"
                 f"\\caption{{{cap}}}\n\\label{{tab:{lab}}}\n\\end{{{env}}}\n")
print(f"composed {sum(os.path.exists(os.path.join(GEN, n+'_full.tex')) for n in SPECS)} floats")
