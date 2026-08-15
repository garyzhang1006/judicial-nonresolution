"""Step 23: pair two narrow supplementary tables into one full-width float.

Both tables are four rows of three columns, so each alone leaves most of a
text-width float empty and the pair pushes the appendix onto an extra page.
Reading the already-generated bodies keeps the numbers derived rather than
transcribed.
"""
import os, re, sys

GEN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "paper", "generated")
PAIR = [("tab_splits", 0.46), ("tab_humanabl", 0.50)]


def part(name):
    with open(os.path.join(GEN, name + "_full.tex")) as fh:
        s = fh.read()
    tab = re.search(r"\\begin\{tabular\}.*?\\end\{tabular\}", s, re.S)
    cap = re.search(r"\\caption\{(.*?)\}\n\\label\{(.*?)\}", s, re.S)
    if not (tab and cap):
        sys.exit(f"could not parse {name}")
    return tab.group(0), cap.group(1), cap.group(2)


def main():
    cells = []
    for name, w in PAIR:
        tab, cap, lab = part(name)
        cells.append(
            "\\begin{minipage}[t]{%.2f\\textwidth}\n\\centering\\small\n%s\n"
            "\\captionof{table}{%s}\n\\label{%s}\n\\end{minipage}" % (w, tab, cap, lab))
    out = ("\\begin{table*}[t]\n\\centering\n" + "\\hfill\n".join(cells) +
           "\n\\end{table*}\n")
    with open(os.path.join(GEN, "tab_tailpair_full.tex"), "w") as fh:
        fh.write(out)
    print("wrote tab_tailpair_full.tex (%d cells)" % len(cells))


if __name__ == "__main__":
    main()
