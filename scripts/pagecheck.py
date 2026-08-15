"""Report where the content ends, so the eight-page limit can be hit exactly."""
import re, sys
from pypdf import PdfReader

PDF = sys.argv[1] if len(sys.argv) > 1 else "paper/build/main.pdf"
MARKS = ["Introduction", "Related Work", "Task Definition", "Corpus and Population",
         "Measuring Non-Resolution", "Building the Benchmark", "Can Models Tell",
         "What Happens Next", "Conclusion", "Limitations", "Ethics Statement",
         "References", "Excluded Federal Courts"]

r = PdfReader(PDF)
page_of = {}
for i, pg in enumerate(r.pages):
    t = pg.extract_text() or ""
    for m in MARKS:
        if m not in page_of and re.search(r"(?m)^\s*\d*\s*" + re.escape(m), t):
            page_of[m] = i + 1

for m in MARKS:
    if m in page_of:
        print(f"  p{page_of[m]:>2}  {m}")
print(f"\ntotal pages: {len(r.pages)}")

lim = page_of.get("Limitations")
if lim is None:
    print("VERDICT: could not locate Limitations")
elif lim == 9:
    print("VERDICT: OK - content fills 8 pages, back matter starts on page 9")
elif lim < 9:
    print(f"VERDICT: SHORT by {9-lim} page(s) - content ends too early, "
          f"Limitations is on p{lim}")
else:
    print(f"VERDICT: OVER by {lim-9} page(s) - content exceeds 8 pages, "
          f"Limitations is on p{lim}")
