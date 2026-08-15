"""Random access to opinion text via the byte-offset index built in step 3."""
import json, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT

TEXTDIR = os.path.join(OUT, "text")


class TextStore:
    def __init__(self):
        idx = pd.read_parquet(os.path.join(OUT, "03_text_index.parquet"))
        self.idx = {int(o): (int(s), int(f))
                    for o, s, f in zip(idx["opinion_id"], idx["shard"], idx["offset"])}
        self.fh = {}

    def get(self, opinion_id):
        rec = self.idx.get(int(opinion_id))
        if rec is None:
            return None
        shard, off = rec
        fh = self.fh.get(shard)
        if fh is None:
            fh = self.fh[shard] = open(os.path.join(TEXTDIR, f"s{shard:02d}.jsonl"),
                                       "r", encoding="utf-8")
        fh.seek(off)
        return json.loads(fh.readline())["text"]

    def close(self):
        for fh in self.fh.values():
            fh.close()
        self.fh = {}
