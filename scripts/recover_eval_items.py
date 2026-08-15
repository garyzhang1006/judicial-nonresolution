"""Recover zero-shot evaluation inputs from the frozen CourtListener snapshot.

The release archive intentionally omits the derived item parquet and text shards.
Archived zero-shot predictions retain every held-out item ID, label, stratum, and
split. This script streams the exact opinions bulk snapshot, keeps only those 186
opinions, deterministically rebuilds their anchors and context windows, and checks
each reconstructed anchor against the released annotation worksheets.
"""
import argparse
import csv
import gzip
import io
import json
import os
import re
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TEXT_FIELDS, best_text
from issues import find_holding_anchors, find_neutral_anchors, issue_clause, sentence_span
from triggers import find_all, merge_overlaps

DEFAULT_BULK_URL = (
    "https://com-courtlistener-storage.s3.us-west-2.amazonaws.com/"
    "bulk-data/opinions-2026-06-30.csv.bz2"
)
WINDOWS = {"SENT": None, "W256": 256, "W1024": 1024, "W4096": 4096}


def clean(value):
    return re.sub(r"\s+", " ", str(value)).strip()


def parse_item_id(item_id):
    match = re.fullmatch(r"([cr]?)(\d+)-(\d+)", str(item_id))
    if not match:
        raise ValueError(f"invalid item_id: {item_id!r}")
    prefix, opinion_id, anchor_index = match.groups()
    family = {"": "TRIGGER", "c": "CTRL_HOLD", "r": "CTRL_RAND"}[prefix]
    return family, int(opinion_id), int(anchor_index)


def target_rows(predictions_path):
    predictions = pd.read_csv(predictions_path)
    required = {"item_id", "stratum", "split", "y", "context"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"{predictions_path} missing columns: {sorted(missing)}")
    counts = predictions.groupby("item_id")["context"].nunique()
    if len(counts) != 186 or not (counts == 4).all():
        raise ValueError(
            f"expected 186 items with four contexts each; got {len(counts)} items, "
            f"context counts {counts.value_counts().to_dict()}"
        )
    metadata = predictions[["item_id", "stratum", "split", "y"]].drop_duplicates()
    if len(metadata) != 186:
        raise ValueError("archived predictions disagree on item metadata")
    return metadata.reset_index(drop=True)


def load_cached_opinions(path, wanted_ids):
    if not path or not os.path.exists(path):
        return {}
    opinions = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            opinion_id = int(record["opinion_id"])
            if opinion_id in wanted_ids:
                opinions[opinion_id] = record["text"]
    return opinions


def stream_target_opinions(url, wanted_ids, cache_path):
    """Stream bzip2 CSV through curl and retain only requested opinion IDs."""
    curl = subprocess.Popen(
        ["curl", "--fail", "--location", "--retry", "8", "--retry-all-errors",
         "--connect-timeout", "30", "--speed-time", "120", "--speed-limit", "1024",
         url],
        stdout=subprocess.PIPE,
    )
    unzip = subprocess.Popen(
        ["bzip2", "-dc"], stdin=curl.stdout, stdout=subprocess.PIPE, bufsize=1 << 22
    )
    curl.stdout.close()
    text_stream = io.TextIOWrapper(
        unzip.stdout, encoding="utf-8", errors="replace", newline=""
    )
    reader = csv.reader(text_stream, escapechar="\\")
    header = next(reader)
    index = {column: position for position, column in enumerate(header)}
    needed_columns = {"id", *TEXT_FIELDS}
    missing = needed_columns - set(index)
    if missing:
        raise KeyError(f"bulk opinions file missing columns: {sorted(missing)}")

    width = len(header)
    opinions = {}
    malformed = 0
    rows_seen = 0
    completed_early = False
    try:
        for row in reader:
            rows_seen += 1
            if rows_seen % 1_000_000 == 0:
                print(
                    f"bulk rows {rows_seen:,}; recovered {len(opinions)}/{len(wanted_ids)}",
                    flush=True,
                )
            if len(row) != width:
                malformed += 1
                continue
            try:
                opinion_id = int(row[index["id"]])
            except ValueError:
                continue
            if opinion_id not in wanted_ids:
                continue
            record = {column: row[index[column]] for column in TEXT_FIELDS}
            text, source = best_text(record)
            if not text:
                raise ValueError(f"opinion {opinion_id} has no usable text")
            opinions[opinion_id] = text
            print(
                f"recovered opinion {opinion_id} from {source}; "
                f"{len(opinions)}/{len(wanted_ids)}",
                flush=True,
            )
            if len(opinions) == len(wanted_ids):
                completed_early = True
                break
    finally:
        text_stream.detach()
        if completed_early:
            unzip.terminate()
            curl.terminate()
        unzip.wait()
        curl.wait()

    missing_ids = sorted(wanted_ids - set(opinions))
    if missing_ids:
        raise RuntimeError(
            f"bulk stream ended without {len(missing_ids)} target opinions: {missing_ids[:20]}"
        )
    print(f"bulk recovery complete after {rows_seen:,} rows; malformed {malformed:,}")

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with gzip.open(cache_path, "wt", encoding="utf-8") as handle:
            for opinion_id in sorted(opinions):
                handle.write(json.dumps(
                    {"opinion_id": opinion_id, "text": opinions[opinion_id]},
                    ensure_ascii=False,
                ) + "\n")
    return opinions


def locate_anchor(item_id, text):
    family, opinion_id, anchor_index = parse_item_id(item_id)
    if family == "TRIGGER":
        anchors = merge_overlaps(find_all(text))
        if anchor_index >= len(anchors):
            raise IndexError(
                f"{item_id}: trigger index {anchor_index} outside {len(anchors)} anchors"
            )
        trigger_ids, start, end, matched = anchors[anchor_index]
        issue = issue_clause(text, start, end)
    elif family == "CTRL_HOLD":
        anchors = find_holding_anchors(text, limit=2)
        if anchor_index >= len(anchors):
            raise IndexError(
                f"{item_id}: holding index {anchor_index} outside {len(anchors)} anchors"
            )
        start, end, matched, issue = anchors[anchor_index]
        trigger_ids = ""
    else:
        anchors = find_neutral_anchors(text, limit=1)
        if anchor_index >= len(anchors):
            raise IndexError(
                f"{item_id}: neutral index {anchor_index} outside {len(anchors)} anchors"
            )
        start, end, matched, issue = anchors[anchor_index]
        trigger_ids = ""
    if not issue:
        raise ValueError(f"{item_id}: anchor produced no issue clause")
    sent_start, sent_end = sentence_span(text, start, end)
    return {
        "family": family,
        "opinion_id": opinion_id,
        "trigger_ids": trigger_ids,
        "matched": matched,
        "char_start": start,
        "char_end": end,
        "sent_start": sent_start,
        "sent_end": sent_end,
        "issue": issue,
        "anchor_text": clean(text[start:end]),
    }


def context_windows(text, anchor):
    start, end = anchor["char_start"], anchor["char_end"]
    windows = {}
    for name, width in WINDOWS.items():
        if width is None:
            segment = text[anchor["sent_start"]:anchor["sent_end"]]
        else:
            segment = text[max(0, start - width):min(len(text), end + width)]
        windows[f"ctx_{name}"] = clean(segment)
    return windows


def worksheet_index(worksheets_dir):
    """Index released worksheet headers by the exact truncated issue and anchor."""
    records = {}
    pattern = os.path.join(worksheets_dir, "batch_")
    files = sorted(
        os.path.join(worksheets_dir, name)
        for name in os.listdir(worksheets_dir)
        if name.startswith("batch_") and name.endswith(".txt")
    )
    if not files:
        raise FileNotFoundError(f"no annotation worksheets under {worksheets_dir}")
    for path in files:
        raw = open(path, encoding="utf-8").read()
        for block in re.split(r"(?=^### )", raw, flags=re.M):
            header = re.search(
                r"^###\s+(\d+)\s+\|\s+(\S+)\s+(\d{4})\s+\|\s+([^|]+?)\s+\|\s+(\S+)",
                block,
                flags=re.M,
            )
            issue = re.search(r"^ISSUE:\s*(.*)$", block, flags=re.M)
            anchor = re.search(r"<<(.*?)>>", block, flags=re.S)
            if not (header and issue and anchor):
                continue
            order_idx, court, year, type_group, stratum = header.groups()
            key = (stratum, clean(issue.group(1)), clean(anchor.group(1)))
            value = {
                "order_idx": int(order_idx),
                "court": court,
                "year": int(year),
                "type_group": type_group.strip(),
            }
            records.setdefault(key, []).append(value)
    return records


def rebuild(metadata, opinions, worksheets_dir):
    worksheet_records = worksheet_index(worksheets_dir)
    rows = []
    for item in metadata.itertuples(index=False):
        _, opinion_id, _ = parse_item_id(item.item_id)
        text = opinions[opinion_id]
        anchor = locate_anchor(item.item_id, text)
        key = (item.stratum, clean(clean(anchor["issue"])[:260]), anchor["anchor_text"])
        worksheet_matches = worksheet_records.get(key, [])
        if len(worksheet_matches) != 1:
            raise ValueError(
                f"{item.item_id}: expected one worksheet match for {key!r}; "
                f"got {len(worksheet_matches)}"
            )
        row = {
            "item_id": item.item_id,
            **anchor,
            "stratum": item.stratum,
            "split": item.split,
            "y": int(item.y),
            "label": "UNRESOLVED" if int(item.y) else "DECIDED",
            **worksheet_matches[0],
            **context_windows(text, anchor),
        }
        row.pop("anchor_text")
        rows.append(row)
    rebuilt = pd.DataFrame(rows)
    if len(rebuilt) != 186 or rebuilt["item_id"].nunique() != 186:
        raise AssertionError("reconstruction did not produce 186 unique items")
    expected = {"TEST": 104, "TEMPORAL": 36, "COURT": 46}
    actual = rebuilt["split"].value_counts().to_dict()
    if actual != expected:
        raise AssertionError(f"split counts changed: expected {expected}, got {actual}")
    return rebuilt


def validate_released_error_passages(rebuilt, opinions, error_cases_path):
    """Require exact agreement with every released qualitative passage."""
    if not error_cases_path or not os.path.exists(error_cases_path):
        raise FileNotFoundError(f"released error cases not found: {error_cases_path}")
    released = pd.read_csv(error_cases_path)
    by_id = rebuilt.set_index("item_id")
    checked = set()
    for record in released.itertuples(index=False):
        if record.item_id not in by_id.index:
            raise ValueError(f"released error case missing from reconstruction: {record.item_id}")
        row = by_id.loc[record.item_id]
        text = opinions[int(row["opinion_id"])]
        start, end = int(row["char_start"]), int(row["char_end"])
        passage = clean(
            text[max(0, start - 260):start]
            + " [[" + text[start:end] + "]] "
            + text[end:min(len(text), end + 300)]
        )
        if passage != str(record.passage):
            raise ValueError(
                f"{record.item_id}: reconstructed passage differs from released error case"
            )
        if clean(row["issue"])[:200] != str(record.issue):
            raise ValueError(
                f"{record.item_id}: reconstructed issue differs from released error case"
            )
        checked.add(record.item_id)
    print(
        f"released error passage validation passed: {len(released)} rows, "
        f"{len(checked)} unique items"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="out/09c_llm_preds.qwen.bak.csv")
    parser.add_argument("--worksheets", default="out/worksheets")
    parser.add_argument("--error-cases", default="out/14_error_cases.csv")
    parser.add_argument("--bulk-url", default=DEFAULT_BULK_URL)
    parser.add_argument("--opinion-cache", default="out/recovered_eval_opinions.jsonl.gz")
    parser.add_argument("--output", default="out/09_items_with_ctx.parquet")
    args = parser.parse_args()

    metadata = target_rows(args.predictions)
    wanted_ids = {
        parse_item_id(item_id)[1] for item_id in metadata["item_id"].tolist()
    }
    opinions = load_cached_opinions(args.opinion_cache, wanted_ids)
    if set(opinions) != wanted_ids:
        if opinions:
            print("partial opinion cache ignored; restarting exact bulk stream")
        opinions = stream_target_opinions(args.bulk_url, wanted_ids, args.opinion_cache)
    rebuilt = rebuild(metadata, opinions, args.worksheets)
    validate_released_error_passages(rebuilt, opinions, args.error_cases)
    rebuilt.to_parquet(args.output, index=False)
    print(f"wrote {args.output}: {len(rebuilt)} rows, {len(rebuilt.columns)} columns")
    print(rebuilt.groupby("split")["y"].agg(["size", "sum"]).to_string())


if __name__ == "__main__":
    main()
