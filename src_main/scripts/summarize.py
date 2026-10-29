"""Summarize every finished run for a dataset in one table.

Reads results/<dataset>/*.meta.json (written by every run script), prints a
table and writes it to results/<dataset>/summary.md and summary.csv.
Smoke runs (any name containing "smoke") are skipped unless --include-smoke.
No model is loaded, so this takes about a second.

Examples:
  python src_main/scripts/summarize.py --dataset imagenet_a
  python src_main/scripts/summarize.py                     # every dataset under results/
  python src_main/scripts/summarize.py --dataset imagenet_a --include-smoke
"""

import argparse
import csv
import json
import os
from pathlib import Path

from tta.registry import REGISTRY

RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", "results"))

# Which run reproduces which published number (registry `targets`, ViT-B/16).
# Only exact reproductions are compared: e.g. 1-view TDA has no published
# counterpart on ImageNet-A, and the published CLIP number is the 80-prompt one.
PUBLISHED_RUN = {"clip_ens80": "clip", "tda": "tda", "tpt": "tpt", "zero": "zero"}
# For the pilot the registry targets are our own runs with the default prompts.
PUBLISHED_RUN_PILOT = {"clip": "clip", "tda": "tda", "tpt": "tpt"}

# Display order; anything not listed goes after, alphabetically.
ORDER = ["clip_photo", "clip", "clip_ens80", "tda_1view", "tda", "zero", "tpt"]
LABELS = {
    "clip_photo": "CLIP, 1 prompt",
    "clip": "CLIP, TDA's prompts",
    "clip_ens80": "CLIP, 80 prompts",
    "tda_1view": "TDA, 1 view",
    "tda": "TDA, 64 views (official)",
    "zero": "ZERO",
    "tpt": "TPT",
}


def fmt_duration(seconds):
    if seconds is None:
        return ""
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def summarize(dataset, include_smoke=False):
    folder = RESULTS_ROOT / dataset
    metas = sorted(folder.glob("*.meta.json"))
    if not include_smoke:
        metas = [m for m in metas if "smoke" not in m.name]
    spec = REGISTRY.get(dataset)
    targets = spec.targets if spec else {}
    published_run = PUBLISHED_RUN_PILOT if dataset == "imagewoof_pilot" else PUBLISHED_RUN

    rows = []
    for path in metas:
        run = path.name[: -len(".meta.json")]
        meta = json.loads(path.read_text())
        summary = meta.get("summary") or {}
        n = meta.get("n")
        acc = 100 * meta["accuracy"]
        median_ms = summary.get("median_ms")
        target = targets.get(published_run.get(run, ""))
        row = {
            "run": run,
            "method": LABELS.get(run, run),
            "n": n,
            "accuracy": round(acc, 2),
            "published": target if target is not None else "",
            "delta": round(acc - target, 2) if target is not None else "",
            "median_ms_per_image": round(median_ms, 1) if median_ms else "",
            "est_total": fmt_duration(n * median_ms / 1000) if (n and median_ms) else "",
            "saved_at": meta.get("saved_at", ""),
            "git_commit": (meta.get("git_commit") or "")[:8],
        }
        if run.startswith("zero") and "accuracy_fp16_votes" in summary:
            row["note"] = (f"fp16 votes {100 * summary['accuracy_fp16_votes']:.2f}; "
                           f"ties {summary.get('ties_one_hot_votes', summary.get('ties', ''))}")
        elif "n_views" in summary:
            row["note"] = f"{summary['n_views']} view" + ("s" if summary["n_views"] != 1 else "")
        elif "templates" in summary:
            row["note"] = f"{summary['templates']} prompt" + ("s" if summary["templates"] != 1 else "")
        else:
            row["note"] = ""
        rows.append(row)

    rank = {name: i for i, name in enumerate(ORDER)}
    rows.sort(key=lambda r: (rank.get(r["run"], len(ORDER)), r["run"]))
    return rows


COLUMNS = [
    ("method", "Method"), ("run", "File"), ("n", "Images"), ("accuracy", "Acc %"),
    ("published", "Published %"), ("delta", "Δ"), ("median_ms_per_image", "Median ms/img"),
    ("est_total", "≈ Total"), ("note", "Notes"), ("saved_at", "Saved"), ("git_commit", "Commit"),
]


def to_markdown(dataset, rows):
    head = "| " + " | ".join(title for _, title in COLUMNS) + " |"
    rule = "|" + "|".join("---" for _ in COLUMNS) + "|"
    body = []
    for r in rows:
        cells = []
        for key, _ in COLUMNS:
            value = r.get(key, "")
            if key == "delta" and value != "":
                value = f"{value + 0.0:+.2f}".replace("-0.00", "+0.00")
            if key == "run":
                value = f"`{value}`"
            cells.append(str(value))
        body.append("| " + " | ".join(cells) + " |")
    lines = [
        f"# {dataset}: run summary",
        "",
        "Accuracy is top-1 on every image in the run. Published numbers are ViT-B/16 "
        "results from the papers. Δ is shown only for runs that reproduce a published "
        "setting exactly. ≈ Total is images × median time per image, so it leaves out "
        "model loading.",
        "",
        head, rule, *body, "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", help="one dataset folder under results/ (default: all)")
    parser.add_argument("--include-smoke", action="store_true", help="also list *smoke* runs")
    args = parser.parse_args()

    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = sorted(p.name for p in RESULTS_ROOT.iterdir() if p.is_dir() and any(p.glob("*.meta.json")))
    if not datasets:
        raise SystemExit(f"No results found under {RESULTS_ROOT}/")

    for dataset in datasets:
        rows = summarize(dataset, args.include_smoke)
        if not rows:
            print(f"{dataset}: no finished runs")
            continue
        md = to_markdown(dataset, rows)
        folder = RESULTS_ROOT / dataset
        (folder / "summary.md").write_text(md)
        with open(folder / "summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[k for k, _ in COLUMNS])
            writer.writeheader()
            writer.writerows([{k: r.get(k, "") for k, _ in COLUMNS} for r in rows])
        print(md)
        print(f"Wrote {folder / 'summary.md'} and {folder / 'summary.csv'}\n")


if __name__ == "__main__":
    main()