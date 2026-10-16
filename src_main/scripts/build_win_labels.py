"""Merge CLIP, TDA and TPT results for one dataset into per-image outcome labels
and gate features, and print the headroom: how often only TPT is right, how
often only TDA is right, and the oracle.

Joins on `id`. Rows are sorted by id so downstream cross-validation splits are
stable across reruns.

Examples:
  python src_main/scripts/build_win_labels.py --dataset imagewoof_pilot
  python src_main/scripts/build_win_labels.py --dataset imagenet_a --tda-tag 1view --out-tag 1view
"""

import argparse
import json
from collections import Counter, defaultdict

import numpy as np

from tta.common import load_results, results_path
from tta.registry import add_dataset_arg, get_dataset


def parse_args():
    p = argparse.ArgumentParser(description="Build per-image TPT/TDA outcome labels.")
    add_dataset_arg(p)
    p.add_argument("--clip-tag", default=None, help="Which clip results to use (clip[_<tag>].json).")
    p.add_argument("--tda-tag", default=None, help="Which TDA results to use, e.g. 1view, seed1.")
    p.add_argument("--tpt-tag", default=None)
    p.add_argument("--out-tag", default=None, help="Output: win_labels[_<tag>].json")
    return p.parse_args()


def main():
    args = parse_args()
    spec = get_dataset(args.dataset)
    vanilla = load_results(results_path(spec.name, "clip", args.clip_tag))
    tda = load_results(results_path(spec.name, "tda", args.tda_tag))
    tpt = load_results(results_path(spec.name, "tpt", args.tpt_tag))
    out = results_path(spec.name, "win_labels", args.out_tag)

    ids = set(vanilla) & set(tpt) & set(tda)
    if not (len(ids) == len(vanilla) == len(tda) == len(tpt)):
        print(f"Warning: {len(ids)} images in all three files (clip {len(vanilla)}, "
              f"tda {len(tda)}, tpt {len(tpt)}) — check for mismatched runs.")

    rows = []
    for i in sorted(ids):
        v, d, t = vanilla[i], tda[i], tpt[i]
        tpt_correct, tda_correct, vanilla_correct = t["correct"], d["correct"], v["correct"]

        if tpt_correct and tda_correct:
            winner = "both"
        elif tpt_correct:
            winner = "tpt"
        elif tda_correct:
            winner = "tda"
        else:
            winner = "neither"

        rows.append({
            # --- identity ---
            "id": i,
            "label": v["label"],
            "corruption_type": v.get("corruption_type"),
            "stream_index": d["stream_index"],          # position in TDA's stream (its cache depends on order)

            # --- vanilla CLIP (single view, dataset templates): free ---
            "vanilla_correct": vanilla_correct,
            "vanilla_confidence": v["clip_conf"],
            "vanilla_entropy": v["clip_entropy"],
            "vanilla_norm_entropy": v["clip_norm_entropy"],
            "vanilla_margin": v["clip_margin"],
            "vanilla_text_alignment": v["text_alignment"],

            # --- TDA outcome + signals: free (TDA runs on every image) ---
            "tda_correct": tda_correct,
            "tda_improved": tda_correct and not vanilla_correct,
            "tda_confidence": d["tda_conf"],
            "tda_margin": d["tda_margin"],
            "tda_norm_entropy": d["tda_norm_entropy"],
            "tda_changed_clip": int(d["tda_changed_clip"]),
            "tda_clip_margin": d["clip_margin"],                 # CLIP as TDA sees it (64 views on ImageNet OOD)
            "tda_clip_norm_entropy": d["clip_norm_entropy"],
            "tda_pos_cache_max_sim": d["pos_cache_max_sim"],
            "tda_pos_cache_mean_sim": d["pos_cache_mean_sim"],
            "tda_pos_cache_sim_clip_pred": d["pos_cache_sim_clip_pred"],
            "tda_cache_fill": d["cache_fill"],
            "tda_cache_agrees_with_clip": d["cache_agrees_with_clip"],
            "tda_neg_cache_fill": d["neg_cache_fill"],
            "tda_pos_cache_logit_max": d["pos_cache_logit_max"],
            "tda_neg_cache_logit_max": d["neg_cache_logit_max"],
            "tda_n_views": d["n_views"],
            "tda_seconds": d["seconds"],

            # --- TPT outcome + signals ---
            "tpt_correct": tpt_correct,
            "tpt_improved": tpt_correct and not vanilla_correct,
            # probe: first k views only (cheap-ish)
            "tpt_probe_entropy_mean": t["probe_entropy_mean"],
            "tpt_probe_entropy_std": t["probe_entropy_std"],
            "tpt_probe_agreement": t["probe_agreement"],
            # all 64 views: costs the view encoding
            "tpt_view_entropy_std": t["view_entropy_std"],
            "tpt_view_entropy_mean": t["view_entropy_mean"],
            "tpt_view_agreement": t["view_agreement"],
            # only after TPT has run: for fusing the two answers, not for routing
            "tpt_confidence": t["tpt_conf"],
            "tpt_margin": t["tpt_margin"],
            "tpt_changed_zs": int(t["tpt_changed_zs"]),
            "tpt_selected_entropy_before": t["selected_entropy_before"],
            "tpt_pred": t["pred"],
            "tda_pred": d["pred"],
            "tpt_seconds": t["seconds"],
            "tpt_view_feature_seconds": t["view_feature_seconds"],

            # --- labels ---
            "winner": winner,
            "needs_tpt": tpt_correct and not tda_correct,  # cost-aware label: only case worth paying for TPT
        })

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print(f"Built win-label dataset with {len(rows)} images, saved to {out}\n")

    print("Winner distribution (overall):")
    for w, c in Counter(r["winner"] for r in rows).items():
        print(f"  {w}: {c}")

    groups = defaultdict(list)
    for r in rows:
        if r["corruption_type"]:
            groups[r["corruption_type"]].append(r["winner"])
    if groups:
        print("\nWinner distribution by corruption type:")
        for name, winners in sorted(groups.items()):
            print(f"  {name}: {dict(Counter(winners))}")

    vanilla_wrong = [r for r in rows if not r["vanilla_correct"]]
    print(f"\nAmong {len(vanilla_wrong)} images vanilla CLIP got wrong:")
    for w, c in Counter(r["winner"] for r in vanilla_wrong).items():
        print(f"  {w}: {c}")

    # --- headroom + cost ---
    n = len(rows)
    n_tpt_only = sum(r["winner"] == "tpt" for r in rows)
    n_tda_only = sum(r["winner"] == "tda" for r in rows)
    tpt_ms = np.median([r["tpt_seconds"] for r in rows]) * 1000
    tda_ms = np.median([r["tda_seconds"] for r in rows]) * 1000
    frac = n_tpt_only / n
    tda_acc = sum(r["tda_correct"] for r in rows) / n
    tpt_acc = sum(r["tpt_correct"] for r in rows) / n
    oracle_acc = sum(r["tpt_correct"] or r["tda_correct"] for r in rows) / n
    oracle_cost = frac * tpt_ms + (1 - frac) * tda_ms
    oracle_cost_conservative = tda_ms + frac * tpt_ms   # TDA always runs, TPT on top

    print("\nHeadroom:")
    print(f"  TPT-only right: {n_tpt_only} ({n_tpt_only / n:.1%}) | TDA-only right: {n_tda_only} "
          f"({n_tda_only / n:.1%}) | oracle {oracle_acc:.3f} "
          f"(+{oracle_acc - max(tda_acc, tpt_acc):.3f} over the better method)")

    print("\nCost-aware view (median ms per image):")
    print(f"  needs_tpt: {n_tpt_only}/{n} images ({frac:.1%})")
    print(f"  Always TDA:          acc {tda_acc:.3f}, cost {tda_ms:.1f} ms")
    print(f"  Always TPT:          acc {tpt_acc:.3f}, cost {tpt_ms:.1f} ms")
    print(f"  Cost-aware oracle:   acc {oracle_acc:.3f}, cost {oracle_cost:.1f} ms "
          f"({tpt_ms / oracle_cost:.1f}x cheaper than always-TPT); "
          f"{oracle_cost_conservative:.1f} ms if TDA also runs on routed images")


if __name__ == "__main__":
    main()