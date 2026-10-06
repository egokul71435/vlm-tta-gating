import json
from collections import Counter
from pathlib import Path

import numpy as np

VANILLA_RESULTS = Path("results/vanilla_clip_results.json")
TPT_RESULTS = Path("results/tpt_results.json")
TDA_RESULTS = Path("results/tda_results.json")
OUTPUT_PATH = Path("results/win_labels.json")


def load_by_path(filepath):
    with open(filepath) as f:
        data = json.load(f)
    return {r["corrupted_path"]: r for r in data}


def main():
    vanilla = load_by_path(VANILLA_RESULTS)
    tpt = load_by_path(TPT_RESULTS)
    tda = load_by_path(TDA_RESULTS)

    # Position of each image in the stream TDA processed (cache depends on order).
    with open(TDA_RESULTS) as f:
        stream_index = {r["corrupted_path"]: i for i, r in enumerate(json.load(f))}

    paths = set(vanilla) & set(tpt) & set(tda)
    if len(paths) != len(vanilla):
        print(f"Warning: only {len(paths)} images present in all three result files "
              f"(expected {len(vanilla)}) — check for mismatched runs.")

    rows = []
    for path in sorted(paths):  # same order as before, so downstream results don't shift
        v = vanilla[path]
        t = tpt[path]
        d = tda[path]

        tpt_correct = t["correct"]
        tda_correct = d["correct"]
        vanilla_correct = v["correct"]

        if tpt_correct and tda_correct:
            winner = "both"
        elif tpt_correct and not tda_correct:
            winner = "tpt"
        elif tda_correct and not tpt_correct:
            winner = "tda"
        else:
            winner = "neither"

        rows.append({
            # --- identity ---
            "corrupted_path": path,
            "true_label": v["true_label"],
            "corruption_type": v["corruption_type"],
            "stream_index": stream_index[path],

            # --- vanilla CLIP signals (cheap: one forward pass) ---
            "vanilla_correct": vanilla_correct,
            "vanilla_confidence": v["confidence"],
            "vanilla_entropy": v["entropy"],
            "vanilla_margin": v["margin"],
            "vanilla_text_alignment": v["text_alignment"],

            # --- TPT outcome + signals (view-spread is costly: 64 views) ---
            "tpt_correct": tpt_correct,
            "tpt_improved": t["improved_over_vanilla"],
            "tpt_view_entropy_std": t["view_entropy_std"],
            "tpt_view_entropy_mean": t["view_entropy_mean"],
            "tpt_seconds": t["seconds"],
            "tpt_view_feature_seconds": t["view_feature_seconds"],

            # --- TDA outcome + cache signals (cheap: taken before cache update) ---
            "tda_correct": tda_correct,
            "tda_improved": d["improved_over_vanilla"],
            "tda_pos_cache_max_sim": d["pos_cache_max_sim"],
            "tda_pos_cache_mean_sim": d["pos_cache_mean_sim"],
            "tda_cache_fill": d["cache_fill"],
            "tda_cache_agrees_with_clip": d["cache_agrees_with_clip"],
            "tda_neg_cache_fill": d["neg_cache_fill"],
            "tda_seconds": d["seconds"],

            # --- labels ---
            "winner": winner,
            "needs_tpt": tpt_correct and not tda_correct,  # cost-aware label: only case worth paying for TPT
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"Built win-label dataset with {len(rows)} images, saved to {OUTPUT_PATH}\n")

    winner_counts = Counter(r["winner"] for r in rows)
    print("Winner distribution (overall):")
    for w, c in winner_counts.items():
        print(f"  {w}: {c}")

    print("\nWinner distribution by corruption type:")
    for ctype in ["gaussian_blur", "gaussian_noise"]:
        subset = [r for r in rows if r["corruption_type"] == ctype]
        counts = Counter(r["winner"] for r in subset)
        print(f"  {ctype}: {dict(counts)}")

    vanilla_wrong = [r for r in rows if not r["vanilla_correct"]]
    print(f"\nAmong {len(vanilla_wrong)} images vanilla CLIP got wrong:")
    fix_counts = Counter(r["winner"] for r in vanilla_wrong)
    for w, c in fix_counts.items():
        print(f"  {w}: {c}")

    # --- Cost-aware summary ---
    n = len(rows)
    n_needs_tpt = sum(r["needs_tpt"] for r in rows)
    tpt_ms = np.median([r["tpt_seconds"] for r in rows]) * 1000
    tda_ms = np.median([r["tda_seconds"] for r in rows]) * 1000
    frac = n_needs_tpt / n
    oracle_acc = sum(r["tpt_correct"] or r["tda_correct"] for r in rows) / n
    oracle_cost = frac * tpt_ms + (1 - frac) * tda_ms

    print("\nCost-aware view (median ms per image):")
    print(f"  needs_tpt: {n_needs_tpt}/{n} images ({frac:.1%})")
    print(f"  Always TDA:          acc {sum(r['tda_correct'] for r in rows)/n:.3f}, cost {tda_ms:.1f} ms")
    print(f"  Always TPT:          acc {sum(r['tpt_correct'] for r in rows)/n:.3f}, cost {tpt_ms:.1f} ms")
    print(f"  Cost-aware oracle:   acc {oracle_acc:.3f}, cost {oracle_cost:.1f} ms "
          f"({tpt_ms / oracle_cost:.1f}x cheaper than always-TPT)")


if __name__ == "__main__":
    main()