import json
from pathlib import Path

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

    paths = set(vanilla) & set(tpt) & set(tda)
    if len(paths) != len(vanilla):
        print(f"Warning: only {len(paths)} images present in all three result files "
              f"(expected {len(vanilla)}) — check for mismatched runs.")

    rows = []
    for path in sorted(paths):
        v = vanilla[path]
        t = tpt[path]
        d = tda[path]

        tpt_correct = t["correct"]
        tda_correct = d["correct"]
        vanilla_correct = v["correct"]

        # "Winner" logic: which strategy is correct where vanilla wasn't,
        # or which strategy stays correct where vanilla already was.
        # Four cases per image: both wrong, both right, TPT only, TDA only.
        if tpt_correct and tda_correct:
            winner = "both"
        elif tpt_correct and not tda_correct:
            winner = "tpt"
        elif tda_correct and not tpt_correct:
            winner = "tda"
        else:
            winner = "neither"

        rows.append({
            "corrupted_path": path,
            "true_label": v["true_label"],
            "corruption_type": v["corruption_type"],
            "vanilla_correct": vanilla_correct,
            "vanilla_confidence": v["confidence"],
            "vanilla_entropy": v["entropy"],
            "tpt_correct": tpt_correct,
            "tpt_improved": t["improved_over_vanilla"],
            "tda_correct": tda_correct,
            "tda_improved": d["improved_over_vanilla"],
            "winner": winner,
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"Built win-label dataset with {len(rows)} images, saved to {OUTPUT_PATH}\n")

    # Summary: winner distribution overall
    from collections import Counter
    winner_counts = Counter(r["winner"] for r in rows)
    print("Winner distribution (overall):")
    for w, c in winner_counts.items():
        print(f"  {w}: {c}")

    # Winner distribution by corruption type
    print("\nWinner distribution by corruption type:")
    for ctype in ["gaussian_blur", "gaussian_noise"]:
        subset = [r for r in rows if r["corruption_type"] == ctype]
        counts = Counter(r["winner"] for r in subset)
        print(f"  {ctype}: {dict(counts)}")

    # The key overlap question: among images vanilla got wrong, how often
    # does TPT-only, TDA-only, both, or neither fix it?
    vanilla_wrong = [r for r in rows if not r["vanilla_correct"]]
    print(f"\nAmong {len(vanilla_wrong)} images vanilla CLIP got wrong:")
    fix_counts = Counter(r["winner"] for r in vanilla_wrong)
    for w, c in fix_counts.items():
        print(f"  {w}: {c}")


if __name__ == "__main__":
    main()