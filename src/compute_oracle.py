import json
from pathlib import Path
from collections import Counter

WIN_LABELS = Path("results/win_labels.json")

def main():
    with open(WIN_LABELS) as f:
        rows = json.load(f)

    vanilla_correct = sum(1 for r in rows if r["vanilla_correct"])
    oracle_correct = sum(1 for r in rows if r["tpt_correct"] or r["tda_correct"])
    tpt_correct = sum(1 for r in rows if r["tpt_correct"])
    tda_correct = sum(1 for r in rows if r["tda_correct"])

    n = len(rows)
    print(f"Total images: {n}\n")
    print(f"Vanilla CLIP accuracy: {vanilla_correct / n:.3f}")
    print(f"TPT accuracy:          {tpt_correct / n:.3f}")
    print(f"TDA accuracy:          {tda_correct / n:.3f}")
    print(f"Oracle accuracy:       {oracle_correct / n:.3f}  "
          f"(best of TPT/TDA per image)")

    print(f"\nOracle gap over best fixed strategy: "
          f"{(oracle_correct - max(tpt_correct, tda_correct)) / n:.3f} "
          f"({oracle_correct - max(tpt_correct, tda_correct)} images)")

if __name__ == "__main__":
    main()