import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler

WIN_LABELS = Path("results/win_labels.json")


def evaluate(X, y, seed):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=seed)
    loo_scores = cross_val_score(model, X_scaled, y, cv=LeaveOneOut())
    return loo_scores.mean()


def main():
    with open(WIN_LABELS) as f:
        rows = json.load(f)

    disagreements = [r for r in rows if r["winner"] in ("tpt", "tda")]
    y = np.array([1 if r["winner"] == "tda" else 0 for r in disagreements])
    majority_baseline = max(sum(y == 0), sum(y == 1)) / len(y)

    X = np.array([
        [r["vanilla_entropy"], r["tpt_view_entropy_std"], r["tpt_view_entropy_mean"]]
        for r in disagreements
    ])

    print(f"Disagreement cases: {len(disagreements)}")
    print(f"Majority-class baseline: {majority_baseline:.3f}\n")

    seeds = [0, 1, 7, 42, 100, 123, 999]
    accuracies = []
    for seed in seeds:
        acc = evaluate(X, y, seed)
        accuracies.append(acc)
        marker = " <-- beats baseline" if acc > majority_baseline else ""
        print(f"Gradient boosting (seed={seed}): LOO accuracy = {acc:.3f}{marker}")

    accuracies = np.array(accuracies)
    print(f"\nMean across {len(seeds)} seeds: {accuracies.mean():.3f}")
    print(f"Std across seeds: {accuracies.std():.3f}")
    print(f"Range: {accuracies.min():.3f} - {accuracies.max():.3f}")

    n_beating_baseline = sum(accuracies > majority_baseline)
    print(f"\n{n_beating_baseline}/{len(seeds)} seeds beat the majority baseline")

    if accuracies.std() > 0.03:
        print("\nHigh variance across seeds — result is NOT stable, "
              "the earlier 61.1% was likely a lucky draw, not real signal.")
    elif accuracies.mean() > majority_baseline:
        print("\nConsistently above baseline across seeds — this looks like "
              "a real, if modest, signal worth investigating further.")
    else:
        print("\nMean does not clearly beat baseline — treat as noise, "
              "not a reliable finding.")


if __name__ == "__main__":
    main()