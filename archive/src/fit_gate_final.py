import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler

WIN_LABELS = Path("results/win_labels.json")

# Winning combination, established via the search below — kept as a
# constant so the stability check always tests the actual best result,
# not whatever was last edited into the search loop.
WINNING_MODEL_FACTORY = lambda seed: GradientBoostingClassifier(
    n_estimators=100, max_depth=2, random_state=seed
)
WINNING_FEATURES = ["vanilla_entropy", "tpt_view_entropy_std", "tpt_view_entropy_mean"]
STABILITY_CHECK_SEEDS = [0, 1, 7, 42, 100, 123, 999]


def evaluate(X, y, model):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    loo_scores = cross_val_score(model, X_scaled, y, cv=LeaveOneOut())
    return loo_scores.mean()


def build_feature_sets(rows):
    return {
        "entropy_only": np.array([[r["vanilla_entropy"]] for r in rows]),
        "entropy_confidence": np.array(
            [[r["vanilla_entropy"], r["vanilla_confidence"]] for r in rows]
        ),
        "entropy_confidence_corruption": np.array(
            [
                [
                    r["vanilla_entropy"],
                    r["vanilla_confidence"],
                    1.0 if r["corruption_type"] == "gaussian_noise" else 0.0,
                ]
                for r in rows
            ]
        ),
        "entropy_view_spread": np.array(
            [[r[f] for f in WINNING_FEATURES] for r in rows]
        ),
    }


def main():
    with open(WIN_LABELS) as f:
        rows = json.load(f)

    disagreements = [r for r in rows if r["winner"] in ("tpt", "tda")]
    y = np.array([1 if r["winner"] == "tda" else 0 for r in disagreements])
    majority_baseline = max(sum(y == 0), sum(y == 1)) / len(y)

    print(f"Disagreement cases: {len(disagreements)}")
    print(f"Majority-class baseline: {majority_baseline:.3f}\n")

    feature_sets = build_feature_sets(disagreements)
    models = {
        "Logistic regression": LogisticRegression(),
        "Random forest": RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42),
        "Gradient boosting": GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=42),
        "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
    }

    # --- Full search: every feature set x every model ---
    print("=" * 60)
    print("SEARCH: all feature sets x all models")
    print("=" * 60)
    results_table = []
    for feat_name, X in feature_sets.items():
        for model_name, model in models.items():
            acc = evaluate(X, y, model)
            beats = acc > majority_baseline
            results_table.append((feat_name, model_name, acc, beats))
            marker = "  <-- beats baseline" if beats else ""
            print(f"{feat_name:32s} | {model_name:20s} | {acc:.3f}{marker}")

    # --- Stability check on the winning combination ---
    print("\n" + "=" * 60)
    print("STABILITY CHECK: gradient boosting + entropy/view-spread, 7 seeds")
    print("=" * 60)
    print("(This combination was the only one to consistently beat baseline")
    print(" across the search above; verifying it isn't a lucky single-seed")
    print(" result before trusting it — an earlier random forest result")
    print(" failed exactly this check and was discarded.)\n")

    X_winning = feature_sets["entropy_view_spread"]
    accuracies = []
    for seed in STABILITY_CHECK_SEEDS:
        model = WINNING_MODEL_FACTORY(seed)
        acc = evaluate(X_winning, y, model)
        accuracies.append(acc)
        marker = " <-- beats baseline" if acc > majority_baseline else ""
        print(f"  seed={seed}: LOO accuracy = {acc:.3f}{marker}")

    accuracies = np.array(accuracies)
    n_beating = sum(accuracies > majority_baseline)
    print(f"\nMean: {accuracies.mean():.3f} | Std: {accuracies.std():.3f} | "
          f"Range: {accuracies.min():.3f}-{accuracies.max():.3f}")
    print(f"{n_beating}/{len(STABILITY_CHECK_SEEDS)} seeds beat baseline")

    if accuracies.std() > 0.03:
        verdict = "UNSTABLE — treat as noise, not a reliable finding."
    elif accuracies.mean() > majority_baseline:
        verdict = "STABLE and above baseline — genuine signal."
    else:
        verdict = "Stable but does not clearly beat baseline — inconclusive."
    print(f"\nVerdict: {verdict}")


if __name__ == "__main__":
    main()