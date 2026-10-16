import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler

WIN_LABELS = Path("results/win_labels.json")


def evaluate(X, y, label, model):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    loo_scores = cross_val_score(model, X_scaled, y, cv=LeaveOneOut())
    accuracy = loo_scores.mean()
    print(f"{label}: LOO accuracy = {accuracy:.3f}")
    return accuracy


def main():
    with open(WIN_LABELS) as f:
        rows = json.load(f)

    disagreements = [r for r in rows if r["winner"] in ("tpt", "tda")]
    print(f"Disagreement cases: {len(disagreements)}\n")

    y = np.array([1 if r["winner"] == "tda" else 0 for r in disagreements])
    majority_baseline = max(sum(y == 0), sum(y == 1)) / len(y)
    print(f"Majority-class baseline: {majority_baseline:.3f}\n")

    # --- Feature sets ---
    entropy_only = np.array([[r["vanilla_entropy"]] for r in disagreements])
    with_view_spread = np.array([
        [r["vanilla_entropy"], r["tpt_view_entropy_std"], r["tpt_view_entropy_mean"]]
        for r in disagreements
    ])

    # --- Test 1: does a non-linear model do better on the same original feature? ---
    print("--- Non-linear model, entropy only ---")
    evaluate(entropy_only, y, "Logistic regression (entropy only)", LogisticRegression())
    evaluate(entropy_only, y, "Random forest (entropy only)",
             RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42))

    # --- Test 2: does the richer view-spread feature help, with either model? ---
    print("\n--- Adding TPT view-entropy spread as a feature ---")
    evaluate(with_view_spread, y, "Logistic regression (+ view spread)", LogisticRegression())
    evaluate(with_view_spread, y, "Random forest (+ view spread)",
             RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42))


if __name__ == "__main__":
    main()