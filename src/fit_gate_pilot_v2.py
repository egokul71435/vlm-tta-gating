# This uses multiple signals (entropy, confidence, corruption type) to predict which of TDA or TPT

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler

WIN_LABELS = Path("results/win_labels.json")


def build_features(rows):
    X = []
    for r in rows:
        is_noise = 1.0 if r["corruption_type"] == "gaussian_noise" else 0.0
        X.append([r["vanilla_entropy"], r["vanilla_confidence"], is_noise])
    return np.array(X)


def evaluate(X, y, label):
    clf = LogisticRegression()
    # Standardize features — important once you mix entropy/confidence
    # (continuous, different scales) with a 0/1 indicator.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    loo_scores = cross_val_score(clf, X_scaled, y, cv=LeaveOneOut())
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

    # Compare feature sets head to head
    entropy_only = np.array([[r["vanilla_entropy"]] for r in disagreements])
    evaluate(entropy_only, y, "Entropy only")

    entropy_conf = np.array([[r["vanilla_entropy"], r["vanilla_confidence"]] for r in disagreements])
    evaluate(entropy_conf, y, "Entropy + confidence")

    full_features = build_features(disagreements)
    evaluate(full_features, y, "Entropy + confidence + corruption type")


if __name__ == "__main__":
    main()