# This uses a single signal (entropy) to predict which of TDA or TPT will win on a given image

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_score

WIN_LABELS = Path("results/win_labels.json")
OUTPUT_PLOT = Path("results/entropy_vs_winner.png")


def main():
    with open(WIN_LABELS) as f:
        rows = json.load(f)

    # Filter to the disagreement cases only — where TPT and TDA differ,
    # since these are the only images that carry information about which
    # strategy to pick. "both" and "neither" are uninformative for this.
    disagreements = [r for r in rows if r["winner"] in ("tpt", "tda")]
    print(f"Disagreement cases: {len(disagreements)} out of {len(rows)} total images")

    if len(disagreements) < 10:
        print("Too few disagreement cases for a meaningful fit — stopping here.")
        return

    X = np.array([[r["vanilla_entropy"]] for r in disagreements])
    y = np.array([1 if r["winner"] == "tda" else 0 for r in disagreements])  # 1 = TDA won, 0 = TPT won

    print(f"  TPT wins: {sum(y == 0)}")
    print(f"  TDA wins: {sum(y == 1)}")

    # Leave-one-out cross-validation — the only sane choice at n=23.
    # A single train/test split would be nearly meaningless at this size.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression()
    loo_scores = cross_val_score(clf, X_scaled, y, cv=LeaveOneOut())
    accuracy = loo_scores.mean()

    majority_baseline = max(sum(y == 0), sum(y == 1)) / len(y)

    print(f"\nLeave-one-out accuracy (entropy -> winner): {accuracy:.3f}")
    print(f"Majority-class baseline (always predict more common winner): {majority_baseline:.3f}")

    if accuracy > majority_baseline:
        print("Entropy alone beats the majority baseline — some signal present.")
    else:
        print("Entropy alone does NOT beat majority baseline at this sample size — "
              "inconclusive, not necessarily 'no signal'.")

    # Fit on full data (for the plot / reported coefficient) — not for
    # accuracy claims, since this reuses training data.
    clf.fit(X_scaled, y)
    print(f"\nCoefficient on entropy: {clf.coef_[0][0]:.3f} "
          f"({'higher entropy -> more likely TDA wins' if clf.coef_[0][0] > 0 else 'higher entropy -> more likely TPT wins'})")

    # --- Plot: entropy vs winner, colored by corruption type ---
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"gaussian_blur": "tab:blue", "gaussian_noise": "tab:orange"}
    markers = {"tpt": "o", "tda": "^"}

    for r in disagreements:
        ax.scatter(
            r["vanilla_entropy"],
            1 if r["winner"] == "tda" else 0,
            c=colors[r["corruption_type"]],
            marker=markers[r["winner"]],
            s=80,
            edgecolors="black",
            linewidths=0.5,
        )

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["TPT won", "TDA won"])
    ax.set_xlabel("Pre-adaptation entropy (vanilla CLIP)")
    ax.set_title(f"Entropy vs. winning strategy (n={len(disagreements)} disagreement cases)")

    # Manual legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=10, label="TPT won"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="gray", markersize=10, label="TDA won"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="tab:blue", markersize=10, label="Blur"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="tab:orange", markersize=10, label="Noise"),
    ]
    ax.legend(handles=legend_elements, loc="best")

    fig.tight_layout()
    fig.savefig(OUTPUT_PLOT, dpi=150)
    print(f"\nSaved plot to {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()