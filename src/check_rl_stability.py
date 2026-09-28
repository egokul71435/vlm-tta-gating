import json
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from train_rl_gate import PolicyNet, build_dataset, evaluate, WIN_LABELS, FEATURES


def train_reinforce_seeded(X_train, r_tpt_train, r_tda_train, seed, epochs=300, lr=0.01):
    torch.manual_seed(seed)
    model = PolicyNet(n_features=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    r_tpt_t = torch.tensor(r_tpt_train, dtype=torch.float32)
    r_tda_t = torch.tensor(r_tda_train, dtype=torch.float32)

    for epoch in range(epochs):
        logits = model(X_t)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        rewards = torch.where(actions == 0, r_tpt_t, r_tda_t)
        baseline = (r_tpt_t + r_tda_t) / 2.0
        advantage = rewards - baseline
        log_probs = dist.log_prob(actions)
        loss = -(log_probs * advantage).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def main():
    with open(WIN_LABELS) as f:
        rows = json.load(f)

    X, r_tpt, r_tda = build_dataset(rows)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    majority_baseline = max(r_tpt.mean(), r_tda.mean())
    oracle = np.maximum(r_tpt, r_tda).mean()
    print(f"Majority baseline: {majority_baseline:.3f} | Oracle: {oracle:.3f}\n")

    proxy_labels = (r_tda > r_tpt).astype(int)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(kf.split(X_scaled, proxy_labels))

    seeds = [0, 1, 7, 42, 100]
    all_seed_means = []

    for seed in seeds:
        fold_accs = []
        for train_idx, test_idx in splits:
            model = train_reinforce_seeded(
                X_scaled[train_idx], r_tpt[train_idx], r_tda[train_idx], seed=seed
            )
            acc = evaluate(model, X_scaled[test_idx], r_tpt[test_idx], r_tda[test_idx])
            fold_accs.append(acc)

        mean_acc = np.mean(fold_accs)
        all_seed_means.append(mean_acc)
        marker = " <-- beats baseline" if mean_acc > majority_baseline else ""
        print(f"seed={seed}: mean across 5 folds = {mean_acc:.3f}{marker}")

    all_seed_means = np.array(all_seed_means)
    print(f"\nAcross {len(seeds)} seeds: mean = {all_seed_means.mean():.3f}, "
          f"std = {all_seed_means.std():.3f}, "
          f"range = {all_seed_means.min():.3f}-{all_seed_means.max():.3f}")

    n_beating = sum(all_seed_means > majority_baseline)
    print(f"{n_beating}/{len(seeds)} seeds beat baseline on average")

    if all_seed_means.std() > 0.02:
        print("\nHigh variance across network initializations — training is "
              "unstable, not yet a trustworthy result either way.")
    elif all_seed_means.mean() > majority_baseline:
        print("\nStable and above baseline — real signal from the RL approach.")
    else:
        print("\nStable but at or below baseline — RL approach, as configured, "
              "does not currently beat the best fixed strategy.")


if __name__ == "__main__":
    main()