import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import LeaveOneOut, permutation_test_score

from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cost_aware_routing import CHEAP_SCALARS, BUDGETS, oof_needs_tpt_scores, route_curve

WIN_LABELS = Path("results/win_labels.json")
EMBEDDINGS_PATH = Path("results/vanilla_clip_embeddings.npz")

FEATURES = ["vanilla_entropy", "vanilla_confidence", "tpt_view_entropy_std", "tpt_view_entropy_mean"]


class PolicyNet(nn.Module):
    """Small MLP mapping per-image features to a distribution over
    {TPT, TDA}. This is the 'gate' itself."""

    def __init__(self, n_features, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),  # logits over [TPT, TDA]
        )

    def forward(self, x):
        return self.net(x)  # raw logits; softmax applied at use sites


def build_dataset(rows):
    X, r_tpt, r_tda = [], [], []
    for r in rows:
        X.append([r[f] for f in FEATURES])
        r_tpt.append(1.0 if r["tpt_correct"] else 0.0)
        r_tda.append(1.0 if r["tda_correct"] else 0.0)
    return np.array(X), np.array(r_tpt), np.array(r_tda)


def train_reinforce(X_train, r_tpt_train, r_tda_train, epochs=30, lr=0.01):
    model = PolicyNet(n_features=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    r_tpt_t = torch.tensor(r_tpt_train, dtype=torch.float32)
    r_tda_t = torch.tensor(r_tda_train, dtype=torch.float32)

    for epoch in range(epochs):
        logits = model(X_t)
        probs = torch.softmax(logits, dim=-1)

        # Sample an action per image (0 = TPT, 1 = TDA)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()

        # Reward for the sampled action, using known ground truth
        rewards = torch.where(actions == 0, r_tpt_t, r_tda_t)

        # Baseline: average reward of BOTH options (variance reduction).
        # This is a nice bonus your setup allows, since you know both
        # outcomes, unlike a typical bandit problem.
        baseline = (r_tpt_t + r_tda_t) / 2.0
        advantage = rewards - baseline

        log_probs = dist.log_prob(actions)
        loss = -(log_probs * advantage).mean()  # REINFORCE objective

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def evaluate(model, X_test, r_tpt_test, r_tda_test):
    X_t = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_t)
        chosen = torch.argmax(logits, dim=-1).numpy()  # greedy, not sampled, at eval time

    rewards = np.where(chosen == 0, r_tpt_test, r_tda_test)
    return rewards.mean()


def main():
    with open(WIN_LABELS) as f:
        rows = json.load(f)

    # Use ALL images with valid TPT/TDA outcomes, not just disagreements —
    # unlike the classifier-based search, the bandit gets a well-defined
    # reward on every image (both-correct and both-wrong cases just have
    # equal reward regardless of action, so they contribute no gradient
    # signal but don't hurt either).
    X, r_tpt, r_tda = build_dataset(rows)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    majority_baseline = max(r_tpt.mean(), r_tda.mean())
    oracle = np.maximum(r_tpt, r_tda).mean()

    print(f"Total images: {len(rows)}")
    print(f"Fixed-TPT accuracy: {r_tpt.mean():.3f}")
    print(f"Fixed-TDA accuracy: {r_tda.mean():.3f}")
    print(f"Majority baseline (best fixed strategy): {majority_baseline:.3f}")
    print(f"Oracle accuracy: {oracle:.3f}\n")

    # 5-fold CV since this is a full dataset (1000 images), not the small
    # 131-case disagreement set — LOO would be unnecessarily slow here.
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []

    # Stratify on a simple proxy label (which action was better) so folds
    # are balanced, even though training itself is not classifier-style.
    proxy_labels = (r_tda > r_tpt).astype(int)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled, proxy_labels)):
        model = train_reinforce(
            X_scaled[train_idx], r_tpt[train_idx], r_tda[train_idx]
        )
        acc = evaluate(model, X_scaled[test_idx], r_tpt[test_idx], r_tda[test_idx])
        fold_accuracies.append(acc)
        print(f"Fold {fold + 1}: gate accuracy = {acc:.3f}")

    fold_accuracies = np.array(fold_accuracies)
    print(f"\nMean gate accuracy: {fold_accuracies.mean():.3f} "
          f"(+/- {fold_accuracies.std():.3f})")
    print(f"Majority baseline: {majority_baseline:.3f}")
    print(f"Oracle: {oracle:.3f}")

    if fold_accuracies.mean() > majority_baseline:
        print("\nGate beats always-using-the-best-fixed-strategy — real signal.")
    else:
        print("\nGate does not clearly beat the best fixed strategy at this scale.")


# --- Embedding-based experiment: raw CLIP image embeddings as gate input,
# trained with plain supervised learning, per Zeru's suggestion to try
# richer features before reaching for a more complex training algorithm. ---

def load_embeddings():
    npz = np.load(EMBEDDINGS_PATH)
    return {k.replace("__", "/"): npz[k] for k in npz.files}


def build_dataset_with_embeddings(rows, embeddings):
    X, r_tpt, r_tda = [], [], []
    for r in rows:
        if r["tpt_correct"] == r["tda_correct"]:
            continue  # skip non-informative cases (both right or both wrong)
        emb = embeddings.get(r["corrupted_path"])
        if emb is None:
            continue
        X.append(emb)
        r_tpt.append(1.0 if r["tpt_correct"] else 0.0)
        r_tda.append(1.0 if r["tda_correct"] else 0.0)
    return np.array(X), np.array(r_tpt), np.array(r_tda)


def train_supervised(X_train, r_tpt_train, r_tda_train, epochs=30, lr=0.01, verbose=True):
    # Label = whichever strategy actually won this image (ties broken toward TPT)
    y_train = (r_tda_train > r_tpt_train).astype(np.int64)

    model = PolicyNet(n_features=X_train.shape[1], hidden=64)  # bigger hidden layer for 512-dim input
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.1)
    loss_fn = nn.CrossEntropyLoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)

    for epoch in range(epochs):
        logits = model(X_t)
        loss = loss_fn(logits, y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_preds = torch.argmax(model(X_t), dim=-1).numpy()
        train_acc = (train_preds == y_train).mean()
    # print(f"    (final train accuracy: {train_acc:.3f})")

    if verbose:
        print(f"    (final train accuracy: {train_acc:.3f})")

    return model


def run_embedding_experiment(n_components=5):
    with open(WIN_LABELS) as f:
        rows = json.load(f)
    embeddings = load_embeddings()

    X, r_tpt, r_tda = build_dataset_with_embeddings(rows, embeddings)

    majority_baseline = max(r_tpt.mean(), r_tda.mean())
    oracle = np.maximum(r_tpt, r_tda).mean()
    print(f"\n=== Embedding-based experiment (PCA-reduced to {n_components} dims) ===")
    print(f"Majority baseline: {majority_baseline:.3f} | Oracle: {oracle:.3f}\n")

    proxy_labels = (r_tda > r_tpt).astype(int)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, proxy_labels)):
        # Fit scaler AND PCA only on the training fold — fitting on the
        # full dataset before splitting would leak test-fold information
        # into the transformation.
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X[train_idx])
        X_test_scaled = scaler.transform(X[test_idx])

        pca = PCA(n_components=n_components, random_state=42)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        if fold == 0:
            print(f"Explained variance retained: {pca.explained_variance_ratio_.sum():.3f}\n")

        model = train_supervised(
            X_train_pca, r_tpt[train_idx], r_tda[train_idx], epochs=30
        )
        acc = evaluate(model, X_test_pca, r_tpt[test_idx], r_tda[test_idx])

        X_test_t = torch.tensor(X_test_pca, dtype=torch.float32)
        with torch.no_grad():
            test_preds = torch.argmax(model(X_test_t), dim=-1).numpy()
        unique, counts = np.unique(test_preds, return_counts=True)
        print(f"    prediction distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

        fold_accuracies.append(acc)
        print(f"Fold {fold + 1}: gate accuracy = {acc:.3f}")

    fold_accuracies = np.array(fold_accuracies)
    print(f"\nMean gate accuracy: {fold_accuracies.mean():.3f} "
          f"(+/- {fold_accuracies.std():.3f})")
    print(f"Majority baseline: {majority_baseline:.3f}")

    if fold_accuracies.mean() > majority_baseline:
        print("PCA-reduced embedding gate beats best fixed strategy — real signal.")
    else:
        print("PCA-reduced embedding gate does not clearly beat best fixed strategy.")


def check_embedding_gate_stability(n_components=5, seeds=(0, 1, 7, 42, 100, 123, 999)):
    with open(WIN_LABELS) as f:
        rows = json.load(f)
    embeddings = load_embeddings()

    X, r_tpt, r_tda = build_dataset_with_embeddings(rows, embeddings)
    majority_baseline = max(r_tpt.mean(), r_tda.mean())

    print(f"\n=== Stability check: PCA-{n_components} embedding gate, {len(seeds)} seeds ===")
    print(f"Majority baseline: {majority_baseline:.3f}\n")

    proxy_labels = (r_tda > r_tpt).astype(int)
    # Fixed fold split across all seeds — isolates network-initialization
    # variance from data-split variance, same principle as the gradient
    # boosting stability check.
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(kf.split(X, proxy_labels))

    seed_means = []
    for seed in seeds:
        torch.manual_seed(seed)
        fold_accs = []
        for train_idx, test_idx in splits:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X[train_idx])
            X_test_scaled = scaler.transform(X[test_idx])

            pca = PCA(n_components=n_components, random_state=42)
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_test_pca = pca.transform(X_test_scaled)

            model = train_supervised(
                X_train_pca, r_tpt[train_idx], r_tda[train_idx], epochs=30
            )
            acc = evaluate(model, X_test_pca, r_tpt[test_idx], r_tda[test_idx])
            fold_accs.append(acc)

        mean_acc = np.mean(fold_accs)
        seed_means.append(mean_acc)
        marker = " <-- beats baseline" if mean_acc > majority_baseline else ""
        print(f"seed={seed}: mean across 5 folds = {mean_acc:.3f}{marker}")

    seed_means = np.array(seed_means)
    n_beating = sum(seed_means > majority_baseline)
    print(f"\nAcross {len(seeds)} seeds: mean = {seed_means.mean():.3f}, "
          f"std = {seed_means.std():.3f}, "
          f"range = {seed_means.min():.3f}-{seed_means.max():.3f}")
    print(f"{n_beating}/{len(seeds)} seeds beat baseline")

    if seed_means.std() > 0.03:
        print("\nHigh variance across seeds — NOT stable, likely a lucky "
              "configuration rather than a real, trustworthy signal.")
    elif seed_means.mean() > majority_baseline:
        print("\nStable and above baseline — genuine signal, comparable "
              "rigor to the validated gradient boosting result.")
    else:
        print("\nStable but at or below baseline — does not hold up.")

# --- Permutation tests: is the gap over baseline larger than chance? ---
# Shuffling which strategy "won" per image keeps the class balance (and
# therefore the 55.7% baseline) fixed while destroying any real link
# between features and labels. If shuffled labels often score as high as
# the real ones, the real gap isn't meaningful.

def load_disagreements():
    with open(WIN_LABELS) as f:
        rows = json.load(f)
    return [r for r in rows if r["winner"] in ("tpt", "tda")]


def permutation_test_gradient_boosting(n_permutations=200):
    rows = load_disagreements()
    X = np.array([
        [r["vanilla_entropy"], r["tpt_view_entropy_std"], r["tpt_view_entropy_mean"]]
        for r in rows
    ])
    y = np.array([1 if r["winner"] == "tda" else 0 for r in rows])

    model = GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=42)
    print(f"\n=== Permutation test: gradient boosting, LOO, {n_permutations} permutations ===")
    print("(this is the slow one, likely several minutes)")

    score, perm_scores, p_value = permutation_test_score(
        model, X, y,
        cv=LeaveOneOut(),
        n_permutations=n_permutations,
        n_jobs=-1,          # use all CPU cores
        random_state=0,
    )
    report_permutation_result(score, perm_scores, p_value, y)


def embedding_cv_accuracy(X, y, n_components=5, seed=42):
    """Same 5-fold PCA-5 MLP procedure as run_embedding_experiment, but
    driven by a label vector y so it can be rerun on shuffled labels.
    On disagreement cases exactly one strategy is right, so
    r_tda = y and r_tpt = 1 - y reproduces the original rewards."""
    torch.manual_seed(seed)
    r_tda = y.astype(float)
    r_tpt = 1.0 - r_tda

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accs = []
    for train_idx, test_idx in kf.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        pca = PCA(n_components=n_components, random_state=42)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        model = train_supervised(X_train, r_tpt[train_idx], r_tda[train_idx],
                                 epochs=30, verbose=False)
        fold_accs.append(evaluate(model, X_test, r_tpt[test_idx], r_tda[test_idx]))
    return np.mean(fold_accs)


def permutation_test_embedding_mlp(n_permutations=200):
    rows = load_disagreements()
    embeddings = load_embeddings()
    X = np.array([embeddings[r["corrupted_path"]] for r in rows])
    y = np.array([1 if r["winner"] == "tda" else 0 for r in rows])

    print(f"\n=== Permutation test: PCA-5 embedding MLP, 5-fold, {n_permutations} permutations ===")
    score = embedding_cv_accuracy(X, y)

    rng = np.random.default_rng(0)
    perm_scores = np.array([
        embedding_cv_accuracy(X, rng.permutation(y)) for _ in range(n_permutations)
    ])
    p_value = (np.sum(perm_scores >= score) + 1) / (n_permutations + 1)
    report_permutation_result(score, perm_scores, p_value, y)


def report_permutation_result(score, perm_scores, p_value, y):
    baseline = max(np.mean(y), 1 - np.mean(y))
    print(f"Observed accuracy:        {score:.3f}")
    print(f"Majority baseline:        {baseline:.3f}")
    print(f"Shuffled-label accuracy:  mean {perm_scores.mean():.3f}, "
          f"95th pct {np.percentile(perm_scores, 95):.3f}, max {perm_scores.max():.3f}")
    print(f"p-value: {p_value:.3f}")
    if p_value < 0.05:
        print("Beats shuffled labels at p < 0.05 (but see note on multiple comparisons).")
    else:
        print("Not distinguishable from shuffled labels at p < 0.05: treat as suggestive only.")

# --- Cost-shaped RL gate ---
# Reward: choose TDA -> tda_correct (TDA always runs, no extra cost)
#         choose TPT -> tpt_correct - lam (extra cost of running TPT)
# The gate should pick TPT only when it expects TPT to beat TDA's chance of
# being right by more than lam. Sweeping lam traces RL's accuracy-vs-cost
# points, compared directly against the supervised cost-aware scorer.

RL_COST_PLOT = Path("results/rl_cost_curve.png")
LAMBDAS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]


def train_reinforce_cost(X_train, r_tpt, r_tda, lam, epochs=300, lr=0.01, seed=0):
    torch.manual_seed(seed)
    model = PolicyNet(n_features=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    rew_tpt = torch.tensor(r_tpt - lam, dtype=torch.float32)  # action 0
    rew_tda = torch.tensor(r_tda, dtype=torch.float32)        # action 1
    baseline = (rew_tpt + rew_tda) / 2.0                      # variance reduction

    for _ in range(epochs):
        probs = torch.softmax(model(X_t), dim=-1)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        rewards = torch.where(actions == 0, rew_tpt, rew_tda)
        loss = -(dist.log_prob(actions) * (rewards - baseline)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def cost_shaped_rl_sweep(n_repeats=3):
    with open(WIN_LABELS) as f:
        rows = json.load(f)

    y = np.array([r["needs_tpt"] for r in rows], dtype=int)
    tpt_c = np.array([r["tpt_correct"] for r in rows], dtype=float)
    tda_c = np.array([r["tda_correct"] for r in rows], dtype=float)
    tpt_s = np.array([r["tpt_seconds"] for r in rows])
    tda_s = np.array([r["tda_seconds"] for r in rows])
    X = np.array([[r[f] for f in CHEAP_SCALARS] for r in rows])

    tpt_acc, tda_acc = tpt_c.mean(), tda_c.mean()
    print("\n=== Cost-shaped RL gate (cheap scalars, 5-fold, "
          f"averaged over {n_repeats} fold splits) ===")
    print(f"Always TDA: acc {tda_acc:.3f} | Always TPT: acc {tpt_acc:.3f}\n")
    print(f"{'lambda':>7} | {'% to TPT':>8} | {'accuracy':>8} | {'cost (ms)':>9} | {'AUC':>5}")

    rl_points = []
    for lam in LAMBDAS:
        fracs, accs, costs, aucs = [], [], [], []
        for rep in range(n_repeats):
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rep)
            route = np.zeros(len(y), dtype=bool)
            p_tpt = np.zeros(len(y))
            for tr, te in skf.split(X, y):
                sc = StandardScaler().fit(X[tr])
                model = train_reinforce_cost(sc.transform(X[tr]), tpt_c[tr], tda_c[tr],
                                             lam, seed=rep)
                with torch.no_grad():
                    probs = torch.softmax(
                        model(torch.tensor(sc.transform(X[te]), dtype=torch.float32)), dim=-1
                    ).numpy()
                p_tpt[te] = probs[:, 0]
                route[te] = probs[:, 0] > 0.5      # greedy: pick the likelier action
            fracs.append(route.mean())
            accs.append(np.where(route, tpt_c, tda_c).mean())
            costs.append((tda_s + route * tpt_s).mean() * 1000)
            aucs.append(roc_auc_score(y, p_tpt))
        point = (lam, np.mean(fracs), np.mean(accs), np.mean(costs), np.mean(aucs))
        rl_points.append(point)
        print(f"{lam:7.2f} | {point[1]:8.1%} | {point[2]:8.3f} | {point[3]:9.1f} | {point[4]:.3f}")

    # Supervised cost-aware scorer curve for comparison (same features + accounting)
    scorer_accs = np.mean([
        route_curve(oof_needs_tpt_scores(X, None, y, "logreg", seed=r),
                    tpt_c, tda_c, tpt_s, tda_s)[0]
        for r in range(5)
    ], axis=0)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(BUDGETS * 100, scorer_accs * 100, marker="o", ms=3,
            label="supervised scorer (logreg | cheap scalars)")
    ax.plot(BUDGETS * 100, ((1 - BUDGETS) * tda_acc + BUDGETS * tpt_acc) * 100,
            "k--", label="random routing")
    ax.axhline(tpt_acc * 100, color="gray", ls=":", label=f"always TPT ({tpt_acc:.1%})")
    ax.scatter([y.mean() * 100], [np.maximum(tpt_c, tda_c).mean() * 100], marker="*",
               s=200, color="red", zorder=5, label="cost-aware oracle")
    for lam, frac, acc, _, _ in rl_points:
        if frac <= 0.5:
            ax.scatter(frac * 100, acc * 100, color="tab:orange", zorder=4)
            ax.annotate(f"λ={lam}", (frac * 100, acc * 100), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
    ax.scatter([], [], color="tab:orange", label="cost-shaped RL (one point per λ)")
    ax.set_xlabel("% of images routed to TPT")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Cost-shaped RL vs. supervised cost-aware scorer (n=1,000)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RL_COST_PLOT, dpi=150)
    print(f"\nSaved plot to {RL_COST_PLOT}")

if __name__ == "__main__":
    # main()
    # run_embedding_experiment()
    # check_embedding_gate_stability()
    # permutation_test_embedding_mlp()
    # permutation_test_gradient_boosting()
    cost_shaped_rl_sweep()