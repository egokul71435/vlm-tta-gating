"""Cost-aware routing between TDA (cheap, runs on every image) and one
expensive method E (TPT or ZERO, whichever build_win_labels.py was run with).

needs_exp = E right AND TDA wrong: the only case worth paying E's cost for.
A scorer ranks images by P(needs_exp) using CHEAP features only (vanilla CLIP
outputs, TDA cache statistics, CLIP embeddings). For each budget, the
top-scoring images go to E and the rest keep TDA's answer.

Reads results/<dataset>/win_labels[_<tag>].json (from build_win_labels.py) and
the CLIP embeddings; writes plots to results/<dataset>/. Older win_labels files
without the generic exp_* fields are read as TPT.

Gates compared:
  - supervised scorers (logistic regression, gradient boosting, a small MLP) on
    cheap scalars (CLIP uncertainty + TDA cache), TDA's own output (margin,
    entropy, ...; free because TDA runs on every image) and/or PCA-5 CLIP embeddings
  - single-feature thresholds (no training)
  - RL and RL-adjacent gates (--rl): an accuracy-only REINFORCE policy, a
    cost-shaped REINFORCE policy swept over the cost penalty lambda, and a
    supervised "winner" MLP trained only on images where the methods disagree

Examples:
  python src_main/scripts/cost_aware_routing.py --dataset imagewoof_pilot
  python src_main/scripts/cost_aware_routing.py --dataset imagewoof_pilot --rl
  python src_main/scripts/cost_aware_routing.py --dataset imagewoof_pilot --rl-only
  python src_main/scripts/cost_aware_routing.py --dataset imagenet_a --tag 1view --skip-permutation
  python src_main/scripts/cost_aware_routing.py --dataset imagenet_a --tag zero_1view --skip-permutation --rl
"""

import argparse

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from tta.common import results_path
from tta.registry import add_dataset_arg, get_dataset

# Set by configure() from --dataset / --tag.
WIN_LABELS = EMBEDDINGS_PATH = COST_CURVE_PLOT = THRESHOLD_PLOT = RL_PLOT = None
EXP = "TPT"   # display name of the expensive method; set by load_rows()


def load_rows():
    """Read win_labels and make sure every row has the generic exp_* fields.
    Files written before --expensive existed only have tpt_* / needs_tpt."""
    global EXP
    with open(WIN_LABELS) as f:
        rows = json.load(f)
    for r in rows:
        if "needs_exp" not in r:
            r.update(expensive="tpt", needs_exp=r["needs_tpt"],
                     exp_correct=r["tpt_correct"], exp_seconds=r["tpt_seconds"])
    EXP = rows[0]["expensive"].upper()
    return rows


def configure(dataset, tag=None, clip_tag=None):
    global WIN_LABELS, EMBEDDINGS_PATH, COST_CURVE_PLOT, THRESHOLD_PLOT, RL_PLOT
    spec = get_dataset(dataset)
    suffix = f"_{tag}" if tag else ""
    WIN_LABELS = results_path(spec.name, "win_labels", tag)
    clip = results_path(spec.name, "clip", clip_tag)
    EMBEDDINGS_PATH = clip.with_name(clip.stem + "_embeddings.npz")
    COST_CURVE_PLOT = WIN_LABELS.with_name(f"cost_curve{suffix}.png")
    THRESHOLD_PLOT = WIN_LABELS.with_name(f"threshold_baseline{suffix}.png")
    RL_PLOT = WIN_LABELS.with_name(f"rl_gates{suffix}.png")

CHEAP_SCALARS = [
    "vanilla_entropy", "vanilla_confidence", "vanilla_margin", "vanilla_text_alignment",
    "tda_pos_cache_max_sim", "tda_pos_cache_mean_sim", "tda_cache_agrees_with_clip",
    "tda_cache_fill", "tda_neg_cache_fill",
]
# TDA's own output on this image. Also free: TDA runs on every image before the
# routing decision, and these come from its prediction, not from the label.
TDA_OUTPUT = [
    "tda_margin", "tda_norm_entropy", "tda_confidence", "tda_changed_clip",
    "tda_pos_cache_logit_max", "tda_neg_cache_logit_max",
]
FULL_SCALARS = CHEAP_SCALARS + TDA_OUTPUT
FEATURE_GROUPS = {
    "vanilla uncertainty": ["vanilla_entropy", "vanilla_confidence",
                            "vanilla_margin", "vanilla_text_alignment"],
    "TDA cache": ["tda_pos_cache_max_sim", "tda_pos_cache_mean_sim",
                  "tda_cache_agrees_with_clip", "tda_cache_fill", "tda_neg_cache_fill"],
    "TDA output": TDA_OUTPUT,
}
# Scalars used by the feature analysis, the learned gate in the threshold
# comparison, and the RL gates. Set from --gate-features in __main__.
GATE_SCALARS, GATE_NAME = FULL_SCALARS, "cheap + TDA output"
BUDGETS = np.round(np.arange(0, 1.0001, 0.05), 2)   # fraction of images routed to the expensive method


def match_str(accs, target):
    """Smallest budget whose accuracy reaches the always-expensive accuracy, and
    the smallest that gets within 1 point of it."""
    first = lambda tol: next((b for b, a in zip(BUDGETS, accs) if a >= target - tol - 1e-9), None)
    exact, near = first(0.0), first(0.01)
    fmt = lambda b: f"{b:.0%}" if b is not None else "never"
    return f"{fmt(exact)} (within 1 pt at {fmt(near)})"


def load_embeddings():
    npz = np.load(EMBEDDINGS_PATH)
    return dict(zip(npz["ids"].tolist(), npz["embeddings"]))


def mlp_scores(X_tr, y_tr, X_te, hidden=32, epochs=300, lr=0.01, weight_decay=1e-3, seed=0):
    """Small MLP scorer for P(needs_exp): one hidden layer, full-batch Adam, L2
    weight decay, and the positive class up-weighted by neg/pos (the same balance
    as class_weight="balanced" in the logistic regression). Returns a score per
    test image; only its ranking is used."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(X_tr.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, 1))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_pos = max(int(y_tr.sum()), 1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y_tr) - n_pos) / n_pos))
    X_t = torch.tensor(X_tr, dtype=torch.float32)
    y_t = torch.tensor(y_tr, dtype=torch.float32)
    for _ in range(epochs):
        loss = loss_fn(model(X_t).squeeze(1), y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X_te, dtype=torch.float32)).squeeze(1)).numpy()


def oof_needs_exp_scores(X_scalar, X_emb, y, model_name, n_components=5, seed=0):
    """Out-of-fold P(needs_exp) for every image. Scaling and PCA are fit
    inside each training fold only, so test images never influence them."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    split_on = X_scalar if X_scalar is not None else X_emb
    scores = np.zeros(len(y))
    for fold, (tr, te) in enumerate(skf.split(split_on, y)):
        parts_tr, parts_te = [], []
        if X_scalar is not None:
            sc = StandardScaler().fit(X_scalar[tr])
            parts_tr.append(sc.transform(X_scalar[tr]))
            parts_te.append(sc.transform(X_scalar[te]))
        if X_emb is not None:
            sc_e = StandardScaler().fit(X_emb[tr])
            pca = PCA(n_components=n_components, random_state=42).fit(sc_e.transform(X_emb[tr]))
            parts_tr.append(pca.transform(sc_e.transform(X_emb[tr])))
            parts_te.append(pca.transform(sc_e.transform(X_emb[te])))
        X_tr, X_te = np.hstack(parts_tr), np.hstack(parts_te)

        if model_name == "mlp":
            scores[te] = mlp_scores(X_tr, y[tr], X_te, seed=seed * 10 + fold)
            continue
        if model_name == "logreg":
            model = LogisticRegression(class_weight="balanced", max_iter=1000)
        else:
            model = GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=seed)
        model.fit(X_tr, y[tr])
        scores[te] = model.predict_proba(X_te)[:, 1]
    return scores


def route_curve(scores, exp_correct, tda_correct, exp_secs, tda_secs):
    """Accuracy and mean cost (ms) at each budget. TDA always runs (its
    cache must keep updating, and the cheap features come from its pass);
    routed images pay for the expensive method on top."""
    n = len(scores)
    order = np.argsort(-scores)
    accs, costs = [], []
    for b in BUDGETS:
        route = np.zeros(n, dtype=bool)
        route[order[:int(round(b * n))]] = True
        accs.append(np.where(route, exp_correct, tda_correct).mean())
        costs.append((tda_secs + route * exp_secs).mean() * 1000)
    return np.array(accs), np.array(costs)


def acc_at(accs, budget):
    return accs[np.where(np.isclose(BUDGETS, budget))[0][0]]


def cost_aware_curve(n_repeats=5):
    rows = load_rows()
    embeddings = load_embeddings()

    y = np.array([r["needs_exp"] for r in rows], dtype=int)
    exp_correct = np.array([r["exp_correct"] for r in rows], dtype=float)
    tda_correct = np.array([r["tda_correct"] for r in rows], dtype=float)
    exp_secs = np.array([r["exp_seconds"] for r in rows])
    tda_secs = np.array([r["tda_seconds"] for r in rows])
    X_scalar = np.array([[r[f] for f in CHEAP_SCALARS] for r in rows])
    X_full = np.array([[r[f] for f in FULL_SCALARS] for r in rows])
    X_emb = np.array([embeddings[r["id"]] for r in rows])

    exp_acc, tda_acc = exp_correct.mean(), tda_correct.mean()
    oracle_route = y.astype(bool)
    oracle_acc = np.where(oracle_route, exp_correct, tda_correct).mean()
    oracle_cost = (tda_secs + oracle_route * exp_secs).mean() * 1000

    print("\n=== Cost-aware routing (5-fold, out-of-fold scores, "
          f"averaged over {n_repeats} fold splits) ===")
    print(f"needs_exp ({EXP} right, TDA wrong) base rate: {y.mean():.3f} ({y.sum()}/{len(y)})")
    print(f"Always TDA:         acc {tda_acc:.3f}, cost {tda_secs.mean()*1000:.1f} ms")
    print(f"Always {EXP + ':':<13s}acc {exp_acc:.3f}, cost {exp_secs.mean()*1000:.1f} ms")
    print(f"Cost-aware oracle:  acc {oracle_acc:.3f}, cost {oracle_cost:.1f} ms "
          f"({EXP} on {y.mean():.1%})")
    print(f"Random routing at 20% / 30%: "
          f"{0.8*tda_acc + 0.2*exp_acc:.3f} / {0.7*tda_acc + 0.3*exp_acc:.3f}\n")

    feature_sets = {
        "cheap scalars": (X_scalar, None),
        "cheap + TDA output": (X_full, None),
        "embeddings (PCA-5)": (None, X_emb),
        "scalars + embeddings": (X_scalar, X_emb),
        "cheap + TDA output + emb": (X_full, X_emb),
    }
    curves = {}
    for fs_name, (Xs, Xe) in feature_sets.items():
        for model_name in ("logreg", "gb", "mlp"):
            accs_r, costs_r, aucs, aps = [], [], [], []
            for rep in range(n_repeats):
                s = oof_needs_exp_scores(Xs, Xe, y, model_name, seed=rep)
                a, c = route_curve(s, exp_correct, tda_correct, exp_secs, tda_secs)
                accs_r.append(a); costs_r.append(c)
                aucs.append(roc_auc_score(y, s)); aps.append(average_precision_score(y, s))
            accs, acc_std, costs = np.mean(accs_r, 0), np.std(accs_r, 0), np.mean(costs_r, 0)
            label = f"{model_name} | {fs_name}"
            curves[label] = (accs, acc_std, costs)

            print(f"{label:32s} AUC {np.mean(aucs):.3f}±{np.std(aucs):.3f} | "
                  f"AP {np.mean(aps):.3f} (base {y.mean():.3f}) | "
                  f"acc@10% {acc_at(accs, 0.10):.3f} @20% {acc_at(accs, 0.20):.3f} "
                  f"@30% {acc_at(accs, 0.30):.3f} @50% {acc_at(accs, 0.50):.3f} | peak {accs.max():.3f} at {BUDGETS[accs.argmax()]:.0%} | "
                  f"matches always-{EXP} at: {match_str(accs, exp_acc)}")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = BUDGETS * 100
    for label, (accs, acc_std, _) in curves.items():
        ax.plot(x, accs * 100, marker="o", ms=3, label=label)
    ax.plot(x, ((1 - BUDGETS) * tda_acc + BUDGETS * exp_acc) * 100, "k--", label="random routing")
    ax.axhline(exp_acc * 100, color="gray", ls=":", label=f"always {EXP} ({exp_acc:.1%})")
    ax.scatter([y.mean() * 100], [oracle_acc * 100], marker="*", s=200, color="red",
               zorder=5, label=f"cost-aware oracle ({oracle_acc:.1%})")
    ax.set_xlabel(f"% of images routed to {EXP}")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Cost-aware routing: accuracy vs. {EXP} budget, TDA always runs (n={len(rows):,})")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(COST_CURVE_PLOT, dpi=150)
    print(f"\nSaved plot to {COST_CURVE_PLOT}")

# --- Feature analysis + significance for logreg on the gate scalars (--gate-features) ---

def mean_oof_auc(X, y, n_repeats=5):
    return np.mean([
        roc_auc_score(y, oof_needs_exp_scores(X, None, y, "logreg", seed=r))
        for r in range(n_repeats)
    ])


def feature_analysis(n_repeats=5, n_permutations=200):
    rows = load_rows()
    y = np.array([r["needs_exp"] for r in rows], dtype=int)
    X = np.array([[r[f] for f in GATE_SCALARS] for r in rows])

    print(f"\n=== Feature analysis: logreg | {GATE_NAME} ===")

    # 1. Coefficients (fit on all data, standardized features).
    # Correlated features (entropy/confidence/margin, r~0.98) split weight
    # between them, so read these alongside the group ablation below.
    sc = StandardScaler().fit(X)
    model = LogisticRegression(class_weight="balanced", max_iter=1000).fit(sc.transform(X), y)
    print(f"\nCoefficients (standardized; + means higher value -> more likely to need {EXP}):")
    for f, c in sorted(zip(GATE_SCALARS, model.coef_[0]), key=lambda t: -abs(t[1])):
        print(f"  {f:28s} {c:+.3f}")

    # 2. Drop-one ablation: how much out-of-fold AUC falls without each feature.
    base_auc = mean_oof_auc(X, y, n_repeats)
    print(f"\nBaseline out-of-fold AUC (all features): {base_auc:.3f}")
    print("Drop-one ablation (negative delta = feature helps):")
    for i, f in enumerate(GATE_SCALARS):
        auc = mean_oof_auc(np.delete(X, i, axis=1), y, n_repeats)
        print(f"  without {f:28s} AUC {auc:.3f}  (delta {auc - base_auc:+.3f})")

    # 3. Group ablation: each group alone, and everything except each group.
    print("\nGroup ablation:")
    for g, feats in FEATURE_GROUPS.items():
        if not all(f in GATE_SCALARS for f in feats):
            continue
        idx = [GATE_SCALARS.index(f) for f in feats]
        only = mean_oof_auc(X[:, idx], y, n_repeats)
        without = mean_oof_auc(np.delete(X, idx, axis=1), y, n_repeats)
        print(f"  {g:20s} alone: AUC {only:.3f} | all except it: AUC {without:.3f}")

    # 4. Permutation test on AUC: shuffle needs_exp, rerun the same CV.
    if n_permutations <= 0:
        print("\nPermutation test skipped.")
        return
    observed = roc_auc_score(y, oof_needs_exp_scores(X, None, y, "logreg", seed=0))
    rng = np.random.default_rng(0)
    perm_aucs = np.array([
        roc_auc_score(y_p, oof_needs_exp_scores(X, None, y_p, "logreg", seed=0))
        for y_p in (rng.permutation(y) for _ in range(n_permutations))
    ])
    p = (np.sum(perm_aucs >= observed) + 1) / (n_permutations + 1)
    print(f"\nPermutation test ({n_permutations} shuffles of needs_exp):")
    print(f"  Observed AUC {observed:.3f} | shuffled mean {perm_aucs.mean():.3f}, "
          f"95th pct {np.percentile(perm_aucs, 95):.3f}, max {perm_aucs.max():.3f}")
    print(f"  p-value: {p:.4f}  (smallest possible with {n_permutations} shuffles: "
          f"{1/(n_permutations+1):.4f})")

# --- Simple-threshold baselines: route by ONE feature, no training ---
# Each rule's direction is fixed a priori (more uncertain -> send to the expensive method),
# not chosen from results, so these are fair baselines for the learned gate.


def threshold_baselines(n_repeats=5):
    rows = load_rows()

    y = np.array([r["needs_exp"] for r in rows], dtype=int)
    exp_c = np.array([r["exp_correct"] for r in rows], dtype=float)
    tda_c = np.array([r["tda_correct"] for r in rows], dtype=float)
    exp_s = np.array([r["exp_seconds"] for r in rows])
    tda_s = np.array([r["tda_seconds"] for r in rows])
    X = np.array([[r[f] for f in GATE_SCALARS] for r in rows])
    exp_acc, tda_acc = exp_c.mean(), tda_c.mean()

    entropy_ = np.array([r["vanilla_entropy"] for r in rows])
    disagree = 1 - np.array([r["tda_cache_agrees_with_clip"] for r in rows])
    # Hand-made two-signal rule: cache disagreement first, ties broken by entropy
    ent_norm = (entropy_ - entropy_.min()) / (entropy_.max() - entropy_.min())

    rules = {
        "threshold: high entropy": entropy_,
        "threshold: low confidence": -np.array([r["vanilla_confidence"] for r in rows]),
        "threshold: low margin": -np.array([r["vanilla_margin"] for r in rows]),
        "threshold: low TDA margin": -np.array([r["tda_margin"] for r in rows]),
        "threshold: low TDA confidence": -np.array([r["tda_confidence"] for r in rows]),
        "rule: cache disagrees, then entropy": disagree + 0.5 * ent_norm,
    }

    # Learned gate (logreg | gate scalars), averaged over fold splits
    learned = [oof_needs_exp_scores(X, None, y, "logreg", seed=r) for r in range(n_repeats)]
    learned_curve = np.mean([route_curve(s, exp_c, tda_c, exp_s, tda_s)[0] for s in learned], 0)

    print("\n=== Simple-threshold baselines vs. learned gate (no training for thresholds) ===")
    print(f"Always TDA {tda_acc:.3f} | Always {EXP} {exp_acc:.3f} | needs_exp base rate {y.mean():.3f}\n")

    def summarize(name, accs, auc, ap):
        print(f"{name:46s} AUC {auc:.3f} | AP {ap:.3f} | "
              f"@10% {acc_at(accs, 0.10):.3f} @20% {acc_at(accs, 0.20):.3f} "
              f"@25% {acc_at(accs, 0.25):.3f} @30% {acc_at(accs, 0.30):.3f} @50% {acc_at(accs, 0.50):.3f} | "
              f"peak {accs.max():.3f} at {BUDGETS[accs.argmax()]:.0%} | "
              f"matches {EXP} at {match_str(accs, exp_acc)}")

    curves = {}
    for name, score in rules.items():
        accs, _ = route_curve(score, exp_c, tda_c, exp_s, tda_s)
        curves[name] = accs
        summarize(name, accs, roc_auc_score(y, score), average_precision_score(y, score))

    curves[f"learned gate (logreg | {GATE_NAME})"] = learned_curve
    summarize(f"learned gate (logreg | {GATE_NAME})", learned_curve,
              np.mean([roc_auc_score(y, s) for s in learned]),
              np.mean([average_precision_score(y, s) for s in learned]))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = BUDGETS * 100
    for name, accs in curves.items():
        style = "-" if name.startswith("learned") else "--"
        ax.plot(x, accs * 100, style, marker="o", ms=3, label=name)
    ax.plot(x, ((1 - BUDGETS) * tda_acc + BUDGETS * exp_acc) * 100, "k:", label="random routing")
    ax.axhline(exp_acc * 100, color="gray", ls=":", label=f"always {EXP} ({exp_acc:.1%})")
    ax.set_xlabel(f"% of images routed to {EXP}")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Simple-threshold baselines vs. learned gate (n={len(rows):,})")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(THRESHOLD_PLOT, dpi=150)
    print(f"\nSaved plot to {THRESHOLD_PLOT}")


# --- RL and RL-adjacent gates ---
# Every image's outcome is known for BOTH actions (E right? TDA right?), so the
# gate is a contextual bandit with full feedback. Three gates from the pilot:
#   1. accuracy-only REINFORCE: reward = whether the chosen method was right
#   2. cost-shaped REINFORCE: choosing E costs lam (reward = exp_correct - lam;
#      TDA always runs, so choosing it is free). The policy should pick E only
#      when it expects E to beat TDA by more than lam. Sweeping lam gives one
#      (budget, accuracy) point per value.
#   3. "winner" MLP (RL-adjacent, supervised): trained only on images where the
#      methods disagree, to predict which one wins.
# All use the same cheap features, folds and cost accounting as the scorers
# above, so their points and curves sit on the same plot. Gates that used E's
# 64-view statistics in the first pilot are not repeated: those features cost
# nearly all of E's view encoding, so they cannot be part of a cheap gate.

LAMBDAS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]


def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def _policy_net(n_features, hidden=16):
    torch, nn = _torch()
    return nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, 2))  # logits [E, TDA]


def train_reinforce(X, r_exp, r_tda, lam=0.0, epochs=300, lr=0.01, seed=0):
    """REINFORCE over {E, TDA}. Baseline = mean reward of the two actions,
    which is available here because both outcomes are known."""
    torch, _ = _torch()
    torch.manual_seed(seed)
    model = _policy_net(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X_t = torch.tensor(X, dtype=torch.float32)
    rew_exp = torch.tensor(r_exp - lam, dtype=torch.float32)   # action 0
    rew_tda = torch.tensor(r_tda, dtype=torch.float32)         # action 1
    baseline = (rew_exp + rew_tda) / 2.0
    for _ in range(epochs):
        dist = torch.distributions.Categorical(logits=model(X_t))
        actions = dist.sample()
        rewards = torch.where(actions == 0, rew_exp, rew_tda)
        loss = -(dist.log_prob(actions) * (rewards - baseline)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def train_winner_mlp(X, r_exp, r_tda, epochs=300, lr=0.01, seed=0, weight_decay=1e-3):
    """Supervised: fit only images where exactly one method is right; label = which."""
    torch, nn = _torch()
    torch.manual_seed(seed)
    keep = r_exp != r_tda
    y = (r_tda[keep] > r_exp[keep]).astype(np.int64)          # 0 = E wins, 1 = TDA wins
    model = _policy_net(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    X_t = torch.tensor(X[keep], dtype=torch.float32)
    y_t = torch.tensor(y)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        loss = loss_fn(model(X_t), y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def oof_gate_scores(X_scalar, X_emb, y, r_exp, r_tda, kind, lam=0.0, n_components=5, seed=0):
    """Out-of-fold P(choose E) for every image, with the same folds, scaling and
    PCA handling as oof_needs_exp_scores."""
    torch, _ = _torch()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    split_on = X_scalar if X_scalar is not None else X_emb
    scores = np.zeros(len(y))
    for fold, (tr, te) in enumerate(skf.split(split_on, y)):
        parts_tr, parts_te = [], []
        if X_scalar is not None:
            sc = StandardScaler().fit(X_scalar[tr])
            parts_tr.append(sc.transform(X_scalar[tr]))
            parts_te.append(sc.transform(X_scalar[te]))
        if X_emb is not None:
            sc_e = StandardScaler().fit(X_emb[tr])
            pca = PCA(n_components=n_components, random_state=42).fit(sc_e.transform(X_emb[tr]))
            parts_tr.append(pca.transform(sc_e.transform(X_emb[tr])))
            parts_te.append(pca.transform(sc_e.transform(X_emb[te])))
        X_tr, X_te = np.hstack(parts_tr), np.hstack(parts_te)
        if kind == "reinforce":
            model = train_reinforce(X_tr, r_exp[tr], r_tda[tr], lam=lam, seed=seed * 10 + fold)
        else:
            model = train_winner_mlp(X_tr, r_exp[tr], r_tda[tr], seed=seed * 10 + fold)
        with torch.no_grad():
            scores[te] = torch.softmax(model(torch.tensor(X_te, dtype=torch.float32)), -1)[:, 0].numpy()
    return scores


def rl_gates(n_repeats=3):
    rows = load_rows()
    embeddings = load_embeddings()

    y = np.array([r["needs_exp"] for r in rows], dtype=int)
    exp_c = np.array([r["exp_correct"] for r in rows], dtype=float)
    tda_c = np.array([r["tda_correct"] for r in rows], dtype=float)
    exp_s = np.array([r["exp_seconds"] for r in rows])
    tda_s = np.array([r["tda_seconds"] for r in rows])
    X = np.array([[r[f] for f in GATE_SCALARS] for r in rows])
    X_emb = np.array([embeddings[r["id"]] for r in rows])
    exp_acc, tda_acc = exp_c.mean(), tda_c.mean()
    oracle_acc = np.maximum(exp_c, tda_c).mean()

    print(f"\n=== RL and RL-adjacent gates ({GATE_NAME}, 5-fold, {n_repeats} fold splits) ===")
    print(f"Always TDA {tda_acc:.3f} | Always {EXP} {exp_acc:.3f} | oracle {oracle_acc:.3f} "
          f"| needs_exp base rate {y.mean():.3f}")

    # 1-2. REINFORCE: greedy decision (P(E) > 0.5) gives one point per lambda
    print("\nREINFORCE, greedy decisions (lambda = 0 is the accuracy-only gate):")
    print(f"{'lambda':>7} | {'% to ' + EXP:>8} | {'accuracy':>8} | {'cost (ms)':>9} | {'AUC':>5}")
    rl_points, scores_lam0 = [], []
    for lam in LAMBDAS:
        fracs, accs, costs, aucs = [], [], [], []
        for rep in range(n_repeats):
            p = oof_gate_scores(X, None, y, exp_c, tda_c, "reinforce", lam=lam, seed=rep)
            if lam == 0.0:
                scores_lam0.append(p)
            route = p > 0.5
            fracs.append(route.mean())
            accs.append(np.where(route, exp_c, tda_c).mean())
            costs.append((tda_s + route * exp_s).mean() * 1000)
            aucs.append(roc_auc_score(y, p))
        point = (lam, np.mean(fracs), np.mean(accs), np.mean(costs), np.mean(aucs))
        rl_points.append(point)
        print(f"{lam:7.2f} | {point[1]:8.1%} | {point[2]:8.3f} | {point[3]:9.1f} | {point[4]:.3f}")

    # Curves: rank by each gate's P(E) and sweep the budget, like the scorers
    def summarize(name, score_list):
        accs = np.mean([route_curve(s, exp_c, tda_c, exp_s, tda_s)[0] for s in score_list], 0)
        auc = np.mean([roc_auc_score(y, s) for s in score_list])
        ap = np.mean([average_precision_score(y, s) for s in score_list])
        print(f"{name:42s} AUC {auc:.3f} | AP {ap:.3f} | @10% {acc_at(accs, 0.10):.3f} "
              f"@20% {acc_at(accs, 0.20):.3f} @30% {acc_at(accs, 0.30):.3f} @50% {acc_at(accs, 0.50):.3f} | "
              f"peak {accs.max():.3f} at {BUDGETS[accs.argmax()]:.0%} | matches {EXP} at {match_str(accs, exp_acc)}")
        return accs

    print("\nAs ranking scores over the same budgets:")
    curves = {
        "logreg scorer (reference)": summarize(
            "logreg scorer (reference)",
            [oof_needs_exp_scores(X, None, y, "logreg", seed=r) for r in range(n_repeats)]),
        f"REINFORCE, accuracy-only (P({EXP}) ranked)": summarize(
            f"REINFORCE, accuracy-only (P({EXP}) ranked)", scores_lam0),
        f"winner MLP | {GATE_NAME}": summarize(
            f"winner MLP | {GATE_NAME}",
            [oof_gate_scores(X, None, y, exp_c, tda_c, "winner", seed=r) for r in range(n_repeats)]),
        "winner MLP | embeddings (PCA-5)": summarize(
            "winner MLP | embeddings (PCA-5)",
            [oof_gate_scores(None, X_emb, y, exp_c, tda_c, "winner", seed=r) for r in range(n_repeats)]),
    }

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = BUDGETS * 100
    for name, accs in curves.items():
        style = "-" if name.startswith("logreg") else "--"
        ax.plot(x, accs * 100, style, marker="o", ms=3, label=name)
    ax.plot(x, ((1 - BUDGETS) * tda_acc + BUDGETS * exp_acc) * 100, "k:", label="random routing")
    ax.axhline(exp_acc * 100, color="gray", ls=":", label=f"always {EXP} ({exp_acc:.1%})")
    ax.scatter([y.mean() * 100], [oracle_acc * 100], marker="*", s=200, color="red",
               zorder=5, label=f"cost-aware oracle ({oracle_acc:.1%})")
    for lam, frac, acc, _, _ in rl_points:
        if frac <= BUDGETS.max():
            ax.scatter(frac * 100, acc * 100, color="tab:orange", zorder=4)
            ax.annotate(f"λ={lam}", (frac * 100, acc * 100), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
    ax.scatter([], [], color="tab:orange", label="cost-shaped REINFORCE (one point per λ)")
    ax.set_xlabel(f"% of images routed to {EXP}")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"RL and RL-adjacent gates vs. the supervised scorer (n={len(rows):,})")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RL_PLOT, dpi=150)
    print(f"\nSaved plot to {RL_PLOT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cost-aware routing analysis for one dataset.")
    add_dataset_arg(parser)
    parser.add_argument("--tag", default=None, help="Which win_labels to use (win_labels[_<tag>].json).")
    parser.add_argument("--clip-tag", default=None, help="Which clip embeddings to use.")
    parser.add_argument("--skip-permutation", action="store_true",
                        help="Skip the 200-shuffle permutation test (slow on large datasets).")
    parser.add_argument("--rl", action="store_true", help="Also run the RL and RL-adjacent gates.")
    parser.add_argument("--rl-only", action="store_true", help="Run only the RL and RL-adjacent gates.")
    parser.add_argument("--gate-features", choices=["full", "cheap"], default="full",
                        help="Scalars for the feature analysis, threshold comparison and RL gates: "
                             "'full' adds TDA's own output to the cheap scalars; 'cheap' is the pilot set.")
    args = parser.parse_args()
    configure(args.dataset, args.tag, args.clip_tag)
    if args.gate_features == "cheap":
        GATE_SCALARS, GATE_NAME = CHEAP_SCALARS, "cheap scalars"

    if not args.rl_only:
        cost_aware_curve()
        feature_analysis(n_permutations=0 if args.skip_permutation else 200)
        threshold_baselines()
    if args.rl or args.rl_only:
        rl_gates()