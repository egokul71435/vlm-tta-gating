"""Cost-aware routing between TPT and TDA.

needs_tpt = TPT right AND TDA wrong: the only case worth paying TPT's ~18x
cost for. A scorer ranks images by P(needs_tpt) using CHEAP features only
(vanilla CLIP outputs, TDA cache statistics, CLIP embeddings). For each TPT
budget, the top-scoring images go to TPT and the rest to TDA.
"""

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

WIN_LABELS = Path("results/win_labels.json")
EMBEDDINGS_PATH = Path("results/vanilla_clip_embeddings.npz")
COST_CURVE_PLOT = Path("results/cost_curve.png")

CHEAP_SCALARS = [
    "vanilla_entropy", "vanilla_confidence", "vanilla_margin", "vanilla_text_alignment",
    "tda_pos_cache_max_sim", "tda_pos_cache_mean_sim", "tda_cache_agrees_with_clip",
    "tda_cache_fill", "tda_neg_cache_fill",
]
BUDGETS = np.round(np.arange(0, 0.51, 0.05), 2)   # fraction of images routed to TPT


def load_embeddings():
    npz = np.load(EMBEDDINGS_PATH)
    return {k.replace("__", "/"): npz[k] for k in npz.files}


def oof_needs_tpt_scores(X_scalar, X_emb, y, model_name, n_components=5, seed=0):
    """Out-of-fold P(needs_tpt) for every image. Scaling and PCA are fit
    inside each training fold only, so test images never influence them."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    split_on = X_scalar if X_scalar is not None else X_emb
    scores = np.zeros(len(y))
    for tr, te in skf.split(split_on, y):
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

        if model_name == "logreg":
            model = LogisticRegression(class_weight="balanced", max_iter=1000)
        else:
            model = GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=seed)
        model.fit(X_tr, y[tr])
        scores[te] = model.predict_proba(X_te)[:, 1]
    return scores


def route_curve(scores, tpt_correct, tda_correct, tpt_secs, tda_secs):
    """Accuracy and mean cost (ms) at each TPT budget. TDA always runs (its
    cache must keep updating, and the cheap features come from its pass);
    routed images pay for TPT on top."""
    n = len(scores)
    order = np.argsort(-scores)
    accs, costs = [], []
    for b in BUDGETS:
        route = np.zeros(n, dtype=bool)
        route[order[:int(round(b * n))]] = True
        accs.append(np.where(route, tpt_correct, tda_correct).mean())
        costs.append((tda_secs + route * tpt_secs).mean() * 1000)
    return np.array(accs), np.array(costs)


def acc_at(accs, budget):
    return accs[np.where(np.isclose(BUDGETS, budget))[0][0]]


def cost_aware_curve(n_repeats=5):
    with open(WIN_LABELS) as f:
        rows = json.load(f)
    embeddings = load_embeddings()

    y = np.array([r["needs_tpt"] for r in rows], dtype=int)
    tpt_correct = np.array([r["tpt_correct"] for r in rows], dtype=float)
    tda_correct = np.array([r["tda_correct"] for r in rows], dtype=float)
    tpt_secs = np.array([r["tpt_seconds"] for r in rows])
    tda_secs = np.array([r["tda_seconds"] for r in rows])
    X_scalar = np.array([[r[f] for f in CHEAP_SCALARS] for r in rows])
    X_emb = np.array([embeddings[r["corrupted_path"]] for r in rows])

    tpt_acc, tda_acc = tpt_correct.mean(), tda_correct.mean()
    oracle_route = y.astype(bool)
    oracle_acc = np.where(oracle_route, tpt_correct, tda_correct).mean()
    oracle_cost = (tda_secs + oracle_route * tpt_secs).mean() * 1000

    print("\n=== Cost-aware routing (5-fold, out-of-fold scores, "
          f"averaged over {n_repeats} fold splits) ===")
    print(f"needs_tpt base rate: {y.mean():.3f} ({y.sum()}/{len(y)})")
    print(f"Always TDA:         acc {tda_acc:.3f}, cost {tda_secs.mean()*1000:.1f} ms")
    print(f"Always TPT:         acc {tpt_acc:.3f}, cost {tpt_secs.mean()*1000:.1f} ms")
    print(f"Cost-aware oracle:  acc {oracle_acc:.3f}, cost {oracle_cost:.1f} ms "
          f"(TPT on {y.mean():.1%})")
    print(f"Random routing at 20% / 30%: "
          f"{0.8*tda_acc + 0.2*tpt_acc:.3f} / {0.7*tda_acc + 0.3*tpt_acc:.3f}\n")

    feature_sets = {
        "cheap scalars": (X_scalar, None),
        "embeddings (PCA-5)": (None, X_emb),
        "scalars + embeddings": (X_scalar, X_emb),
    }
    curves = {}
    for fs_name, (Xs, Xe) in feature_sets.items():
        for model_name in ("logreg", "gb"):
            accs_r, costs_r, aucs, aps = [], [], [], []
            for rep in range(n_repeats):
                s = oof_needs_tpt_scores(Xs, Xe, y, model_name, seed=rep)
                a, c = route_curve(s, tpt_correct, tda_correct, tpt_secs, tda_secs)
                accs_r.append(a); costs_r.append(c)
                aucs.append(roc_auc_score(y, s)); aps.append(average_precision_score(y, s))
            accs, acc_std, costs = np.mean(accs_r, 0), np.std(accs_r, 0), np.mean(costs_r, 0)
            label = f"{model_name} | {fs_name}"
            curves[label] = (accs, acc_std, costs)

            match = next((b for b, a in zip(BUDGETS, accs) if a >= tpt_acc), None)
            match_str = f"{match:.0%} TPT" if match is not None else "not reached by 50%"
            print(f"{label:32s} AUC {np.mean(aucs):.3f}±{np.std(aucs):.3f} | "
                  f"AP {np.mean(aps):.3f} (base {y.mean():.3f}) | "
                  f"acc@10% {acc_at(accs, 0.10):.3f} @20% {acc_at(accs, 0.20):.3f} "
                  f"@30% {acc_at(accs, 0.30):.3f} | matches always-TPT at: {match_str}")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = BUDGETS * 100
    for label, (accs, acc_std, _) in curves.items():
        ax.plot(x, accs * 100, marker="o", ms=3, label=label)
    ax.plot(x, ((1 - BUDGETS) * tda_acc + BUDGETS * tpt_acc) * 100, "k--", label="random routing")
    ax.axhline(tpt_acc * 100, color="gray", ls=":", label=f"always TPT ({tpt_acc:.1%})")
    ax.scatter([y.mean() * 100], [oracle_acc * 100], marker="*", s=200, color="red",
               zorder=5, label=f"cost-aware oracle ({oracle_acc:.1%})")
    ax.set_xlabel("% of images routed to TPT")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Cost-aware routing: accuracy vs. TPT budget (n=1,000)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(COST_CURVE_PLOT, dpi=150)
    print(f"\nSaved plot to {COST_CURVE_PLOT}")


if __name__ == "__main__":
    cost_aware_curve()