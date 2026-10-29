"""Fusion, cascade and CAS-style baselines from saved runs (no model is loaded).

Everything here is simulated exactly from per-image outputs: 1-view TDA runs on
every image (its cache sees the whole stream either way), and ZERO and TPT
reset per image, so running them on a subset gives the same per-image result
as running them on everything.

Needs, in results/<dataset>/:
  clip.json                    gate signal (CLIP top-2 margin)
  tda_<tda-tag>.json + .probs.npz   cheap method (run with --save-probs)
  zero.json + zero.probs.npz        ZERO (run_zero.py --save-probs)
  tpt.json + tpt.probs.npz          TPT (optional; run_tpt.py --save-probs)
  tda.json                     64-view TDA reference point (optional)

What it reports:
  1. Fixed points: each method alone, and averages of methods run on every image.
  2. Routing TDA -> E (E = ZERO or TPT) with a label-free low-margin gate and
     four ways to answer on routed images: replace (E's answer), average of
     TDA's and E's probabilities, the more confident of the two, and a weighted
     average swept over w (tuned on this same data, so reference only).
  3. Cascade TDA -> ZERO -> TPT: the margin gate picks images for ZERO, then
     ZERO's own uncertainty picks which of those also get TPT; answers are the
     average of every method that ran. TPT's cost is shown twice: as run here
     (its own 64 views), and reusing ZERO's encoded views (TPT minus its view
     and encoding time), which is what a real cascade would pay.
  4. CAS-style adapt-or-skip: CAS's gate needs the augmented views encoded, so
     every image pays ZERO's view + encoding cost; skipped images fall back to
     CLIP or to 1-view TDA. Scored with the consistency signals logged by
     run_zero.py (CAS's code is not released; this follows its paper).
Plus the accuracy-cost Pareto front over all of the above, a plot and a JSON.

Example:
  python src_main/scripts/fusion_cascade.py --dataset imagenet_a --tda-tag 1view
"""

import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tta.common import load_results, results_path
from tta.registry import add_dataset_arg, get_dataset

BUDGETS = np.round(np.arange(0, 1.0001, 0.05), 2)


def load_probs(path, key, ids):
    npz = np.load(path)
    index = {str(i): k for k, i in enumerate(npz["ids"])}
    missing = [i for i in ids if i not in index]
    if missing:
        raise SystemExit(f"{path}: {len(missing)} ids missing (e.g. {missing[0]}); was it run with --save-probs?")
    return npz[key][[index[i] for i in ids]].astype(np.float64)


def top_k_mask(score, frac):
    """Boolean mask of the round(frac * n) highest scores."""
    n = len(score)
    mask = np.zeros(n, dtype=bool)
    mask[np.argsort(-score, kind="stable")[: int(round(frac * n))]] = True
    return mask


def acc(pred, labels):
    return float((pred == labels).mean())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_arg(p)
    p.add_argument("--tda-tag", default="1view", help="Cheap TDA run: tda_<tag>.json (default 1view).")
    p.add_argument("--clip-tag", default=None)
    p.add_argument("--out-tag", default=None, help="Output: fusion_cascade[_<tag>].json/.png")
    args = p.parse_args()
    spec = get_dataset(args.dataset)
    R = lambda method, tag=None: results_path(spec.name, method, tag)

    clip = load_results(R("clip", args.clip_tag))
    tda = load_results(R("tda", args.tda_tag))
    zero = load_results(R("zero"))
    tpt_path, tda64_path = R("tpt"), R("tda")
    tpt = load_results(tpt_path) if tpt_path.exists() and tpt_path.with_suffix(".probs.npz").exists() else None
    tda64 = load_results(tda64_path) if (tda64_path.exists() and args.tda_tag) else None

    ids = sorted(set(clip) & set(tda) & set(zero) & (set(tpt) if tpt else set(zero)))
    if len(ids) != len(tda):
        print(f"Note: {len(ids)} images in every file (tda {len(tda)}, zero {len(zero)}"
              f"{f', tpt {len(tpt)}' if tpt else ''}).")
    n = len(ids)
    labels = np.array([clip[i]["label"] for i in ids])
    margin = np.array([clip[i]["clip_margin"] for i in ids])
    gate = -margin                                   # low CLIP margin first; no labels used

    P_tda = load_probs(R("tda", args.tda_tag).with_suffix(".probs.npz"), "final_probs", ids)
    P_zero = load_probs(R("zero").with_suffix(".probs.npz"), "zero_sel_avg_probs", ids)
    V_zero = load_probs(R("zero").with_suffix(".probs.npz"), "zero_votes", ids)
    pred_tda = np.array([tda[i]["pred"] for i in ids])
    pred_zero = np.array([zero[i]["pred"] for i in ids])
    pred_clip = np.array([clip[i]["pred"] for i in ids])
    ms = lambda rows, key="seconds": np.array([rows[i][key] for i in ids]) * 1000
    c_tda, c_zero = ms(tda), ms(zero)
    c_zero_views = ms(zero, "view_seconds") + ms(zero, "encode_seconds")
    zero_agree = np.array([zero[i]["zero_agreement"] for i in ids])

    methods = {"TDA": (P_tda, pred_tda, c_tda), "ZERO": (P_zero, pred_zero, c_zero)}
    if tpt:
        P_tpt = load_probs(tpt_path.with_suffix(".probs.npz"), "final_probs", ids)
        pred_tpt = np.array([tpt[i]["pred"] for i in ids])
        c_tpt = ms(tpt)
        c_tpt_reuse = np.clip(c_tpt - ms(tpt, "view_feature_seconds"), 0, None)
        methods["TPT"] = (P_tpt, pred_tpt, c_tpt)

    points = []   # (name, family, accuracy, mean cost ms)

    def add(name, family, a, cost):
        points.append({"name": name, "family": family, "acc": a, "cost_ms": float(np.mean(cost))})

    # ---- 1. fixed points ----
    print(f"\n=== {spec.name}: {n} images | gate = low CLIP top-2 margin (no labels) ===")
    print("\n1. Fixed points (every image)")
    add("CLIP (cost shown as 1-view TDA's)", "fixed", acc(pred_clip, labels), c_tda)   # CLIP isn't timed per image; TDA is ~CLIP
    for name, (_, pred, cost) in methods.items():
        add(f"{name} alone", "fixed", acc(pred, labels), cost)
    if tda64:
        add("TDA, 64 views", "fixed", acc(np.array([tda64[i]["pred"] for i in ids]), labels), ms(tda64))
    add("avg TDA + ZERO", "fixed", acc((P_tda + P_zero).argmax(1), labels), c_tda + c_zero)
    if tpt:
        add("avg TDA + TPT", "fixed", acc((P_tda + P_tpt).argmax(1), labels), c_tda + c_tpt)
        add("avg ZERO + TPT", "fixed", acc((P_zero + P_tpt).argmax(1), labels), c_zero + c_tpt)
        add("avg TDA + ZERO + TPT", "fixed", acc((P_tda + P_zero + P_tpt).argmax(1), labels),
            c_tda + c_zero + c_tpt)
    for pt in points:
        print(f"  {pt['name']:36s} acc {pt['acc']:.3f} | {pt['cost_ms']:7.1f} ms")
    zero_alone = next(pt["acc"] for pt in points if pt["name"] == "ZERO alone")

    # ---- 2. routing TDA -> E with combination rules ----
    curves = {}
    weights = [0.25, 0.5, 0.75]
    for E in [m for m in methods if m != "TDA"]:
        P_e, pred_e, c_e = methods[E]
        rules = {
            "replace": pred_e,
            "average": (P_tda + P_e).argmax(1),
            "more confident": np.where(P_e.max(1) >= P_tda.max(1), P_e.argmax(1), pred_tda),
        }
        if E == "ZERO":
            rules["average (TDA + ZERO votes)"] = (P_tda + V_zero).argmax(1)
        best_w = {}
        for w in weights:
            best_w[w] = (w * P_e + (1 - w) * P_tda).argmax(1)
        print(f"\n2. Routing TDA -> {E} (TDA runs on every image; {E} on the routed share)")
        header = " | ".join(f"{b:>4.0%}" for b in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0))
        print(f"  {'rule on routed images':30s} | {header} | peak")
        for rule, fused in list(rules.items()) + [(f"weighted w={w} (tuned here)", f) for w, f in best_w.items()]:
            accs, costs = [], []
            for b in BUDGETS:
                route = top_k_mask(gate, b)
                accs.append(acc(np.where(route, fused, pred_tda), labels))
                costs.append(np.mean(c_tda + route * c_e))
            accs, costs = np.array(accs), np.array(costs)
            curves[f"TDA -> {E}, {rule}"] = (accs, costs)
            at = lambda b: accs[np.isclose(BUDGETS, b)][0]
            row = " | ".join(f"{at(b):.3f}" for b in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0))
            print(f"  {rule:30s} | {row} | {accs.max():.3f} at {BUDGETS[accs.argmax()]:.0%}")
            if "tuned" not in rule:
                for b, a, c in zip(BUDGETS, accs, costs):
                    add(f"TDA -> {E} {rule} @{b:.0%}", f"route {E}", a, c)

    # ---- 3. cascade TDA -> ZERO -> TPT ----
    if tpt:
        print("\n3. Cascade TDA -> ZERO -> TPT (answer = average of every method that ran)")
        print("   b1 = share routed to ZERO by CLIP margin; b2 = share of those also sent to TPT,"
              " lowest ZERO vote agreement first")
        print(f"  {'b1':>4} {'b2':>5} | {'acc':>5} | {'cost as run':>11} | {'cost, TPT reuses views':>22}")
        zero_uncertainty = -zero_agree + 1e-6 * -P_zero.max(1)    # ties broken by ZERO's avg confidence
        for b1 in (0.2, 0.3, 0.5, 0.7, 1.0):
            r1 = top_k_mask(gate, b1)
            for b2 in (0.0, 0.25, 0.5, 1.0):
                score2 = np.where(r1, zero_uncertainty, -np.inf)
                r2 = top_k_mask(score2, b1 * b2) & r1
                fused = np.where(r2[:, None], P_tda + P_zero + P_tpt,
                                 np.where(r1[:, None], P_tda + P_zero, P_tda)).argmax(1)
                a = acc(fused, labels)
                cost = c_tda + r1 * c_zero + r2 * c_tpt
                cost_reuse = c_tda + r1 * c_zero + r2 * c_tpt_reuse
                print(f"  {b1:>4.0%} {b2:>5.0%} | {a:.3f} | {cost.mean():9.1f} ms | {cost_reuse.mean():18.1f} ms")
                add(f"cascade b1={b1:.0%} b2={b2:.0%}", "cascade (TPT reuses views)", a, cost_reuse)

    # ---- 4. CAS-style adapt-or-skip ----
    print("\n4. CAS-style adapt-or-skip (gate encodes ZERO's 64 views for every image)")
    cas_signals = {
        "label agreement": np.array([zero[i]["cas_label_agree_excl0"] for i in ids]),
        "probability cosine": np.array([zero[i]["cas_prob_cos_excl0"] for i in ids]),
    }
    for sig_name, sig in cas_signals.items():
        adapt_score = -sig                          # inconsistent views -> adapt
        for fb_name, fb_pred, fb_cost in (("CLIP", pred_clip, np.zeros(n)), ("TDA", pred_tda, c_tda)):
            accs, costs = [], []
            for b in BUDGETS:
                adapt = top_k_mask(adapt_score, b)
                accs.append(acc(np.where(adapt, pred_zero, fb_pred), labels))
                costs.append(np.mean(fb_cost + c_zero_views + adapt * (c_zero - c_zero_views)))
            accs, costs = np.array(accs), np.array(costs)
            label = f"CAS-style ({sig_name}), {fb_name} fallback"
            curves[label] = (accs, costs)
            print(f"  {label:48s} @30% {accs[np.isclose(BUDGETS, .3)][0]:.3f} "
                  f"@50% {accs[np.isclose(BUDGETS, .5)][0]:.3f} | peak {accs.max():.3f} | "
                  f"cost {costs.min():.0f}-{costs.max():.0f} ms")
            for b, a, c in zip(BUDGETS, accs, costs):
                add(f"{label} @{b:.0%}", "CAS-style", a, c)

    # ---- Pareto front ----
    pts = sorted(points, key=lambda q: (q["cost_ms"], -q["acc"]))
    front, best = [], -1.0
    for q in pts:
        if q["acc"] > best + 1e-12:
            front.append(q)
            best = q["acc"]
    print("\nAccuracy-cost Pareto front (no point is both cheaper and more accurate):")
    for q in front:
        print(f"  {q['cost_ms']:7.1f} ms  acc {q['acc']:.3f}  {q['name']}")
    print(f"\nAbove always-ZERO ({zero_alone:.3f}): " + (", ".join(
        f"{q['name']} ({q['acc']:.3f}, {q['cost_ms']:.0f} ms)" for q in front if q["acc"] > zero_alone + 1e-9)
        or "nothing"))

    # ---- outputs ----
    out = R("fusion_cascade", args.out_tag)
    out.write_text(json.dumps({"n": n, "points": points, "pareto": front}, indent=1))
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, (accs, costs) in curves.items():
        if "tuned" in label or "votes" in label:
            continue
        ls = "--" if "CAS" in label else "-"
        ax.plot(costs, accs * 100, ls, marker="o", ms=2.5, lw=1.2, label=label)
    for q in points:
        if q["family"] == "fixed":
            ax.scatter(q["cost_ms"], q["acc"] * 100, marker="*", s=90, zorder=5)
            ax.annotate(q["name"], (q["cost_ms"], q["acc"] * 100), fontsize=7,
                        xytext=(4, 3), textcoords="offset points")
        elif q["family"].startswith("cascade"):
            ax.scatter(q["cost_ms"], q["acc"] * 100, marker="x", s=18, color="black", zorder=4)
    ax.scatter([], [], marker="x", color="black", label="cascade TDA -> ZERO -> TPT (views reused)")
    ax.set_xscale("log")
    ax.set_xlabel("Mean cost per image (ms, log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{spec.name}: routing, fusion, cascade and CAS-style baselines (n={n:,})")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=150)
    print(f"\nSaved {out} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()