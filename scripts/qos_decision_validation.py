#!/usr/bin/env python3
"""Decision-region validation of the twin's QoS safety signal.

Global R^2 on delay and loss says nothing about the quantity a controller
actually consumes, which is the BOOLEAN

    is_safe = (predicted_loss <= max_loss_pkts) AND (predicted_delay <= delay_budget)

and, in particular, nothing about the asymmetry of its errors: predicting
"safe" for a load that is in fact unsafe puts traffic on a data plane that
will drop it, whereas the opposite error only wastes energy.

This script evaluates that boolean on HELD-OUT MEASUREMENT RUNS, using the
same grouped split the trainer uses, and reports:

  A. the safe/unsafe confusion matrix, whole range and in the decision band;
  B. false-safe rate, both as P(pred safe | truly unsafe) and as the
     operationally relevant P(truly unsafe | pred safe);
  C. a bootstrap interval for the derived QoS threshold, resampling RUNS;
  D. sensitivity of that threshold to the 200 us / 5-packet budgets.

Two input conventions are reported side by side, because they answer different
questions:
  measured  — the model is fed the run's actual TX features. Validates the
              MODEL.
  twin      — the model is fed [load, load] as _lite_X does at inference, which
              is what the controller consumes. Validates the TWIN AS DEPLOYED.

Usage:
    python scripts/qos_decision_validation.py [--models models_nested]
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit

DATA = "data/processed/features_with_run.csv"
LOSS = "gtpu_packets_dn__packets_lost_delta"
DELAY = "downlink_one_way_delay_distribution__weighted_mean_delay_us"
LOAD_KBITS = "gtpu_kbitss_dn__kbits_tx_s"
LITE = ["gtpu_kbitss_dn__kbits_tx_s", "gtpu_kbitss_ngran__gtpu_kbits_tx_s"]

DELAY_BUDGET_US = 200.0
MAX_LOSS_PKTS = 5.0
GBPS_TO_KBITS = 1_000_000.0


def load_lite(models_dir: Path, variant: str, target: str):
    p = models_dir / "layer1" / f"{variant}__{target}__lite.pkl"
    with open(p, "rb") as f:
        return pickle.load(f)


def confusion(truth_safe: np.ndarray, pred_safe: np.ndarray) -> dict:
    tp = int((truth_safe & pred_safe).sum())        # safe, called safe
    tn = int((~truth_safe & ~pred_safe).sum())      # unsafe, called unsafe
    fs = int((~truth_safe & pred_safe).sum())       # FALSE SAFE - the dangerous one
    fu = int((truth_safe & ~pred_safe).sum())       # false unsafe - merely wasteful
    n = tp + tn + fs + fu
    n_unsafe = int((~truth_safe).sum())
    n_pred_safe = int(pred_safe.sum())
    return {
        "n": n,
        "n_truly_unsafe": n_unsafe,
        "n_pred_safe": n_pred_safe,
        "true_safe": tp, "true_unsafe": tn,
        "false_safe": fs, "false_unsafe": fu,
        "accuracy": (tp + tn) / n if n else float("nan"),
        # P(called safe | truly unsafe): how much of the danger is missed
        "false_safe_rate_given_unsafe": fs / n_unsafe if n_unsafe else float("nan"),
        # P(truly unsafe | called safe): how often acting on "safe" hurts
        "unsafe_given_called_safe": fs / n_pred_safe if n_pred_safe else float("nan"),
        "false_unsafe_rate_given_safe": (
            fu / int(truth_safe.sum()) if truth_safe.sum() else float("nan")
        ),
    }


def derive_qos_limit(loss_model, delay_model,
                     max_loss=MAX_LOSS_PKTS, delay_budget=DELAY_BUDGET_US,
                     sweep_max_gbps=0.5, res_mbps=1.0) -> float:
    """Replicates threshold_derivation.py: last swept load where USR is safe."""
    n = int(sweep_max_gbps * 1000 / res_mbps)
    loads = np.linspace(res_mbps / 1000, sweep_max_gbps, n)
    X = np.column_stack([loads * GBPS_TO_KBITS, loads * GBPS_TO_KBITS])
    safe = (loss_model.predict(X) <= max_loss) & (delay_model.predict(X) <= delay_budget)
    if safe.all():
        return float(sweep_max_gbps)
    if not safe.any():
        return 0.0
    first_unsafe = int(np.argmax(~safe))
    return float(loads[first_unsafe - 1]) if first_unsafe > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models_nested")
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--out", default="reports_nested/qos_decision_validation.json")
    args = ap.parse_args()

    models_dir = Path(args.models)
    params = yaml.safe_load(open("params.yaml"))["train"]
    rs = params["random_state"]

    df = pd.read_csv(DATA)
    usr = df[df["is_dpdk"] == 0].copy()

    gss = GroupShuffleSplit(n_splits=1, test_size=params["test_split"], random_state=rs)
    tr_idx, te_idx = next(gss.split(usr, groups=usr["run_dir"]))
    train, test = usr.iloc[tr_idx].copy(), usr.iloc[te_idx].copy()

    print(f"USR: {len(usr)} samples / {usr['run_dir'].nunique()} runs")
    print(f"  train {len(train)} / {train['run_dir'].nunique()} runs   "
          f"test {len(test)} / {test['run_dir'].nunique()} runs\n")

    loss_m = load_lite(models_dir, "usr_full", LOSS)
    delay_m = load_lite(models_dir, "usr_full", DELAY)

    ok = test[LOSS].notna() & test[DELAY].notna()
    te = test[ok].copy()
    truth_safe = (te[LOSS].values <= MAX_LOSS_PKTS) & (te[DELAY].values <= DELAY_BUDGET_US)

    load_mbps = te[LOAD_KBITS].values / 1000.0
    X_measured = te[LITE].values
    X_twin = np.column_stack([te[LOAD_KBITS].values, te[LOAD_KBITS].values])

    results = {
        "config": {
            "models_dir": str(models_dir),
            "delay_budget_us": DELAY_BUDGET_US,
            "max_loss_pkts": MAX_LOSS_PKTS,
            "test_runs": int(te["run_dir"].nunique()),
            "test_samples": int(len(te)),
            "truly_unsafe_frac": float((~truth_safe).mean()),
        }
    }

    band = (load_mbps >= 50.0) & (load_mbps <= 250.0)
    print(f"Decision band 50-250 Mbps holds {band.sum()} of {len(te)} test samples "
          f"({band.mean():.1%}); {(~truth_safe).mean():.1%} of all test samples are "
          f"truly unsafe.\n")

    for tag, X in (("measured", X_measured), ("twin", X_twin)):
        pred_safe = ((loss_m.predict(X) <= MAX_LOSS_PKTS) &
                     (delay_m.predict(X) <= DELAY_BUDGET_US))
        whole = confusion(truth_safe, pred_safe)
        inband = confusion(truth_safe[band], pred_safe[band])
        results[f"confusion_{tag}"] = {"all": whole, "band_50_250_mbps": inband}

        print(f"--- inputs: {tag} " + "-" * 46)
        for name, c in (("whole range", whole), ("50-250 Mbps", inband)):
            print(f"  {name:<12} n={c['n']:<5} acc={c['accuracy']:.4f}   "
                  f"false-safe {c['false_safe']:<4} "
                  f"(P(safe|unsafe)={c['false_safe_rate_given_unsafe']:.4f}, "
                  f"P(unsafe|safe)={c['unsafe_given_called_safe']:.4f})   "
                  f"false-unsafe {c['false_unsafe']}")
        print()

    # ── B2. is the classifier informative WITHIN a band, or only across? ────
    from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
    pred_twin = ((loss_m.predict(X_twin) <= MAX_LOSS_PKTS) &
                 (delay_m.predict(X_twin) <= DELAY_BUDGET_US))
    bands = {"all": np.ones(len(te), bool),
             "lt_50_mbps": load_mbps < 50,
             "50_250_mbps": (load_mbps >= 50) & (load_mbps <= 250),
             "gt_250_mbps": load_mbps > 250}
    by_band = {}
    print("Informativeness by load band (twin inputs). A balanced accuracy of")
    print("0.5 means the verdict carries no information beyond the base rate:\n")
    print(f"  {'band':<13}{'n':>5}{'safe rate':>11}{'majority':>10}"
          f"{'acc':>8}{'bal.acc':>9}{'MCC':>8}")
    for name, m in bands.items():
        t, pr = truth_safe[m], pred_twin[m]
        if m.sum() == 0:
            continue
        base = float(max(t.mean(), 1 - t.mean()))
        ba = (float(balanced_accuracy_score(t, pr)) if len(set(t)) > 1 else float("nan"))
        mcc = (float(matthews_corrcoef(t, pr))
               if len(set(t)) > 1 and len(set(pr)) > 1 else float("nan"))
        by_band[name] = {"n": int(m.sum()), "safe_base_rate": float(t.mean()),
                         "majority_baseline": base, "accuracy": float((t == pr).mean()),
                         "balanced_accuracy": ba, "mcc": mcc}
        print(f"  {name:<13}{m.sum():>5}{t.mean():>11.3f}{base:>10.3f}"
              f"{(t == pr).mean():>8.3f}{ba:>9.3f}{mcc:>8.3f}")
    results["informativeness_by_band"] = by_band

    # which constraint binds in the ground truth?
    lb = te[LOSS].values > MAX_LOSS_PKTS
    db = te[DELAY].values > DELAY_BUDGET_US
    results["unsafe_attribution"] = {
        "loss_only": int((lb & ~db).sum()), "delay_only": int((~lb & db).sum()),
        "both": int((lb & db).sum()), "safe": int((~lb & ~db).sum())}
    print(f"\n  ground-truth unsafe: loss-only {int((lb & ~db).sum())}, "
          f"delay-only {int((~lb & db).sum())}, both {int((lb & db).sum())}\n")

    # ── C. bootstrap the derived threshold over runs ────────────────────────
    print("Bootstrapping the derived QoS limit over training runs "
          f"({args.n_boot} resamples) ...")
    from sklearn.base import clone
    runs = train["run_dir"].unique()
    rng = np.random.default_rng(rs)
    point = derive_qos_limit(loss_m, delay_m)
    boots = []
    for b in range(args.n_boot):
        pick = rng.choice(runs, size=len(runs), replace=True)
        sub = pd.concat([train[train["run_dir"] == r] for r in pick], ignore_index=True)
        sub = sub[sub[LOSS].notna() & sub[DELAY].notna()]
        if len(sub) < 50:
            continue
        Xb = sub[LITE].values
        lm = clone(loss_m).fit(Xb, sub[LOSS].values)
        dm = clone(delay_m).fit(Xb, sub[DELAY].values)
        boots.append(derive_qos_limit(lm, dm))
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    results["qos_limit_bootstrap"] = {
        "point_mbps": point * 1000,
        "ci95_mbps": [float(lo * 1000), float(hi * 1000)],
        "median_mbps": float(np.median(boots) * 1000),
        "n_resamples": int(len(boots)),
        "unique_values_mbps": sorted({round(float(v * 1000), 1) for v in boots}),
    }
    print(f"  point estimate {point*1000:.0f} Mbps; "
          f"bootstrap median {np.median(boots)*1000:.0f}, "
          f"95% CI [{lo*1000:.0f}, {hi*1000:.0f}] Mbps "
          f"over {len(boots)} resamples\n")

    # ── D. sensitivity to the QoS budgets ───────────────────────────────────
    print("Sensitivity of the derived QoS limit to the budgets (Mbps):")
    delays = [100.0, 150.0, 200.0, 300.0, 500.0]
    losses = [0.0, 1.0, 5.0, 10.0, 20.0]
    grid = {}
    header = "    loss\\delay " + "".join(f"{d:>8.0f}" for d in delays)
    print(header)
    for L in losses:
        row = []
        for D in delays:
            v = derive_qos_limit(loss_m, delay_m, max_loss=L, delay_budget=D) * 1000
            row.append(v)
        grid[str(L)] = {str(d): float(v) for d, v in zip(delays, row)}
        print(f"    {L:>10.0f} " + "".join(f"{v:>8.0f}" for v in row))
    results["budget_sensitivity_mbps"] = grid

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWritten -> {out}")


if __name__ == "__main__":
    main()
