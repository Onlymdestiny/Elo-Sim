"""
validate_real_data.py
Real-data validation for the Match Quality Optimizer.

Two modes:

  Simulation mode (default):
    Train RF on simulated data (elo.py), test on a real CSV.
    Tests whether the simulation generalises to real ranked games.
      python validate_real_data.py --csv real_matches.csv

  Real-data mode (--train-csv / --test-csv):
    Train RF directly on one real CSV, test on a second held-out real CSV.
    Tests whether pre-match MMR features predict real match outcomes.
      python validate_real_data.py --train-csv real_matches.csv --test-csv real_matches2.csv

Other options:
  --load-model model.pkl   Skip training, load a saved model
  --save-model model.pkl   Save the trained model after training
  --system trueskill|openskill
  --n-estimators N
"""

import argparse
import csv
import math
import os
import random
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
)

try:
    from elo import (
        player,
        _extract_features,
        calculate_mmr_change_trueskill,
        calculate_mmr_change_openskill,
        LABEL_NAMES, TS_BETA, OS_BETA,
        server,
    )
    from model import train_fairness_model, save_model, load_model, MODEL_NAMES
except ImportError:
    print("ERROR: Cannot import elo.py / model.py — make sure they are in the same directory.", file=sys.stderr)
    sys.exit(1)

# ── Paths ──────────────────────────────────────────────────────────────────────

DIR       = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(DIR)
DATA_DIR  = os.path.join(REPO_ROOT, "data")
CSV_PATH  = os.path.join(DATA_DIR, "real_matches.csv")
PLOTS_DIR = os.path.join(REPO_ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Dark-theme palette (matches ml_analysis.py) ────────────────────────────────

DARK_BG  = "#12121E"
C_ACCENT = "#74C7EC"
C_GREEN  = "#A6E3A1"
C_YELLOW = "#F9E2AF"
C_RED    = "#F38BA8"
PALETTE  = [C_ACCENT, C_GREEN, C_YELLOW, C_RED, "#CBA6F7", "#FAB387"]


def _apply_dark(fig, axes):
    fig.patch.set_facecolor(DARK_BG)
    ax_list = axes if hasattr(axes, "__iter__") else [axes]
    for ax in ax_list:
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color(C_ACCENT)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")


# ── Model training ─────────────────────────────────────────────────────────────

def build_model_from_real(train_rows: list[dict], beta: float,
                          n_estimators: int, model_name: str = "rf",
                          feature_set: str = "baseline"):
    """Train selected classifier directly on real match data.

    Features are extracted identically to the simulation path so the two modes
    are directly comparable.  Labels are derived from outcome signals
    (duration / gold / kill voting) — the same heuristic used throughout.
    """
    X, y = [], []
    skipped = 0
    for row in train_rows:
        feats  = row_to_features(row, beta, feature_set=feature_set)
        label  = _derive_actual_label(row)
        if feats is None or label is None:
            skipped += 1
            continue
        X.append(feats)
        y.append(LABEL_ORDER[label])

    if not X:
        print("ERROR: no valid training rows extracted from --train-csv.", file=sys.stderr)
        sys.exit(1)

    counts = {n: y.count(i) for i, n in enumerate(LABEL_NAMES)}
    print(f"  Real training set: {len(X)} matches  (skipped {skipped})")
    print(f"  Label distribution: " + "  ".join(f"{k}: {v}" for k, v in counts.items()))
    print(f"  Training {model_name.upper()} (n_estimators={n_estimators})…")
    model, report = train_fairness_model(X, y, model_name=model_name, n_estimators=n_estimators)
    print("  Training report (test 20%):")
    for line in report.splitlines():
        print("   ", line)
    return model


def build_model(system: str, warmup: int, collect: int, n_estimators: int,
                model_name: str = "rf"):
    rating_func = calculate_mmr_change_trueskill if system == "trueskill" else calculate_mmr_change_openskill
    beta        = TS_BETA                        if system == "trueskill" else OS_BETA

    print(f"  Building {system} + {model_name.upper()} model — warmup={warmup}, collect={collect}…")
    srv = server()
    for i in range(warmup):
        srv.iterate(rating_func=rating_func, beta=beta)
        if (i + 1) % 100 == 0:
            print(f"    warm-up {i+1}/{warmup}", end="\r")
    print()

    X, y = srv.collect_training_data(n_iters=collect, rating_func=rating_func, beta=beta)
    print(f"  Training {model_name.upper()} ({n_estimators} trees, {len(X):,} samples)…")
    model, report = train_fairness_model(X, y, model_name=model_name, n_estimators=n_estimators)
    print("  Training report (test 20%):")
    for line in report.splitlines():
        print("   ", line)
    return model, beta


# ── CSV loading ────────────────────────────────────────────────────────────────

def load_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        print(f"ERROR: CSV not found at {path}", file=sys.stderr)
        sys.exit(1)

    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"  Loaded {len(rows):,} rows from {path}")
    return rows


# ── Feature extraction from CSV row ───────────────────────────────────────────

def row_to_features(row: dict, beta: float,
                    feature_set: str = "baseline") -> list[float] | None:
    """Build real player objects from CSV columns and extract feature vector.

    baseline (8-D):   [mean_A, mean_B, mmr_gap, var_A, var_B, sigma_A, sigma_B, p_win]
    extended (14-D):  baseline + [avg_wr_a, avg_wr_b, winrate_gap,
                                  hot_streak_diff, fresh_blood_diff, inactive_diff]

    Extended features require real-CSV-only columns (avg_winrate_*, *_count_*),
    so extended is only valid in real-data mode.
    """
    try:
        mmrs_a  = [float(row[f"mmr_a{i}"]) for i in range(1, 6)]
        mmrs_b  = [float(row[f"mmr_b{i}"]) for i in range(1, 6)]
        sig_a   = float(row["avg_sigma_a"])
        sig_b   = float(row["avg_sigma_b"])
    except (KeyError, ValueError):
        return None

    team_a = [player() for _ in range(5)]
    team_b = [player() for _ in range(5)]
    for p, m in zip(team_a, mmrs_a):
        p.mmr   = m
        p.sigma = sig_a
    for p, m in zip(team_b, mmrs_b):
        p.mmr   = m
        p.sigma = sig_b
    feats = list(_extract_features(team_a, team_b, beta))

    if feature_set == "baseline":
        return feats

    # Extended features — fail soft if the CSV wasn't collected with status columns
    try:
        avg_wr_a = float(row["avg_winrate_a"])
        avg_wr_b = float(row["avg_winrate_b"])
        hs_diff  = int(row["hot_streak_count_a"])  - int(row["hot_streak_count_b"])
        fb_diff  = int(row["fresh_blood_count_a"]) - int(row["fresh_blood_count_b"])
        in_diff  = int(row["inactive_count_a"])    - int(row["inactive_count_b"])
    except (KeyError, ValueError):
        return None

    feats.extend([
        avg_wr_a,
        avg_wr_b,
        abs(avg_wr_a - avg_wr_b),
        hs_diff,
        fb_diff,
        in_diff,
    ])
    return feats


def predict_row(row: dict, model, beta: float,
                feature_set: str = "baseline") -> tuple[int, str] | None:
    feats = row_to_features(row, beta, feature_set=feature_set)
    if feats is None:
        return None
    proba     = model.predict_proba([feats])[0]
    label_idx = int(model.predict([feats])[0])
    fairness  = round(100 * (proba[0] + 0.5 * proba[1]))
    return fairness, LABEL_NAMES[label_idx]


# ── Analysis ───────────────────────────────────────────────────────────────────

LABEL_ORDER = {"Close": 0, "Competitive": 1, "Stomp": 2}


def _derive_actual_label(row: dict) -> str | None:
    """Compute actual match quality label from outcome signals when the
    'actual_label' column is absent from the CSV (three-signal voting)."""
    try:
        dur  = float(row["game_duration_s"])
        gold = float(row["gold_diff_end"])
        kill = float(row["kill_diff"])
    except (KeyError, ValueError):
        return None

    def _vote_dur(s):  return "Stomp" if s < 1320  else ("Close" if s > 2100   else "Competitive")
    def _vote_gld(g):  return "Stomp" if g > 10000 else ("Close" if g < 4000   else "Competitive")
    def _vote_kll(k):  return "Stomp" if k > 15    else ("Close" if k < 7      else "Competitive")

    votes = [_vote_dur(dur), _vote_gld(gold), _vote_kll(kill)]
    return max(votes, key=lambda v: LABEL_ORDER[v])


def run_validation(rows: list[dict], model, beta: float,
                   feature_set: str = "baseline") -> dict:
    fairness_scores, pred_labels, actual_labels = [], [], []
    durations, gold_diffs, kill_diffs = [], [], []
    skipped = 0

    for row in rows:
        result = predict_row(row, model, beta, feature_set=feature_set)
        if result is None:
            skipped += 1
            continue
        score, label = result

        # Use CSV column if present, otherwise derive from outcome signals
        actual = row.get("actual_label", "")
        if actual not in LABEL_NAMES:
            actual = _derive_actual_label(row)
        if actual is None or actual not in LABEL_NAMES:
            skipped += 1
            continue

        fairness_scores.append(score)
        pred_labels.append(label)
        actual_labels.append(actual)
        durations.append(float(row["game_duration_s"]))
        gold_diffs.append(float(row["gold_diff_end"]))
        kill_diffs.append(float(row["kill_diff"]))

    n = len(fairness_scores)
    print(f"\n  Predicted {n} matches  (skipped {skipped} due to missing data)")

    # Spearman correlations — expected direction:
    #   duration:  higher fairness → longer (close) games → positive ρ
    #   gold_diff: higher fairness → smaller gap → negative ρ
    #   kill_diff: higher fairness → smaller gap → negative ρ
    rho_dur,  p_dur  = spearmanr(fairness_scores, durations)
    rho_gold, p_gold = spearmanr(fairness_scores, gold_diffs)
    rho_kill, p_kill = spearmanr(fairness_scores, kill_diffs)

    acc = accuracy_score(actual_labels, pred_labels)

    print(f"\n  Spearman correlations (fairness score vs outcome proxies):")
    print(f"    vs game_duration_s : ρ = {rho_dur:+.4f}  (p={p_dur:.3g})  [expect +]")
    print(f"    vs gold_diff_end   : ρ = {rho_gold:+.4f}  (p={p_gold:.3g})  [expect -]")
    print(f"    vs kill_diff       : ρ = {rho_kill:+.4f}  (p={p_kill:.3g})  [expect -]")
    print(f"\n  Label accuracy (predicted vs actual): {acc:.4f}  ({acc*100:.1f}%)")
    print()
    print(classification_report(actual_labels, pred_labels, target_names=LABEL_NAMES))

    return {
        "n": n, "skipped": skipped,
        "rho_duration": rho_dur,  "p_duration": p_dur,
        "rho_gold":     rho_gold, "p_gold":     p_gold,
        "rho_kill":     rho_kill, "p_kill":     p_kill,
        "accuracy":     acc,
        "fairness_scores": fairness_scores,
        "pred_labels":     pred_labels,
        "actual_labels":   actual_labels,
        "durations":       durations,
        "gold_diffs":      gold_diffs,
        "kill_diffs":      kill_diffs,
    }


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_scatter_correlations(results: dict, prefix: str = ""):
    """3-panel scatter: fairness score vs each outcome proxy."""
    fs  = results["fairness_scores"]
    dur = results["durations"]
    gd  = results["gold_diffs"]
    kd  = results["kill_diffs"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    _apply_dark(fig, axes)

    panels = [
        (dur, "Game Duration (s)",  results["rho_duration"], C_ACCENT, "expect +"),
        (gd,  "Gold Diff at End",   results["rho_gold"],     C_GREEN,  "expect −"),
        (kd,  "Kill Diff",          results["rho_kill"],     C_YELLOW, "expect −"),
    ]

    for ax, (vals, xlabel, rho, color, note) in zip(axes, panels):
        ax.scatter(fs, vals, alpha=0.35, s=12, color=color)
        ax.set_xlabel("Fairness Score", color="white")
        ax.set_ylabel(xlabel, color="white")
        ax.set_title(f"ρ = {rho:+.3f}  ({note})", color=C_ACCENT, fontsize=11)
        # Regression line
        m, b = np.polyfit(fs, vals, 1)
        xs = np.linspace(min(fs), max(fs), 100)
        ax.plot(xs, m * xs + b, color="white", lw=1.5, alpha=0.6, linestyle="--")

    fig.suptitle("Fairness Score vs Real Match Outcome Proxies  (Spearman ρ)",
                 color="white", fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{prefix}validation_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_confusion_matrix(results: dict, prefix: str = ""):
    """Confusion matrix: predicted label vs actual label."""
    y_true = results["actual_labels"]
    y_pred = results["pred_labels"]

    cm   = confusion_matrix(y_true, y_pred, labels=LABEL_NAMES)
    fig, ax = plt.subplots(figsize=(6, 5))
    _apply_dark(fig, ax)

    cmap = LinearSegmentedColormap.from_list("dark_blue", ["#12121E", "#1A3A5E", "#74C7EC"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
    disp.plot(ax=ax, cmap=cmap, colorbar=False)
    ax.set_title(f"Predicted vs Actual Label  (acc={results['accuracy']:.3f})",
                 color=C_ACCENT, fontsize=12)
    for text in disp.text_.ravel():
        text.set_color("white")
        text.set_fontsize(12)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{prefix}validation_confusion.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_label_distribution(results: dict, prefix: str = ""):
    """Side-by-side bar chart: actual vs predicted label counts."""
    counts_actual = {lb: results["actual_labels"].count(lb) for lb in LABEL_NAMES}
    counts_pred   = {lb: results["pred_labels"].count(lb)   for lb in LABEL_NAMES}

    x = np.arange(len(LABEL_NAMES))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    _apply_dark(fig, ax)

    bars_a = ax.bar(x - w/2, [counts_actual[lb] for lb in LABEL_NAMES],
                    w, label="Actual", color=C_ACCENT, alpha=0.9)
    bars_p = ax.bar(x + w/2, [counts_pred[lb]   for lb in LABEL_NAMES],
                    w, label="Predicted", color=C_GREEN, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_NAMES, color="white", fontsize=12)
    ax.set_ylabel("Count", color="white")
    ax.set_title("Label Distribution: Actual vs Predicted", color=C_ACCENT)
    ax.legend(facecolor="#1C1C38", labelcolor="white")

    for bar in list(bars_a) + list(bars_p):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(int(bar.get_height())), ha="center", va="bottom",
                fontsize=10, color="white")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{prefix}validation_label_dist.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_fairness_histogram(results: dict, prefix: str = ""):
    """Histogram of predicted fairness scores split by actual label."""
    fs     = np.array(results["fairness_scores"])
    actual = results["actual_labels"]

    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_dark(fig, ax)

    colors = {"Close": C_GREEN, "Competitive": C_YELLOW, "Stomp": C_RED}
    for lb in LABEL_NAMES:
        subset = [fs[i] for i, a in enumerate(actual) if a == lb]
        if subset:
            ax.hist(subset, bins=20, alpha=0.65, label=lb,
                    color=colors[lb], density=True)

    ax.set_xlabel("Fairness Score", color="white")
    ax.set_ylabel("Density", color="white")
    ax.set_title("Fairness Score Distribution by Actual Match Label", color=C_ACCENT)
    ax.legend(facecolor="#1C1C38", labelcolor="white")
    ax.grid(color="#333355", linestyle="--", alpha=0.4)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{prefix}validation_fairness_hist.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Validate fairness model on real match data")

    # Real-data mode
    parser.add_argument("--train-csv",    type=str, default="",
                        help="Train RF on this real CSV (real-data mode)")
    parser.add_argument("--test-csv",     type=str, default="",
                        help="Test on this held-out real CSV (real-data mode)")

    # Simulation mode
    parser.add_argument("--csv",          type=str, default=CSV_PATH,
                        help="Test CSV when using simulation mode (default: real_matches.csv)")
    parser.add_argument("--system",       choices=["trueskill", "openskill"], default="trueskill")
    parser.add_argument("--warmup",       type=int, default=500)
    parser.add_argument("--collect",      type=int, default=100)

    # Shared
    parser.add_argument("--load-model",   type=str, default="",
                        help="Load pre-trained model .pkl (skips training)")
    parser.add_argument("--save-model",   type=str, default="",
                        help="Save trained model to .pkl after training")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--model",        choices=list(MODEL_NAMES), default="rf",
                        help="Which classifier to train: rf | xgb | logreg")
    parser.add_argument("--features",     choices=["baseline", "extended"], default="baseline",
                        help="baseline = 8-D MMR/sigma; extended = +winrate +status flags (real-data only)")
    args = parser.parse_args()

    beta = TS_BETA

    # Guard: extended features need columns that only exist in real CSVs
    if args.features == "extended" and not args.train_csv and not args.load_model:
        print("ERROR: --features extended requires --train-csv (simulation has no winrate/status columns).",
              file=sys.stderr)
        sys.exit(1)

    # ── Load or train model ──────────────────────────────────────────────────
    if args.load_model:
        if not os.path.exists(args.load_model):
            print(f"ERROR: model file not found: {args.load_model}", file=sys.stderr)
            sys.exit(1)
        model, beta, mode = load_model(args.load_model)

    elif args.train_csv:
        # ── Real-data mode ───────────────────────────────────────────────────
        if not args.test_csv:
            print("ERROR: --train-csv requires --test-csv.", file=sys.stderr)
            sys.exit(1)
        print(f"\n=== Step 1: Train on real data ({args.train_csv}) ===")
        print(f"  Feature set: {args.features}")
        train_rows = load_csv(args.train_csv)
        model = build_model_from_real(train_rows, beta, args.n_estimators,
                                      model_name=args.model, feature_set=args.features)
        mode  = f"real-{args.model}-{args.features}"

    else:
        # ── Simulation mode ──────────────────────────────────────────────────
        print(f"\n=== Step 1: Train {args.system} simulation model ===")
        model, beta = build_model(args.system, args.warmup, args.collect,
                                  args.n_estimators, model_name=args.model)
        mode = f"simulation-{args.system}-{args.model}"

    if args.save_model:
        save_model(model, args.save_model, beta=beta, system=mode)

    # ── Load test CSV ────────────────────────────────────────────────────────
    test_path = args.test_csv if args.train_csv else args.csv
    print(f"\n=== Step 2: Load test data ({test_path}) ===")
    test_rows = load_csv(test_path)

    # ── Run validation ───────────────────────────────────────────────────────
    print(f"\n=== Step 3: Run predictions + compute metrics ===")
    results = run_validation(test_rows, model, beta, feature_set=args.features)

    # ── Plots ────────────────────────────────────────────────────────────────
    # Use a prefix so real-data and simulation runs don't overwrite each other
    plot_prefix = f"{'real' if args.train_csv else 'sim'}_{args.model}_{args.features}_"
    print(f"\n=== Step 4: Generate plots ===")
    plot_scatter_correlations(results, prefix=plot_prefix)
    plot_confusion_matrix(results, prefix=plot_prefix)
    plot_label_distribution(results, prefix=plot_prefix)
    plot_fairness_histogram(results, prefix=plot_prefix)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Mode                 : {mode}")
    print(f"  Test matches         : {results['n']}")
    print(f"  Label accuracy       : {results['accuracy']:.4f}  ({results['accuracy']*100:.1f}%)")
    print(f"  ρ vs duration        : {results['rho_duration']:+.4f}")
    print(f"  ρ vs gold diff       : {results['rho_gold']:+.4f}")
    print(f"  ρ vs kill diff       : {results['rho_kill']:+.4f}")
    print(f"  Plots saved to       : {PLOTS_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
