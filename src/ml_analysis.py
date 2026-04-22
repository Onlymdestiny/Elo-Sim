"""
ml_analysis.py
Checkpoint 2: Comprehensive ML analysis for Match Quality Optimizer.

Runs:
  1. Multi-model comparison (RF, GBT, SVM, MLP, LogReg)
  2. Ablation study  — 1 feature (p_win only) vs. 8 features
  3. Feature importance plot (RF mean decrease in impurity + permutation)
  4. TrueSkill vs. OpenSkill matchmaking quality comparison
  5. Confusion matrices for best model
  6. Saves all plots to  ./plots/  for use in slides
"""

import os
import math
import random
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import LinearSegmentedColormap

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, accuracy_score
)
from sklearn.inspection import permutation_importance

# ── import simulation helpers ────────────────────────────────────────────────
from elo import (
    server,
    calculate_mmr_change_trueskill,
    calculate_mmr_change_openskill,
    _extract_features, _label_match,
    LABEL_NAMES, TS_BETA, OS_BETA,
    player as Player,
)
from model import train_fairness_model, predict_fairness

PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

FEATURE_NAMES = [
    "mean_MMR_A", "mean_MMR_B", "MMR_gap",
    "var_A", "var_B", "sigma_A", "sigma_B", "p_win",
]

# ═══════════════════════════════════════════════════════════════════════════════
# 0.  Palette helper
# ═══════════════════════════════════════════════════════════════════════════════
DARK_BG  = "#12121E"
C_ACCENT = "#74C7EC"
C_GREEN  = "#A6E3A1"
C_YELLOW = "#F9E2AF"
C_RED    = "#F38BA8"
PALETTE  = [C_ACCENT, C_GREEN, C_YELLOW, C_RED,
            "#CBA6F7", "#FAB387", "#89DCEB"]


def _apply_dark(fig, ax_or_axes):
    """Apply a consistent dark theme to a figure."""
    fig.patch.set_facecolor(DARK_BG)
    axes = [ax_or_axes] if hasattr(ax_or_axes, "set_facecolor") else ax_or_axes
    for ax in axes:
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color(C_ACCENT)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Simulate & collect data
# ═══════════════════════════════════════════════════════════════════════════════

def build_dataset(rating_func, beta, warmup=1000, collect=200, label=""):
    """Return (X_arr, y_arr) from a fresh server after warmup iterations."""
    print(f"  [{label}] warm-up ({warmup} iters)…", flush=True)
    srv = server()
    for i in range(warmup):
        srv.iterate(rating_func=rating_func, beta=beta)
        if (i + 1) % 200 == 0:
            print(f"    warm-up {i+1}/{warmup}", flush=True)

    print(f"  [{label}] collecting data ({collect} iters)…", flush=True)
    X, y = srv.collect_training_data(
        n_iters=collect, rating_func=rating_func, beta=beta
    )
    return np.array(X), np.array(y), srv


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Multi-model comparison
# ═══════════════════════════════════════════════════════════════════════════════

CLASSIFIERS = {
    "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_SEED),
    "Logistic Regression":  LogisticRegression(max_iter=500, random_state=RANDOM_SEED),
    "SVM (RBF)":            SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED),
    "MLP (256-128)":        MLPClassifier(hidden_layer_sizes=(256, 128),
                                          max_iter=300, random_state=RANDOM_SEED),
}


def compare_models(X, y):
    """Train all classifiers, return dict of results."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    results = {}
    for name, clf in CLASSIFIERS.items():
        print(f"    training {name}…", flush=True)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        report = classification_report(
            y_te, y_pred, target_names=LABEL_NAMES, output_dict=True
        )
        results[name] = {
            "model": clf,
            "accuracy": acc,
            "report": report,
            "y_pred": y_pred,
            "y_test": y_te,
        }
        print(f"      {name}: accuracy={acc:.4f}", flush=True)
    return results, X_te, y_te


def plot_model_comparison(results, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _apply_dark(fig, axes)

    names = list(results.keys())
    accs  = [results[n]["accuracy"] for n in names]
    # macro-avg F1
    f1s   = [results[n]["report"]["macro avg"]["f1-score"] for n in names]

    x = np.arange(len(names))
    w = 0.35

    ax = axes[0]
    bars_a = ax.bar(x - w/2, accs, w, label="Accuracy", color=C_ACCENT, alpha=0.9)
    bars_f = ax.bar(x + w/2, f1s,  w, label="Macro F1",  color=C_GREEN,  alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9, color="white")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", color="white")
    ax.set_title("Model Comparison — Accuracy & F1", color=C_ACCENT)
    ax.legend(facecolor="#1C1C38", labelcolor="white")
    for bar in list(bars_a) + list(bars_f):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom",
                fontsize=7.5, color="white")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

    # per-class F1 for best model
    best_name = max(results, key=lambda n: results[n]["accuracy"])
    best_rep  = results[best_name]["report"]
    classes   = LABEL_NAMES
    per_f1    = [best_rep[c]["f1-score"] for c in classes]

    ax2 = axes[1]
    bars2 = ax2.bar(classes, per_f1, color=[C_GREEN, C_YELLOW, C_RED], alpha=0.9)
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel("F1 Score", color="white")
    ax2.set_title(f"Per-class F1  ({best_name})", color=C_ACCENT)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{bar.get_height():.3f}", ha="center", va="bottom",
                 fontsize=9, color="white")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return best_name


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Ablation study
# ═══════════════════════════════════════════════════════════════════════════════

def ablation_study(X, y, save_path):
    """Train RF on subsets of features; compare accuracy."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    feature_subsets = {
        "p_win only\n(1 feature)":        [7],
        "p_win + gap\n(2 features)":       [2, 7],
        "p_win + μ\n(3 features)":         [0, 1, 7],
        "p_win + μ + var\n(5 features)":   [0, 1, 2, 4, 7],
        "All 8 features":                  list(range(8)),
    }

    accs = {}
    for label, idxs in feature_subsets.items():
        clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
        clf.fit(X_tr[:, idxs], y_tr)
        accs[label] = accuracy_score(y_te, clf.predict(X_te[:, idxs]))
        print(f"    Ablation [{label.replace(chr(10), ' ')}]: {accs[label]:.4f}", flush=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_dark(fig, ax)

    labels = list(accs.keys())
    values = list(accs.values())
    colors = [C_ACCENT if i < len(labels)-1 else C_GREEN for i in range(len(labels))]
    bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.9)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10, color="white")
    ax.set_ylim(min(values) - 0.02, 1.02)
    ax.set_ylabel("Test Accuracy", color="white")
    ax.set_title("Ablation Study: Feature Subset vs. RF Accuracy", color=C_ACCENT)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.002,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9.5, color="white")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return accs


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Feature importance (MDI + permutation)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_feature_importance(model, X, y, save_path):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    # Mean Decrease in Impurity
    mdi = model.feature_importances_

    # Permutation importance on test set
    perm_res = permutation_importance(model, X_te, y_te,
                                      n_repeats=15, random_state=RANDOM_SEED,
                                      n_jobs=-1)
    perm_mean = perm_res.importances_mean
    perm_std  = perm_res.importances_std

    order = np.argsort(mdi)[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _apply_dark(fig, axes)

    # Left: MDI
    ax = axes[0]
    ax.bar(range(8), mdi[order], color=PALETTE, alpha=0.9)
    ax.set_xticks(range(8))
    ax.set_xticklabels([FEATURE_NAMES[i] for i in order],
                        rotation=30, ha="right", fontsize=9, color="white")
    ax.set_ylabel("Mean Decrease in Impurity", color="white")
    ax.set_title("RF Feature Importance (MDI)", color=C_ACCENT)

    # Right: Permutation
    ax2 = axes[1]
    order2 = np.argsort(perm_mean)[::-1]
    ax2.barh(range(8), perm_mean[order2], xerr=perm_std[order2],
             color=PALETTE, alpha=0.9, error_kw=dict(ecolor="white", lw=1.2))
    ax2.set_yticks(range(8))
    ax2.set_yticklabels([FEATURE_NAMES[i] for i in order2], fontsize=9, color="white")
    ax2.set_xlabel("Mean Accuracy Decrease", color="white")
    ax2.set_title("Permutation Importance (test set)", color=C_ACCENT)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Confusion matrix
# ═══════════════════════════════════════════════════════════════════════════════

def plot_confusion(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    _apply_dark(fig, ax)

    cmap = LinearSegmentedColormap.from_list(
        "dark_blue", ["#12121E", "#1A3A5E", "#74C7EC"]
    )
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
    disp.plot(ax=ax, cmap=cmap, colorbar=False)

    ax.set_title(title, color=C_ACCENT, fontsize=13)
    for text in disp.text_.ravel():
        text.set_color("white")
        text.set_fontsize(12)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  TrueSkill vs. OpenSkill comparison
# ═══════════════════════════════════════════════════════════════════════════════

def compare_rating_systems(srv_ts, srv_os, save_path):
    """Compare MMR distribution and rating spread between the two systems."""
    mmrs_ts = sorted(p.mmr for p in srv_ts.players)
    mmrs_os = sorted(p.mmr for p in srv_os.players)
    sigs_ts = sorted(p.sigma for p in srv_ts.players)
    sigs_os = sorted(p.sigma for p in srv_os.players)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _apply_dark(fig, axes)

    # MMR distribution
    ax = axes[0]
    ax.hist(mmrs_ts, bins=80, alpha=0.7, color=C_ACCENT, label="TrueSkill", density=True)
    ax.hist(mmrs_os, bins=80, alpha=0.7, color=C_GREEN,  label="OpenSkill", density=True)
    ax.axvline(np.mean(mmrs_ts), color=C_ACCENT, linestyle="--", lw=1.5,
               label=f"TS mean {np.mean(mmrs_ts):.0f}")
    ax.axvline(np.mean(mmrs_os), color=C_GREEN,  linestyle="--", lw=1.5,
               label=f"OS mean {np.mean(mmrs_os):.0f}")
    ax.set_xlabel("MMR (μ)", color="white")
    ax.set_ylabel("Density", color="white")
    ax.set_title("MMR Distribution: TrueSkill vs. OpenSkill", color=C_ACCENT)
    ax.legend(facecolor="#1C1C38", labelcolor="white", fontsize=8)

    # Sigma (uncertainty) distribution
    ax2 = axes[1]
    ax2.hist(sigs_ts, bins=60, alpha=0.7, color=C_ACCENT, label="TrueSkill", density=True)
    ax2.hist(sigs_os, bins=60, alpha=0.7, color=C_GREEN,  label="OpenSkill", density=True)
    ax2.set_xlabel("σ (skill uncertainty)", color="white")
    ax2.set_ylabel("Density", color="white")
    ax2.set_title("Uncertainty (σ) Distribution", color=C_ACCENT)
    ax2.legend(facecolor="#1C1C38", labelcolor="white", fontsize=8)

    stats = {
        "TrueSkill": {
            "mmr_mean":  float(np.mean(mmrs_ts)),
            "mmr_std":   float(np.std(mmrs_ts)),
            "sigma_mean": float(np.mean(sigs_ts)),
        },
        "OpenSkill": {
            "mmr_mean":  float(np.mean(mmrs_os)),
            "mmr_std":   float(np.std(mmrs_os)),
            "sigma_mean": float(np.mean(sigs_os)),
        },
    }

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Matchmaking quality over time (Close-match % per epoch)
# ═══════════════════════════════════════════════════════════════════════════════

def track_matchmaking_quality(save_path, epochs=30, iters_per_epoch=100):
    """Simulate two fresh servers and track %Close matches each epoch."""
    srv_ts = server()
    srv_os = server()

    pct_close_ts, pct_close_os = [], []

    for epoch in range(epochs):
        # TrueSkill epoch
        close_ts = 0
        total_ts = 0
        for _ in range(iters_per_epoch):
            selected = random.sample(srv_ts.players, 9500)
            random.shuffle(selected)
            for i in range(0, 9500, 10):
                match  = selected[i:i+10]
                team_a = match[:5]
                team_b = match[5:]
                feats  = _extract_features(team_a, team_b, TS_BETA)
                label  = _label_match(feats[-1])
                if label == 0:
                    close_ts += 1
                total_ts += 1
                import elo as _elo_mod
                if random.random() < feats[-1]:
                    _elo_mod.calculate_mmr_change_trueskill(team_a, team_b)
                else:
                    _elo_mod.calculate_mmr_change_trueskill(team_b, team_a)
        pct_close_ts.append(close_ts / total_ts * 100)

        # OpenSkill epoch
        close_os = 0
        total_os = 0
        for _ in range(iters_per_epoch):
            selected = random.sample(srv_os.players, 9500)
            random.shuffle(selected)
            for i in range(0, 9500, 10):
                match  = selected[i:i+10]
                team_a = match[:5]
                team_b = match[5:]
                feats  = _extract_features(team_a, team_b, OS_BETA)
                label  = _label_match(feats[-1])
                if label == 0:
                    close_os += 1
                total_os += 1
                if random.random() < feats[-1]:
                    _elo_mod.calculate_mmr_change_openskill(team_a, team_b)
                else:
                    _elo_mod.calculate_mmr_change_openskill(team_b, team_a)
        pct_close_os.append(close_os / total_os * 100)

        print(f"  Epoch {epoch+1}/{epochs}  TS close={pct_close_ts[-1]:.1f}%  OS close={pct_close_os[-1]:.1f}%", flush=True)

    fig, ax = plt.subplots(figsize=(11, 5))
    _apply_dark(fig, ax)
    xs = range(1, epochs+1)
    ax.plot(xs, pct_close_ts, color=C_ACCENT, lw=2, marker="o", markersize=4,
            label="TrueSkill")
    ax.plot(xs, pct_close_os, color=C_GREEN,  lw=2, marker="s", markersize=4,
            label="OpenSkill")
    ax.set_xlabel("Epoch (×100 iterations)", color="white")
    ax.set_ylabel("% Close matches", color="white")
    ax.set_title("Matchmaking Quality Over Time: % Close Matches per Epoch", color=C_ACCENT)
    ax.legend(facecolor="#1C1C38", labelcolor="white")
    ax.set_ylim(0, 35)
    ax.grid(color="#333355", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return pct_close_ts, pct_close_os


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Learning curve
# ═══════════════════════════════════════════════════════════════════════════════

def plot_learning_curve(X, y, save_path):
    clf = RandomForestClassifier(n_estimators=50, random_state=RANDOM_SEED, n_jobs=-1)
    train_sizes = np.linspace(0.02, 1.0, 12)
    train_sz, train_sc, val_sc = learning_curve(
        clf, X, y, train_sizes=train_sizes,
        cv=5, scoring="accuracy", n_jobs=-1,
        random_state=RANDOM_SEED
    )
    train_mean = np.mean(train_sc, axis=1)
    train_std  = np.std(train_sc,  axis=1)
    val_mean   = np.mean(val_sc,   axis=1)
    val_std    = np.std(val_sc,    axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_dark(fig, ax)
    ax.plot(train_sz, train_mean, color=C_ACCENT, lw=2, marker="o", markersize=4,
            label="Train score")
    ax.fill_between(train_sz, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color=C_ACCENT)
    ax.plot(train_sz, val_mean, color=C_GREEN, lw=2, marker="s", markersize=4,
            label="CV val score")
    ax.fill_between(train_sz, val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color=C_GREEN)
    ax.set_xlabel("Training samples", color="white")
    ax.set_ylabel("Accuracy", color="white")
    ax.set_title("Learning Curve (Random Forest, 50 trees)", color=C_ACCENT)
    ax.legend(facecolor="#1C1C38", labelcolor="white")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.grid(color="#333355", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    results_summary = {}

    # ── 1.  Build TrueSkill dataset ──────────────────────────────────────────
    print("\n=== Step 1: Build TrueSkill dataset ===")
    X_ts, y_ts, srv_ts = build_dataset(
        calculate_mmr_change_trueskill, TS_BETA,
        warmup=500, collect=100, label="TrueSkill"
    )
    counts = {name: int(np.sum(y_ts == i)) for i, name in enumerate(LABEL_NAMES)}
    print(f"  Dataset size: {len(X_ts):,}  |  {counts}")
    results_summary["dataset_trueskill"] = {"size": len(X_ts), "counts": counts}

    # ── 2.  Build OpenSkill dataset ──────────────────────────────────────────
    print("\n=== Step 2: Build OpenSkill dataset ===")
    X_os, y_os, srv_os = build_dataset(
        calculate_mmr_change_openskill, OS_BETA,
        warmup=500, collect=100, label="OpenSkill"
    )
    counts_os = {name: int(np.sum(y_os == i)) for i, name in enumerate(LABEL_NAMES)}
    print(f"  Dataset size: {len(X_os):,}  |  {counts_os}")
    results_summary["dataset_openskill"] = {"size": len(X_os), "counts": counts_os}

    # ── 3.  Multi-model comparison ───────────────────────────────────────────
    print("\n=== Step 3: Multi-model comparison (TrueSkill data) ===")
    model_results, X_te, y_te = compare_models(X_ts, y_ts)
    best_name = plot_model_comparison(
        model_results,
        os.path.join(PLOTS_DIR, "model_comparison.png")
    )
    results_summary["best_model"] = best_name
    results_summary["model_accuracies"] = {
        n: float(model_results[n]["accuracy"]) for n in model_results
    }
    print(f"  Best model: {best_name}")

    # ── 4.  Confusion matrix (best model) ───────────────────────────────────
    print("\n=== Step 4: Confusion matrix ===")
    best_clf = model_results[best_name]["model"]
    plot_confusion(
        model_results[best_name]["y_test"],
        model_results[best_name]["y_pred"],
        f"Confusion Matrix — {best_name}",
        os.path.join(PLOTS_DIR, "confusion_matrix.png")
    )

    # ── 5.  Feature importance ───────────────────────────────────────────────
    print("\n=== Step 5: Feature importance ===")
    rf_model = model_results["Random Forest"]["model"]
    plot_feature_importance(
        rf_model, X_ts, y_ts,
        os.path.join(PLOTS_DIR, "feature_importance.png")
    )
    fi = sorted(zip(FEATURE_NAMES, rf_model.feature_importances_),
                key=lambda x: x[1], reverse=True)
    results_summary["feature_importances"] = [(n, float(v)) for n, v in fi]

    # ── 6.  Ablation study ───────────────────────────────────────────────────
    print("\n=== Step 6: Ablation study ===")
    ablation_accs = ablation_study(
        X_ts, y_ts,
        os.path.join(PLOTS_DIR, "ablation_study.png")
    )
    results_summary["ablation"] = {k.replace("\n", " "): float(v)
                                   for k, v in ablation_accs.items()}

    # ── 7.  Rating system comparison ────────────────────────────────────────
    print("\n=== Step 7: Rating system comparison ===")
    rating_stats = compare_rating_systems(
        srv_ts, srv_os,
        os.path.join(PLOTS_DIR, "rating_comparison.png")
    )
    results_summary["rating_stats"] = rating_stats

    # ── 8.  Matchmaking quality over time ────────────────────────────────────
    print("\n=== Step 8: Matchmaking quality over time ===")
    pct_ts, pct_os = track_matchmaking_quality(
        os.path.join(PLOTS_DIR, "matchmaking_quality.png"),
        epochs=20, iters_per_epoch=50
    )
    results_summary["close_pct_ts_final"] = float(np.mean(pct_ts[-5:]))
    results_summary["close_pct_os_final"] = float(np.mean(pct_os[-5:]))

    # ── 9.  Learning curve ───────────────────────────────────────────────────
    print("\n=== Step 9: Learning curve ===")
    plot_learning_curve(
        X_ts, y_ts,
        os.path.join(PLOTS_DIR, "learning_curve.png")
    )

    # ── 10. Save summary ─────────────────────────────────────────────────────
    summary_path = os.path.join(PLOTS_DIR, "results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nResults summary saved → {summary_path}")

    print("\n=== All done. Plots saved to:", PLOTS_DIR, "===")
    return results_summary


if __name__ == "__main__":
    main()
