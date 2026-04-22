"""
cli_fairness.py
Checkpoint 2: Standalone CLI for Match Quality / Fairness Score prediction.

Usage examples
--------------
  # Quick fairness check (trains model on-the-fly):
  python cli_fairness.py --team-a 2000,2100,1900,2200,2000 \
                         --team-b 1800,1900,2000,2100,2300

  # Use OpenSkill rating system:
  python cli_fairness.py --system openskill \
                         --team-a 2500,2500,2000,1500,1500 \
                         --team-b 2000,2000,2000,2000,2000

  # Save the trained model for future use (avoids retraining each run):
  python cli_fairness.py --team-a 2000,2000,2000,2000,2000 \
                         --team-b 2000,2000,2000,2000,2000 \
                         --save-model model_ts.pkl

  # Load a previously saved model (fast, skips training):
  python cli_fairness.py --team-a 3000,3000,3000,3000,3000 \
                         --team-b 1000,1000,1000,1000,1000 \
                         --load-model model_ts.pkl

  # Batch mode: compare multiple team compositions from a JSON file:
  python cli_fairness.py --batch matchups.json --load-model model_ts.pkl
"""

import argparse
import json
import math
import os
import pickle
import random
import sys

# ---------------------------------------------------------------------------
# Colour helpers (ANSI, graceful fallback if terminal does not support them)
# ---------------------------------------------------------------------------
try:
    import colorama
    colorama.init(autoreset=True)
    ANSI = True
except ImportError:
    ANSI = False

def _col(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if ANSI else text

CYAN   = lambda t: _col("96", t)
GREEN  = lambda t: _col("92", t)
YELLOW = lambda t: _col("93", t)
RED    = lambda t: _col("91", t)
BOLD   = lambda t: _col("1",  t)
GRAY   = lambda t: _col("90", t)

# ---------------------------------------------------------------------------
# Import simulation / ML helpers
# ---------------------------------------------------------------------------
try:
    from elo import (
        server,
        calculate_mmr_change_trueskill,
        calculate_mmr_change_openskill,
        LABEL_NAMES,
        TS_BETA, OS_BETA,
    )
    from model import train_fairness_model, predict_fairness
except ImportError:
    print("ERROR: Cannot import elo.py. Make sure it is in the same directory.", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Fairness score → colour
# ---------------------------------------------------------------------------
def _colour_score(score: int) -> str:
    if score >= 70:
        return GREEN(f"{score}/100")
    if score >= 40:
        return YELLOW(f"{score}/100")
    return RED(f"{score}/100")

def _colour_label(label: str) -> str:
    c = {"Close": GREEN, "Competitive": YELLOW, "Stomp": RED}
    return c.get(label, CYAN)(label)

# ---------------------------------------------------------------------------
# Progress bar helper
# ---------------------------------------------------------------------------
def _progress(current: int, total: int, width: int = 40) -> str:
    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {current}/{total}"

# ---------------------------------------------------------------------------
# Model training with progress output
# ---------------------------------------------------------------------------
def build_model(system: str = "trueskill",
                warmup: int = 500,
                collect: int = 100,
                n_estimators: int = 100):
    """Train a fresh RF model from simulation data."""
    if system == "trueskill":
        rating_func = calculate_mmr_change_trueskill
        beta        = TS_BETA
    else:
        rating_func = calculate_mmr_change_openskill
        beta        = OS_BETA

    print(BOLD(f"\n  Building model ({system})…"))
    print(f"  Warming up server ({warmup} iterations)…")
    srv = server()
    for i in range(warmup):
        srv.iterate(rating_func=rating_func, beta=beta)
        if (i + 1) % 100 == 0:
            print(f"  {_progress(i+1, warmup)}", end="\r")
    print()

    print(f"  Collecting training data ({collect} iterations)…")
    X, y = srv.collect_training_data(
        n_iters=collect, rating_func=rating_func, beta=beta
    )
    print(f"  Training Random Forest ({n_estimators} trees, {len(X):,} samples)…")
    model, report = train_fairness_model(X, y, n_estimators=n_estimators)
    print(GRAY("  ─" * 36))
    print(GRAY("  Classification report (test set, 20%):"))
    for line in report.splitlines():
        print(GRAY("  " + line))
    print(GRAY("  ─" * 36))
    return model, beta

# ---------------------------------------------------------------------------
# Single match display
# ---------------------------------------------------------------------------
def _display_result(team_a: list, team_b: list,
                    score: int, label: str,
                    p_win_a: float, desc: str = ""):
    bar_len = 40
    filled  = round(bar_len * score / 100)
    bar     = "█" * filled + "░" * (bar_len - filled)

    print()
    if desc:
        print(BOLD(f"  {desc}"))
    print(f"  Team A  MMRs: {CYAN(str(team_a))}")
    print(f"  Team B  MMRs: {CYAN(str(team_b))}")
    print(f"  Avg MMR  A={sum(team_a)/len(team_a):.0f}   B={sum(team_b)/len(team_b):.0f}  "
          f"  Gap={abs(sum(team_a)-sum(team_b))/len(team_a):.0f}")
    print(f"  P(A wins): {p_win_a:.3f}")
    print(f"  Match label  : {_colour_label(label)}")
    print(f"  Fairness score: {_colour_score(score)}  [{bar}]")

# ---------------------------------------------------------------------------
# p_win helper (for display only)
# ---------------------------------------------------------------------------
def _p_win(team_a: list, team_b: list, beta: float) -> float:
    from elo import _Phi, _extract_features, player as _P  # noqa: PLC0415
    pa = [_P() for _ in team_a]
    pb = [_P() for _ in team_b]
    for p, m in zip(pa, team_a): p.mmr = m
    for p, m in zip(pb, team_b): p.mmr = m
    return _extract_features(pa, pb, beta)[-1]

# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------
def run_batch(path: str, model, beta: float):
    """Read a JSON file of matchups and print results for each."""
    try:
        with open(path) as f:
            matchups = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(RED(f"ERROR reading batch file: {e}"), file=sys.stderr)
        sys.exit(1)

    print(BOLD(f"\n  Batch mode: {len(matchups)} matchup(s) from {path}"))
    summary = []
    for idx, m in enumerate(matchups, 1):
        a = m.get("team_a", [])
        b = m.get("team_b", [])
        desc = m.get("description", f"Matchup {idx}")
        if len(a) != 5 or len(b) != 5:
            print(RED(f"  [{idx}] Skipped — each team must have exactly 5 MMR values."))
            continue
        score, label = predict_fairness(a, b, model, beta=beta)
        pw = _p_win(a, b, beta)
        _display_result(a, b, score, label, pw, desc=f"[{idx}] {desc}")
        summary.append({"description": desc, "score": score, "label": label})

    print(BOLD(f"\n  Summary ({len(summary)} matchups):"))
    for row in summary:
        s = row["score"]
        print(f"    {row['description']:<40}  {_colour_score(s)}  {_colour_label(row['label'])}")

# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cli_fairness",
        description="Match Quality Optimizer — Fairness Score CLI (Checkpoint 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Team composition
    p.add_argument("--team-a", metavar="MMRs",
                   help="Comma-separated MMR values for Team A (5 values).")
    p.add_argument("--team-b", metavar="MMRs",
                   help="Comma-separated MMR values for Team B (5 values).")

    # Rating system
    p.add_argument("--system", choices=["trueskill", "openskill"],
                   default="trueskill",
                   help="Rating algorithm to use (default: trueskill).")

    # Model persistence
    p.add_argument("--save-model", metavar="FILE",
                   help="Save the trained model to this .pkl file after use.")
    p.add_argument("--load-model", metavar="FILE",
                   help="Load a pre-trained model from this .pkl file (skips training).")

    # Training hyperparameters
    p.add_argument("--warmup",      type=int, default=500,
                   help="Simulation warm-up iterations before data collection (default 500).")
    p.add_argument("--collect",     type=int, default=100,
                   help="Data-collection iterations (default 100 → ~95 k samples).")
    p.add_argument("--n-estimators", type=int, default=100,
                   help="Number of RF trees (default 100).")

    # Batch mode
    p.add_argument("--batch", metavar="FILE",
                   help="JSON file of matchups for batch prediction.")

    return p

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = build_parser()
    args   = parser.parse_args()

    beta = TS_BETA if args.system == "trueskill" else OS_BETA

    # ── Load or train model ─────────────────────────────────────────────────
    model = None
    if args.load_model:
        if not os.path.exists(args.load_model):
            print(RED(f"ERROR: model file not found: {args.load_model}"), file=sys.stderr)
            sys.exit(1)
        with open(args.load_model, "rb") as f:
            saved = pickle.load(f)
        model = saved["model"]
        beta  = saved.get("beta", beta)
        print(GREEN(f"  Loaded model from {args.load_model}  (system: {saved.get('system', '?')})"))
    else:
        model, beta = build_model(
            system       = args.system,
            warmup       = args.warmup,
            collect      = args.collect,
            n_estimators = args.n_estimators,
        )

    # ── Save model if requested ─────────────────────────────────────────────
    if args.save_model:
        with open(args.save_model, "wb") as f:
            pickle.dump({"model": model, "beta": beta, "system": args.system}, f)
        print(GREEN(f"\n  Model saved → {args.save_model}"))

    # ── Batch mode ──────────────────────────────────────────────────────────
    if args.batch:
        run_batch(args.batch, model, beta)
        return

    # ── Single match ────────────────────────────────────────────────────────
    if not args.team_a or not args.team_b:
        parser.print_help()
        print(RED("\nERROR: Provide --team-a and --team-b (or --batch)."), file=sys.stderr)
        sys.exit(1)

    try:
        team_a = [int(x.strip()) for x in args.team_a.split(",")]
        team_b = [int(x.strip()) for x in args.team_b.split(",")]
    except ValueError:
        print(RED("ERROR: MMR values must be integers separated by commas."), file=sys.stderr)
        sys.exit(1)

    if len(team_a) != 5 or len(team_b) != 5:
        print(RED("ERROR: Each team must have exactly 5 MMR values."), file=sys.stderr)
        sys.exit(1)

    score, label = predict_fairness(team_a, team_b, model, beta=beta)
    pw = _p_win(team_a, team_b, beta)
    _display_result(team_a, team_b, score, label, pw)
    print()


if __name__ == "__main__":
    main()
