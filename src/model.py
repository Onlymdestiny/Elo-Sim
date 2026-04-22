"""
model.py
Match quality ML layer for the Elo Sim project.

Three models via a factory pattern:
  - rf      : RandomForestClassifier (class_weight='balanced')
  - xgb     : XGBClassifier (sample-weighted for imbalance)
  - logreg  : LogisticRegression inside StandardScaler pipeline

Interface:
  - build_model(name)       — returns an unfitted estimator
  - train_fairness_model()  — fit selected model on feature/label data
  - predict_fairness()      — score a single match from raw MMR lists
  - save_model / load_model — pickle persistence

Run directly for a quick demo:
  python model.py
"""

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from elo import (
    player,
    server,
    _extract_features,
    calculate_mmr_change_trueskill,
    calculate_mmr_change_openskill,
    LABEL_NAMES, TS_BETA, OS_BETA,
)

MODEL_NAMES = ("rf", "xgb", "logreg")


def build_model(name: str, random_state: int = 42, n_estimators: int = 100):
    """Factory: return an unfitted estimator for the given model name."""
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    if name == "xgb":
        return XGBClassifier(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softprob",
            num_class=len(LABEL_NAMES),
            random_state=random_state,
            n_jobs=-1,
            eval_metric="mlogloss",
        )
    if name == "logreg":
        # Scaling is critical for LR — wrap in a pipeline so .fit/.predict stay uniform
        return Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=random_state,
            )),
        ])
    raise ValueError(f"Unknown model name: {name!r}. Choose from {MODEL_NAMES}.")


def train_fairness_model(X, y, model_name: str = "rf",
                         n_estimators: int = 100, random_state: int = 42):
    """Train one of the supported classifiers on collected match data.

    Args:
        X: feature matrix (list or ndarray, shape [n_matches, 8]).
        y: label vector (0=Close, 1=Competitive, 2=Stomp).
        model_name: one of MODEL_NAMES ("rf", "xgb", "logreg").
        n_estimators: tree count for RF / XGB (ignored for logreg).

    Returns:
        model:  fitted estimator.
        report: classification_report string on a held-out 20% test split.
    """
    X_arr = np.array(X)
    y_arr = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=0.2, random_state=random_state, stratify=y_arr
    )
    model = build_model(model_name, random_state=random_state, n_estimators=n_estimators)

    # XGBoost: no class_weight param — pass sample weights manually
    if model_name == "xgb":
        weights = compute_sample_weight(class_weight="balanced", y=y_train)
        model.fit(X_train, y_train, sample_weight=weights)
    else:
        model.fit(X_train, y_train)

    report = classification_report(
        y_test, model.predict(X_test), target_names=LABEL_NAMES, zero_division=0
    )
    return model, report


def predict_fairness(team_a_mmrs, team_b_mmrs, model, beta=TS_BETA):
    """Predict match quality for two teams given as plain MMR lists.

    Args:
        team_a_mmrs: list of 5 MMR values for team A.
        team_b_mmrs: list of 5 MMR values for team B.
        model:       trained RandomForestClassifier.
        beta:        performance variance constant (default: TS_BETA).

    Returns:
        fairness_score (int 0-100): higher = more balanced.
        label (str):                "Close", "Competitive", or "Stomp".
    """
    team_a = [player() for _ in range(len(team_a_mmrs))]
    team_b = [player() for _ in range(len(team_b_mmrs))]
    for p, mmr in zip(team_a, team_a_mmrs):
        p.mmr = mmr
    for p, mmr in zip(team_b, team_b_mmrs):
        p.mmr = mmr

    features  = _extract_features(team_a, team_b, beta)
    proba     = model.predict_proba([features])[0]
    label_idx = int(model.predict([features])[0])
    fairness  = round(100 * (proba[0] + 0.5 * proba[1]))
    return fairness, LABEL_NAMES[label_idx]


def save_model(model, path: str, beta: float = TS_BETA, system: str = "trueskill"):
    """Persist a trained model to disk.

    Args:
        model:  fitted RandomForestClassifier.
        path:   destination .pkl file path.
        beta:   beta value used during training (saved for reproducibility).
        system: rating system name ("trueskill" or "openskill").
    """
    with open(path, "wb") as f:
        pickle.dump({"model": model, "beta": beta, "system": system}, f)
    print(f"Model saved → {path}")


def load_model(path: str):
    """Load a model previously saved with save_model().

    Returns:
        model:  fitted RandomForestClassifier.
        beta:   beta constant used during training.
        system: rating system name.
    """
    with open(path, "rb") as f:
        saved = pickle.load(f)
    print(f"Model loaded ← {path}  (system: {saved.get('system', '?')})")
    return saved["model"], saved.get("beta", TS_BETA), saved.get("system", "trueskill")


def _feature_importance(model, feature_names):
    """Return ranked importance list across model types (RF, XGB, LogReg)."""
    if hasattr(model, "feature_importances_"):
        scores = model.feature_importances_
    elif hasattr(model, "named_steps") and "lr" in model.named_steps:
        # LogReg coefficients — average magnitude across classes
        scores = np.mean(np.abs(model.named_steps["lr"].coef_), axis=0)
    else:
        return []
    return sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    print("=== Match Quality Model Demo (RF / XGB / LogReg) ===\n")

    # Build one dataset and reuse it for all three models
    print("Warming up simulation (500 iterations)...")
    srv = server()
    for i in range(500):
        srv.iterate()

    print("Collecting training data (100 iterations)...")
    X, y = srv.collect_training_data(n_iters=100)
    counts = {name: y.count(i) for i, name in enumerate(LABEL_NAMES)}
    print(f"Dataset: {len(X):,} matches  |  " +
          "  ".join(f"{k}: {v}" for k, v in counts.items()))
    print()

    feature_names = ["mean_A", "mean_B", "mmr_gap",
                     "var_A", "var_B", "sigma_A", "sigma_B", "p_win"]
    demos = [
        ([2000] * 5, [2000] * 5,           "Perfectly balanced"),
        ([2300] * 5, [1700] * 5,           "300 avg MMR gap"),
        ([3000] * 5, [1000] * 5,           "2000 avg MMR gap (stomp)"),
        ([2500, 2500, 2000, 1500, 1500],
         [2000] * 5,                       "Mixed-skill vs. uniform"),
    ]

    for name in MODEL_NAMES:
        print(f"--- {name.upper()} ---")
        model, report = train_fairness_model(X, y, model_name=name)
        for line in report.splitlines():
            print("   ", line)

        importances = _feature_importance(model, feature_names)
        if importances:
            print("\n  Feature importances:")
            for fname, imp in importances:
                print(f"    {fname:<10} {imp:.4f}")

        print("\n  Sample predictions:")
        for a_mmrs, b_mmrs, desc in demos:
            score, label = predict_fairness(a_mmrs, b_mmrs, model)
            print(f"    {desc:<40}  Fairness: {score:3d}/100  [{label}]")
        print()
