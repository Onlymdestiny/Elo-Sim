# Elo-Sim — Match-Quality Prediction for Ranked 5v5 Games

An end-to-end pipeline that collects live ranked match data from the Riot Games
API, reconstructs each player's approximate MMR from their public rank,
builds a match-quality feature vector, and trains a classifier that predicts
whether a future match will be a **Close**, **Competitive**, or **Stomp** game
before it starts.

The full paper is in [`iclr2026/paper.pdf`](iclr2026/paper.pdf).

---

## What you get

- A data collector that calls the Riot API safely (respects rate limits,
  samples across every rank from Iron to Challenger).
- A feature-extraction layer that turns ten players' public ranks into an
  eight-dimensional match-quality vector.
- Three interchangeable classifiers (Random Forest, XGBoost, Logistic
  Regression) trained on the collected data.
- A command-line tool that takes two teams' MMRs and prints a fairness score
  from 0 to 100.

---

## Requirements

- Python 3.10 or newer
- A free Riot Games **development API key** (instructions below)
- About 20 minutes to collect 500 matches

---

## Step 1 — Get a Riot API key

1. Go to <https://developer.riotgames.com/> and sign in with the Riot account
   you use for League of Legends.
2. Accept the developer agreement if prompted.
3. On the dashboard, find **"Development API Key"** in the middle of the page.
4. Click **"Regenerate API Key"**. A long string starting with `RGAPI-…` will
   appear.
5. Copy that string. **Keep it secret** — anyone with it can make requests on
   your behalf.

**Important:** development keys expire every 24 hours. If collection starts
failing with `401 Unauthorized`, come back to this page and regenerate.

For a permanent key, apply for a "Personal API Key" on the same site (takes a
few days to approve). Not required for this project.

---

## Step 2 — Clone and install

```bash
git clone https://github.com/Onlymdestiny/Elo-Sim.git
cd Elo-Sim
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install requests python-dotenv numpy pandas scikit-learn xgboost matplotlib scipy
```

---

## Step 3 — Configure your API key

Create a file called `.env` in the project root (same folder as this README).
Paste your key into it:

```
RIOT_API_KEY=RGAPI-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
RIOT_PLATFORM=na1
RIOT_REGION=americas
MATCHES_TARGET=500
```

- `RIOT_PLATFORM` picks the server (`na1`, `euw1`, `kr`, `br1`, etc.).
- `RIOT_REGION` is the routing cluster for that platform
  (`americas` for NA/BR/LAN/LAS, `europe` for EUW/EUNE/TR/RU, `asia` for KR/JP,
  `sea` for OCE/PH/SG/TH/TW/VN).

The `.env` file is already in `.gitignore`, so it won't be pushed to GitHub.

---

## Step 4 — Collect match data

Run the collector. It will sample players from every tier (Iron through
Challenger) to get a representative spread of skill levels:

```bash
python src/collect_real_data.py --tier-seed --players-per-tier 100
```

This takes about 18 minutes for 500 matches on a development key. The output
goes to `data/real_matches.csv`. If you run the collector again, it writes to
`data/real_matches2.csv`, `data/real_matches3.csv`, and so on, so you never
overwrite an existing collection.

**Useful flags:**

- `--target 1000` — collect more than the default 500 matches.
- `--platform euw1` — use a different server for one run.
- `--reset` — delete the cache and overwrite `real_matches.csv`.
- `--summoner "SomeName"` — start from one specific summoner instead of
  sampling the ladder.

---

## Step 5 — Train and validate the model

To train on the first collection and test on the second (the evaluation used
in the paper):

```bash
python src/validate_real_data.py \
  --train-csv data/real_matches.csv \
  --test-csv  data/real_matches2.csv \
  --model xgb --features baseline
```

This prints accuracy and Spearman correlations, and writes confusion matrix
and scatter plots into `plots/`.

**Options:**

- `--model {rf, xgb, logreg}` — pick the classifier family.
- `--features {baseline, extended}` — baseline is the 8-D vector from the
  paper; extended adds winrate and Riot status flags.
- `--save-model saved/my_model.pkl` — save the trained model for reuse.

---

## Step 6 — Predict fairness for a hypothetical match

Once you have a saved model (or just the defaults), you can ask the tool to
score any ten-player matchup:

```bash
python src/cli_fairness.py \
  --team-a 2400 2500 2300 2600 2400 \
  --team-b 2550 2400 2500 2350 2500
```

Output looks like:

```
Team A win probability : 0.47
Fairness score         : 71 / 100
Predicted label        : Competitive
```

Feed it a CSV of matchups with `--batch matchups.csv` to score many at once.

---

## Project layout

```
Elo-Sim/
├── src/
│   ├── collect_real_data.py   # Riot API collector, rate limiter, MMR reconstruction
│   ├── validate_real_data.py  # Train + held-out evaluation
│   ├── ml_analysis.py         # Extra analysis, plots, feature comparisons
│   ├── cli_fairness.py        # Command-line fairness-score tool
│   ├── model.py               # Classifier factory (RF / XGB / LogReg)
│   └── elo.py                 # TrueSkill-style update math (simulation sandbox)
├── plots/                     # Output figures from validation runs
├── iclr2026/                  # Paper source + compiled PDF
├── docs/                      # Course-project documents (rubric, write-up)
├── RUBRIC_MAPPING.md          # Mapping from rubric to paper sections
└── slides.html                # reveal.js presentation
```

Data collected by the scripts goes to `data/` (git-ignored).

---

## Notes and caveats

- **API key rotation.** Development keys die after 24 hours. If a long
  collection fails partway through, just regenerate the key on the Riot
  portal, update `.env`, and rerun — the collector's `.seen_matches.json`
  cache means it will resume instead of restarting.
- **Player privacy.** The collected CSVs contain Riot PUUIDs (stable
  per-account identifiers). Do not publish them. `data/` is excluded from git
  by default.
- **Why only \~50% accuracy?** Because ground-truth match quality is noisy
  (two reasonable raters would disagree on borderline games) and the public
  rank we see is a lossy projection of Riot's hidden matchmaking MMR. See
  Sections 6 and 7 of the paper for the full diagnosis.

---

## License

Riot Games API terms apply to any data you collect — you may not redistribute
raw match data or PUUIDs. The code in `src/` is free to use and modify for
non-commercial research.

## Citation

If this work is useful to you, the paper is at
[`iclr2026/paper.pdf`](iclr2026/paper.pdf).
