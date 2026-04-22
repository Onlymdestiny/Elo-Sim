# Rubric Mapping — Paper & Code → Rubric Criteria

This document maps each row of the course-project rubric
(`docs/rubric_ECE570_final_Spring2026.pdf`) to the exact section of the
term paper (`iclr2026/paper.tex`) or the exact file in the codebase that
satisfies it. Quoted prose is pulled verbatim from the paper.

---

## Term Paper — 55 points

### 1. Abstract (5 pts) — Excellent target

> *"Clearly summarizes the main problem, approach, and results concisely."*

**Paper section:** `\begin{abstract} ... \end{abstract}` in
[iclr2026/paper.tex](iclr2026/paper.tex) (lines 23–44).

Covers all three required elements in a single paragraph:

- **Problem:** *"estimates pre-match quality for ranked 5v5 online games"*
- **Approach:** *"collects 500 live RANKED\_SOLO\_5x5 matches from the Riot
  Games API, reconstructs per-player approximate MMR from public tier and LP,
  and assembles an eight-dimensional match-quality feature vector …
  fed to one of three interchangeable classifiers (Random Forest, XGBoost,
  Logistic Regression)"*
- **Results:** *"all three classifiers converge to ≈50% accuracy with weak
  Spearman correlation (|ρ|≤0.07) against outcome proxies … ceiling is not
  model capacity but the label/feature signal itself"*

---

### 2. Introduction (10 pts) — Excellent target

> *"Provides clear context, motivation, significance, objectives, and goals."*

**Paper section:** `\section{Introduction}` (§1).

- **Context** (¶1): *"Competitive online games rely on hidden rating systems
  (Elo, Glicko, TrueSkill) to pair players such that each match is close and
  engaging."*
- **Motivation** (¶1): *"When matchmaking misjudges skill, the result is a
  stomp: one team snowballs early, the game ends in under 22 minutes, and
  both sides report a poor experience."*
- **User / objective** (¶2): *"We target the matchmaking engineer as the
  intended user. The tool must run before the match starts, use only data
  the server has at lobby time, and return a fairness score together with
  a coarse quality label."*
- **Significance** (¶2): design-constraint framing explaining why an oracle
  trained on post-match telemetry would be unusable and why transparency
  matters.
- **Goals / contributions** (¶3, enumerated):
  1. End-to-end data-to-prediction prototype with factory boundary.
  2. Authenticated Riot API collection pipeline with dual-window limiter,
     tier-balanced seed sampler, piecewise MMR reconstruction.
  3. Honest held-out evaluation on an independently-seeded second collection.
- **Scope statement** (¶4): *"we treat the data-acquisition pipeline as a
  first-class artifact … every performance ceiling we observe is ultimately
  a statement about the data, not the classifier."*

---

### 3. Related Work (10 pts) — Excellent target

> *"Reviews existing research, highlighting differences or enhancements
> clearly."*

**Paper section:** `\section{Related Work}` (§2).

- **Rating-systems line:** cites Elo (1978) and Herbrich et al. TrueSkill
  (2006), noting *"whose closed-form team win probability we adopt directly
  as one feature component"* — explicit statement of what is reused.
- **Gap called out:** *"Most production matchmakers optimise a
  skill-difference loss; few publish the quality-of-match predictor
  separately"* — positions our contribution.
- **Contrast with prior LoL-prediction work:** Do et al. (2021) use
  post-match statistics (gold graph, KDA) *"which leak outcome information
  and are unusable at lobby time. Our features are strictly pre-match"* —
  explicit difference.
- **ML-methodology line:** Caruana & Niculescu-Mizil (2006) cited as the
  basis for the three-classifier (bagging / boosting / linear) diagnostic
  of feature-ceiling vs. model-capacity-ceiling.

Each citation is tied to a concrete reuse, contrast, or methodological
grounding rather than being listed passively.

---

### 4. Methodology (10 pts) — Excellent target

> *"Detailed explanation of methods, models, and setup with logical flow."*

**Paper section:** `\section{Methodology}` (§3), five subsections.

| Subsection | Content | Key equations |
|---|---|---|
| §3.1 Team skill summary and win probability | Gaussian player model, team aggregation, closed-form TrueSkill win prob | eq. (1) team-agg, (2) p_win |
| §3.2 Match-quality feature extraction | 8-D baseline + 14-D extended feature vectors | (3) φ_base, (4) φ_ext |
| §3.3 Real-data retrieval pipeline | Tier-balanced seed sampling, dual-window token-bucket limiter, piecewise MMR reconstruction, σ estimation, throughput | (5) seeds, (6) ratelimit, (7) mmr, (8) sigma |
| §3.4 Match-quality label | Worst-of-three vote over duration/gold/kill | (9) votes, (10) label |
| §3.5 Classifier factory and fairness score | Three-model factory, fairness score in [0,100] | (11) fairness |

Scope clarifier at §3.1 pins which equations run at inference
(`eq:mmr`, `eq:sigma`, `eq:phi-base`, `eq:fairness`) vs. which live in the
design-phase sandbox (Appendix B). Logical flow: retrieval → features +
labels → classifier.

---

### 5. Experimental Results & Result Analysis (10 pts) — Excellent target

> *"Results are presented with clear explanations and analysis, supported
> by visual aids or tables."*

**Paper section:** `\section{Experiments}` (§4).

- **Setup paragraph:** *"train-on-real, test-on-real-held-out: train on one
  500-match collection and test on a second, independently-seeded 500-match
  collection gathered on a different day."*
- **Metrics:** label accuracy, majority-class baseline, Spearman ρ — with
  justification *"Spearman rather than Pearson because fairness and
  duration/gold/kill are on different scales."*
- **Table 1** (`\label{tab:results}`): all three classifiers × {baseline,
  extended} features, plus majority and random-guess baselines.
- **Figure 1** (`\label{fig:pipeline}`): confusion matrix, fairness
  histogram by true label, fairness vs. outcome scatter — drawn from
  `plots/real_xgb_baseline_*.png`.
- **Analysis subsections:**
  - §4.1 *Three-family convergence implies a feature ceiling* — explicitly
    explains why RF/XGB/LogReg landing within 2 pp signals a feature
    ceiling, not a model ceiling.
  - §4.2 *Extended features do not help* — two concrete reasons
    (collinearity with tier; sparse status flags at n=500).
  - §4.3 *Where is the remaining signal?* — attributes the gap to
    (i) label noise in the voted thresholds and (ii) MMR reconstruction
    error from tier+LP lossy projection.

---

### 6. Conclusion with Contribution (5 pts) — Excellent target

> *"Clear extension of an existing idea or a reimplementation featuring
> significant improvements and technical depth. Demonstrates profound
> understanding through thoughtful design, new experiments, optimizations,
> or in-depth analysis."*

**Paper sections:** `\section{Limitations and Future Work}` (§6) +
`\section{Conclusion}` (§7).

- **System summary** paragraph (Conclusion) re-anchors to the equations
  actually deployed: seed sampling (5), rate limiter (6), MMR
  reconstruction (7), σ estimation (8), feature/label layer (3)–(10),
  closed-form p_win (2) as a feature component, fairness score (11).
- **Contribution statement:** *"the system itself — with equal engineering
  attention paid to the data pipeline and the model — plus the diagnostic
  finding that the performance bottleneck is label noise and MMR
  reconstruction rather than model choice."*
- **In-depth analysis:** §5 *Discussion* expands this with the *"Data is
  the bottleneck, not the model"* paragraph, showing every accuracy-cap
  quantity is in the data pipeline (eq:mmr, eq:votes, eq:label) rather
  than in the classifier.
- **Dedicated Limitations & Future Work section (§6):** four paragraphs
  covering (i) the author-chosen label thresholds in eq. (9) and paths
  to replace them (re-calibration, ordinal/soft labels, studio-canonical
  labels); (ii) the critical observation that the $\widehat{M}$ feature
  is a remap of *visible* rank whereas Riot's matchmaker decides on the
  *hidden* MMR $M^\star$, making the reconstruction error
  systematically aligned with outcome-determining information; (iii)
  scope limitations (1000 matches, NA1-only, single patch, ordering
  discarded by 3-class formulation); and (iv) concrete methodological
  extensions — ordinal regression, calibration / conformal prediction,
  multi-region held-out, targeted active collection.

---

### 7. Formatting and References (4 pts) — Excellent target

> *"Fully adheres to ICLR-26 guidelines, has correctly formatted references,
> and the paper length is within page limit."*

- **ICLR-26 style:** `\documentclass{article}` with
  `\usepackage{iclr2026_conference,times}` at the top of
  [iclr2026/paper.tex](iclr2026/paper.tex). All ICLR assets
  (`iclr2026_conference.sty`, `.bst`, `fancyhdr.sty`, `natbib.sty`) are
  in the `iclr2026/` directory alongside the source.
- **References:** `\begin{thebibliography}` with four entries (Caruana &
  Niculescu-Mizil 2006, Do et al. 2021, Elo 1978, Herbrich et al. 2006),
  all \citet/\citep-able from the main text. Every bibliography entry is
  actually cited.
- **Page length:** within the ICLR main-text limit (target 5–6 pages of
  main content + bibliography + appendices; appendices are permitted
  beyond the main limit).

---

### 8. LLM Acknowledgement (1 pt)

> *"LLM usage is well documented, or not used LLM at all."*

**Paper section:** `\subsubsection*{LLM Acknowledgments}` immediately
before the bibliography.

Specifies: (a) which model was used, (b) what it was used for (boilerplate
LaTeX, prose refinement, code review on rate-limiter and feature pipeline),
(c) what remained the author's (experimental design, methodology,
collection, numerical results, plots, framing), and (d) that the LLM never
had access to the Riot API key or collected match data.

---

## Code Quality and Implementation — 30 points

### 9. Functionality and Relevance (10 pts) — Excellent target

> *"Code is fully functional, directly addresses the project's goals, and
> is easily reproducible."*

**Files:**

- [src/collect_real_data.py](src/collect_real_data.py) — Riot API
  collection pipeline. Implements tier-balanced seed sampling
  (§3.3(a), eq. 5), dual-window token-bucket rate limiter (§3.3(b),
  eq. 6), piecewise MMR reconstruction (§3.3(c), eq. 7), σ estimation
  (eq. 8), worst-of-three label voting (§3.4, eqs. 9–10). Writes CSVs to
  `data/real_matches*.csv`.
- [src/validate_real_data.py](src/validate_real_data.py) — held-out
  evaluation used to produce Table 1 and Figure 1. Supports
  `--train-csv` / `--test-csv` for the paper's train-on-real,
  test-on-held-out setting and `--features {baseline,extended}` for the
  8-D / 14-D ablation.
- [src/model.py](src/model.py) — three-classifier factory (RF / XGB /
  LogReg, §3.5).
- [src/elo.py](src/elo.py) — feature extraction and closed-form p_win
  (eq. 2).
- [src/cli_fairness.py](src/cli_fairness.py) — user-facing entry point
  that calls `predict_fairness` on a single matchup.

Reproducibility hooks are explicit in the paper: module-level constants
for CSV schema, label thresholds, MMR bases, and rate-limit windows are
called out in §5 (Discussion, last paragraph) and the Appendix.

---

### 10. Code Quality and Organization (10 pts) — Excellent target

> *"Code is well-structured, documented, non-trivial, and readable."*

**Directory layout** (after reorganization):

```
src/      Python source (6 files, cross-imported via from elo import … etc.)
data/     CSV datasets + .seen_matches.json runtime state
plots/    Generated figures referenced from iclr2026/paper.tex
iclr2026/ LaTeX source + ICLR 2026 style assets
docs/     Rubric + project-brief PDFs
Archive/  Superseded checkpoints (pptx, old test_api.py)
```

**Non-trivial engineering** surfaced in both paper and code:

- Dual-window token-bucket limiter with deque-based sliding windows
  ([src/collect_real_data.py](src/collect_real_data.py), paper eq. 6).
- Piecewise tier+LP → MMR reconstruction
  ([src/collect_real_data.py](src/collect_real_data.py) TIER_BASE /
  RANK_OFFSET / APEX_BASE constants, paper eq. 7).
- Three-classifier factory with uniform fit/predict_proba contract
  ([src/model.py](src/model.py), paper §3.5).
- Tier-balanced BFS seeding across Iron → Challenger
  ([src/collect_real_data.py](src/collect_real_data.py), paper eq. 5).

**Readability:** docstrings on public functions; module-level constants
at file head (paper §5 calls this out); auto-incrementing output files
(`real_matches.csv → real_matches2.csv → …`) avoid silent overwrites.

---

### 11. README (5 pts)

> *"Includes clear, functional README file for easy setup to run the
> project."*

**Status: MISSING — action required.**

Recommended contents (not yet written): setup steps
(`pip install -r requirements.txt`, `.env` with `RIOT_API_KEY`), command
to collect (`python src/collect_real_data.py --tier-seed
--players-per-tier 100`), command to validate (`python
src/validate_real_data.py --train-csv data/real_matches.csv --test-csv
data/real_matches2.csv --model xgb --features baseline`), and a one-line
description of the directory layout above.

---

### 12. Substantive Evaluation (5 pts) — Excellent target

> *"The code's performance aligns closely with the reported results in
> the term paper."*

**Reproducibility contract:** running
`python src/validate_real_data.py --train-csv data/real_matches.csv
--test-csv data/real_matches2.csv --model {rf,xgb,logreg} --features
{baseline,extended}` over the six combinations in Table 1 should
reproduce the ≈50% accuracy numbers and the Spearman ρ values within
random-seed noise. Output plots land in `plots/` with the prefix
`real_{model}_{features}_validation_*.png` (referenced in the paper's
Figure 1).

Any minor variation (tier-seed order, which accounts are active at
collection time) is expected due to the live Riot API, and is
documented in §3.3 and the Appendix.

---

## Summary — at-a-glance status

| Rubric row | Points | Status |
|---|---:|---|
| Abstract | 5 | ✅ covered by abstract |
| Introduction | 10 | ✅ covered by §1 |
| Related Work | 10 | ✅ covered by §2 |
| Methodology | 10 | ✅ covered by §3 |
| Experimental Results & Analysis | 10 | ✅ covered by §4 + Table 1 + Figure 1 |
| Conclusion with Contribution | 5 | ✅ covered by §6 + §5 |
| Formatting and References | 4 | ✅ ICLR style + 4 bib entries, all cited |
| LLM Acknowledgement | 1 | ✅ covered by dedicated paragraph |
| Functionality and Relevance | 10 | ✅ `src/` scripts map to paper equations |
| Code Quality and Organization | 10 | ✅ `src/`, `data/`, `plots/`, `iclr2026/` split |
| README | 5 | ⚠️ **missing — must be written before submission** |
| Substantive Evaluation | 5 | ✅ reproduction command above |
