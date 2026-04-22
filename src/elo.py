import random
import math
import matplotlib.pyplot as plt

MIN_MMR = 400

# --- TrueSkill default parameters ---
TS_BETA = 100.0    # Performance variance per player
TS_TAU = 2.0       # Dynamics factor (skill drift per game)

# --- OpenSkill default parameters ---
OS_BETA = 100.0    # Performance variance per player
OS_KAPPA = 0.0001  # Minimum variance multiplier to prevent sigma collapse

# --- Match quality labels ---
LABEL_NAMES = ["Close", "Competitive", "Stomp"]


class player:
    def __init__(self):
        # Sample from real ranked distribution: mean ~2800, std ~1300 spans Iron→Challenger
        self.mmr   = max(MIN_MMR, random.gauss(2800, 1300))
        self.sigma = random.uniform(60, 210)   # mirrors tier-based uncertainty (Challenger→Iron)
        self.display_rp = self.mmr

    def update_mmr(self, delta):
        self.mmr = max(MIN_MMR, self.mmr + delta)

    def update_display_rp(self):
        self.display_rp = max(MIN_MMR, self.mmr - 3 * self.sigma)


# ---------------------------------------------------------------------------
# Helper functions: standard normal PDF and CDF
# ---------------------------------------------------------------------------

def _phi(x):
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _Phi(x):
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# TrueSkill truncated-Gaussian correction functions
# ---------------------------------------------------------------------------

def _v_func(t, epsilon=0.0):
    """Additive correction for the mean (V function) under a win outcome."""
    denom = _Phi(t - epsilon)
    if denom < 1e-10:
        return -t + epsilon
    return _phi(t - epsilon) / denom


def _w_func(t, epsilon=0.0):
    """Multiplicative correction for the variance (W function) under a win outcome."""
    v = _v_func(t, epsilon)
    return v * (v + t - epsilon)


# ---------------------------------------------------------------------------
# Match quality helpers
# ---------------------------------------------------------------------------

def _extract_features(team_a, team_b, beta=TS_BETA):
    """Return an 8-D feature vector describing a 5v5 match.

    Features: [mean_mmr_a, mean_mmr_b, abs_mmr_gap,
               intra_var_a, intra_var_b, avg_sigma_a, avg_sigma_b, p_win_a]
    """
    mu_a   = sum(p.mmr for p in team_a)
    mu_b   = sum(p.mmr for p in team_b)
    mean_a = mu_a / len(team_a)
    mean_b = mu_b / len(team_b)
    var_a  = sum((p.mmr - mean_a) ** 2 for p in team_a) / len(team_a)
    var_b  = sum((p.mmr - mean_b) ** 2 for p in team_b) / len(team_b)
    sig_a  = sum(p.sigma for p in team_a) / len(team_a)
    sig_b  = sum(p.sigma for p in team_b) / len(team_b)
    c      = math.sqrt(sum(p.sigma ** 2 for p in team_a + team_b)
                       + (len(team_a) + len(team_b)) * beta ** 2)
    p_win  = _Phi((mu_a - mu_b) / c)
    return [mean_a, mean_b, abs(mu_a - mu_b), var_a, var_b, sig_a, sig_b, p_win]


def _label_match(p_win):
    """Map a win probability to a match quality class index.

    0 = Close        (|p_win - 0.5| < 0.10, i.e. p_win in [0.40, 0.60])
    1 = Competitive  (|p_win - 0.5| < 0.25, i.e. p_win in [0.25, 0.75])
    2 = Stomp        (otherwise)
    """
    gap = abs(p_win - 0.5)
    if gap < 0.10:
        return 0
    if gap < 0.25:
        return 1
    return 2


# ---------------------------------------------------------------------------
# Rating algorithm 1 – TrueSkill
# Herbrich, Minka & Graepel (2006)
# "TrueSkill: A Bayesian Skill Rating System", NIPS 2006
# https://www.microsoft.com/en-us/research/wp-content/uploads/2007/01/NIPS2006_0688.pdf
# ---------------------------------------------------------------------------

def calculate_mmr_change_trueskill(winners, losers, beta=TS_BETA, tau=TS_TAU):
    """TrueSkill update for a two-team match (winners beat losers).

    Uses approximate message passing on a factor graph with truncated
    Gaussian correction functions V and W.  Each player's mmr (mu),
    sigma, and display_rp are updated in-place.
    """
    all_players = winners + losers

    # Dynamics: add small variance each game to model skill drift
    for p in all_players:
        p.sigma = math.sqrt(p.sigma ** 2 + tau ** 2)

    # Team skill totals
    mu_w = sum(p.mmr for p in winners)
    mu_l = sum(p.mmr for p in losers)

    # Total variance of the team-performance difference
    # c^2 = sum_all(sigma_i^2) + n_total * beta^2
    n = len(all_players)
    c_sq = sum(p.sigma ** 2 for p in all_players) + n * beta ** 2
    c = math.sqrt(c_sq)

    # Normalised mean performance difference (winners minus losers)
    t = (mu_w - mu_l) / c

    # Truncated-Gaussian correction factors
    v = _v_func(t)
    w = _w_func(t)

    # Update winners (positive mean shift, reduced variance)
    for p in winners:
        mean_adj = (p.sigma ** 2 / c) * v
        var_mult = 1.0 - (p.sigma ** 2 / c_sq) * w
        p.mmr = max(MIN_MMR, p.mmr + mean_adj)
        p.sigma *= math.sqrt(max(var_mult, 1e-6))
        p.update_display_rp()

    # Update losers (negative mean shift, reduced variance)
    for p in losers:
        mean_adj = (p.sigma ** 2 / c) * v
        var_mult = 1.0 - (p.sigma ** 2 / c_sq) * w
        p.mmr = max(MIN_MMR, p.mmr - mean_adj)
        p.sigma *= math.sqrt(max(var_mult, 1e-6))
        p.update_display_rp()


# ---------------------------------------------------------------------------
# Rating algorithm 2 – OpenSkill (Plackett-Luce)
# Joshy (2024)
# "OpenSkill: A faster asymmetric multi-team, multiplayer rating system"
# Journal of Open Source Software, 9, 5901
# https://arxiv.org/abs/2401.05451
# ---------------------------------------------------------------------------

def calculate_mmr_change_openskill(winners, losers, beta=OS_BETA, kappa=OS_KAPPA):
    """OpenSkill (Plackett-Luce / Thurstone-Mosteller) update for a
    two-team match (winners beat losers).

    Computes pairwise team win probabilities and updates each player
    via the gradient of the Plackett-Luce log-likelihood and its
    Fisher information.  Each player's mmr (mu), sigma, and display_rp
    are updated in-place.
    """
    teams = [winners, losers]
    ranks = [1, 2]  # rank 1 = winner, rank 2 = loser

    # Pre-compute team-level aggregates
    team_mu = [sum(p.mmr for p in team) for team in teams]
    team_sigma_sq = [sum(p.sigma ** 2 for p in team) for team in teams]

    for i, team in enumerate(teams):
        omega_sum = 0.0   # mean-update accumulator
        delta_sum = 0.0   # variance-update accumulator

        for q in range(len(teams)):
            if q == i:
                continue

            # Pairwise total variance of performance difference
            # c^2 = sigma^2_team_i + sigma^2_team_q + (n_i + n_q) * beta^2
            n_pair = len(teams[i]) + len(teams[q])
            c_sq = team_sigma_sq[i] + team_sigma_sq[q] + n_pair * beta ** 2
            c = math.sqrt(c_sq)

            # Win probability for team i over team q
            p_iq = _Phi((team_mu[i] - team_mu[q]) / c)

            # Expected score based on actual outcome
            if ranks[i] < ranks[q]:
                s_iq = 1.0   # team i won
            elif ranks[i] > ranks[q]:
                s_iq = 0.0   # team i lost
            else:
                s_iq = 0.5   # draw

            # Gradient of Plackett-Luce log-likelihood
            omega_sum += (s_iq - p_iq) / c

            # Fisher information (controls variance shrinkage)
            delta_sum += p_iq * (1.0 - p_iq) / c_sq

        # Apply per-player updates within this team
        for p in team:
            sig_sq = p.sigma ** 2

            # Mean update – proportional to player's own uncertainty
            p.mmr = max(MIN_MMR, p.mmr + sig_sq * omega_sum)

            # Variance update – reduce uncertainty, floor at kappa
            new_sig_sq = sig_sq * max(1.0 - sig_sq * delta_sum, kappa)
            p.sigma = math.sqrt(new_sig_sq)

            p.update_display_rp()


class server:
    def __init__(self):
        self.players = [player() for _ in range(10000)]

    def iterate(self, rating_func=calculate_mmr_change_trueskill, beta=TS_BETA):
        """Run one iteration: select 9500 players, form 950 matches of 10,
        decide winners by skill-based probability, and update ratings.

        Args:
            rating_func: The rating function to use.  Pass
                calculate_mmr_change_trueskill  (default) or
                calculate_mmr_change_openskill.
            beta: Performance variance per player used for win probability.
        """
        selected = random.sample(self.players, 9500)
        random.shuffle(selected)
        for i in range(0, 9500, 10):
            match = selected[i:i + 10]
            random.shuffle(match)
            team_a = match[:5]
            team_b = match[5:]

            # Win probability for team_a based on MMR difference
            mu_a = sum(p.mmr for p in team_a)
            mu_b = sum(p.mmr for p in team_b)
            c = math.sqrt(sum(p.sigma ** 2 for p in match) + 10 * beta ** 2)
            p_a_wins = _Phi((mu_a - mu_b) / c)

            if random.random() < p_a_wins:
                winners, losers = team_a, team_b
            else:
                winners, losers = team_b, team_a

            rating_func(winners, losers)

    def visualize_mmr(self):
        """Display a histogram of the current MMR distribution across all players."""
        mmrs = [p.mmr for p in self.players]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(mmrs, bins=80, edgecolor="black", linewidth=0.4, color="#4C72B0")

        mean_mmr = sum(mmrs) / len(mmrs)
        median_mmr = sorted(mmrs)[len(mmrs) // 2]
        min_mmr = min(mmrs)
        max_mmr = max(mmrs)

        ax.axvline(mean_mmr, color="red", linestyle="--", linewidth=1.2, label=f"Mean: {mean_mmr:.0f}")
        ax.axvline(median_mmr, color="orange", linestyle=":", linewidth=1.2, label=f"Median: {median_mmr:.0f}")

        ax.set_title("MMR Distribution", fontsize=14)
        ax.set_xlabel("MMR", fontsize=12)
        ax.set_ylabel("Number of Players", fontsize=12)
        ax.legend(fontsize=10)
        ax.text(0.98, 0.95,
                f"Players: {len(mmrs)}\nMin: {min_mmr:.0f}\nMax: {max_mmr:.0f}",
                transform=ax.transAxes, fontsize=9,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        plt.tight_layout()
        plt.show()

    def collect_training_data(self, n_iters=200,
                              rating_func=calculate_mmr_change_trueskill,
                              beta=TS_BETA):
        """Run n_iters iterations while recording per-match features and labels.

        Also updates player ratings exactly like iterate(), so the two methods
        can be used interchangeably or chained.

        Returns:
            X (list[list[float]]): 8-D feature vectors, one per match.
            y (list[int]):         Quality labels (0=Close, 1=Competitive, 2=Stomp).
        """
        X, y = [], []
        for _ in range(n_iters):
            selected = random.sample(self.players, 9500)
            random.shuffle(selected)
            for i in range(0, 9500, 10):
                match  = selected[i:i + 10]
                random.shuffle(match)
                team_a = match[:5]
                team_b = match[5:]

                features = _extract_features(team_a, team_b, beta)
                X.append(features)
                y.append(_label_match(features[-1]))  # features[-1] == p_win

                # Decide winner with the same probability already in features
                if random.random() < features[-1]:
                    winners, losers = team_a, team_b
                else:
                    winners, losers = team_b, team_a
                rating_func(winners, losers)
        return X, y


if __name__ == "__main__":
    s_trueskill = server()
    s_openskill = server()

    for i in range(10000):
        s_trueskill.iterate(rating_func=calculate_mmr_change_trueskill)
        s_openskill.iterate(rating_func=calculate_mmr_change_openskill)
        if (i + 1) % 1000 == 0:
            print(f"Iteration {i + 1}/10000")

    print("\n--- TrueSkill MMR Distribution ---")
    s_trueskill.visualize_mmr()

    print("\n--- OpenSkill MMR Distribution ---")
    s_openskill.visualize_mmr()
