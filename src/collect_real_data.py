"""
collect_real_data.py
Collect ranked 5v5 match data from Riot API for real-data validation.

Pipeline:
  1. Start from a seed PUUID (or fetch by summoner name)
  2. BFS-expand match history to collect N matches
  3. For each match: fetch all 10 participants' tier/LP → approximate MMR
  4. Record pre-match MMR arrays + match quality proxies (duration, gold diff)
  5. Save to real_matches.csv

Usage:
  # First time — provide a summoner name to get a seed PUUID:
  python collect_real_data.py --summoner "SummonerName" --region na1

  # Subsequent runs — use cached PUUID or set SEED_PUUID in .env:
  python collect_real_data.py

Rate limits (dev key): 20 req/s, 100 req/2min  →  script targets ~80 req/2min
"""

import os
import csv
import json
import time
import random
import argparse
import threading
import requests
from urllib.parse import quote
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(_SCRIPT_DIR)
_DATA_DIR   = os.path.join(_REPO_ROOT, "data")

load_dotenv(dotenv_path=os.path.join(_REPO_ROOT, ".env"))

# ── Config ─────────────────────────────────────────────────────────────────────

API_KEY      = os.getenv("RIOT_API_KEY", "")
PLATFORM     = os.getenv("RIOT_PLATFORM", "na1")
REGION       = os.getenv("RIOT_REGION", "americas")
SEED_PUUID   = os.getenv("SEED_PUUID", "")
TARGET       = int(os.getenv("MATCHES_TARGET", 500))
QUEUE_ID     = int(os.getenv("QUEUE_ID", 420))   # 420 = Solo/Duo Ranked

PLATFORM_URL = f"https://{PLATFORM}.api.riotgames.com"
REGION_URL   = f"https://{REGION}.api.riotgames.com"

OUTPUT_FILE  = os.path.join(_DATA_DIR, "real_matches.csv")
PLAYERS_FILE = os.path.join(_DATA_DIR, "players.csv")
SEEN_FILE    = os.path.join(_DATA_DIR, ".seen_matches.json")


def next_output_file(base: str = "real_matches") -> str:
    """Return next non-existent CSV under data/:
    real_matches.csv → real_matches2.csv → real_matches3.csv → …
    """
    os.makedirs(_DATA_DIR, exist_ok=True)
    first = os.path.join(_DATA_DIR, f"{base}.csv")
    if not os.path.exists(first):
        return first
    i = 2
    while True:
        candidate = os.path.join(_DATA_DIR, f"{base}{i}.csv")
        if not os.path.exists(candidate):
            return candidate
        i += 1

# Approximate MMR per tier (base + LP-interpolated range of 400 points per division)
TIER_BASE = {
    "IRON":        400,
    "BRONZE":      800,
    "SILVER":     1200,
    "GOLD":       1600,
    "PLATINUM":   2000,
    "EMERALD":    2400,
    "DIAMOND":    2800,
}

APEX_TIERS  = {"MASTER", "GRANDMASTER", "CHALLENGER"}
RANK_OFFSET = {"I": 300, "II": 200, "III": 100, "IV": 0}

# Apex tier constants.
# Master/Grandmaster/Challenger share one continuous LP ladder with no divisions.
# LP is the direct ranking metric in apex tiers, so 1 LP = 1 MMR (scale = 1.0).
APEX_BASE     = 3200
APEX_LP_SCALE = 1.0

CSV_HEADER = [
    # Team A MMR (5 players)
    "mmr_a1", "mmr_a2", "mmr_a3", "mmr_a4", "mmr_a5",
    # Team B MMR (5 players)
    "mmr_b1", "mmr_b2", "mmr_b3", "mmr_b4", "mmr_b5",
    # Team-level aggregates (direct ML features)
    "avg_sigma_a",      "avg_sigma_b",
    "avg_winrate_a",    "avg_winrate_b",
    # Status flag counts per team (0-5) — from league-v4 entries
    "hot_streak_count_a",  "hot_streak_count_b",
    "fresh_blood_count_a", "fresh_blood_count_b",
    "inactive_count_a",    "inactive_count_b",
    "veteran_count_a",     "veteran_count_b",
    # Derived features (computed from per-player MMR/winrate)
    "winrate_gap",
    "max_mmr_spread_a",
    "max_mmr_spread_b",
    # Ground-truth quality proxies
    "game_duration_s",
    "gold_diff_end",
    "kill_diff",
    "winner",
    # Ground-truth label derived from outcome proxies + dataset role
    "actual_label",     # Close / Competitive / Stomp — from real match outcome
    "split",            # always "validation" — never used for training
    # Metadata
    "match_id",
    "game_version",
]

PLAYER_CSV_HEADER = [
    "puuid", "tier", "rank", "lp", "wins", "losses", "winrate", "sigma", "mmr",
    "hot_streak", "veteran", "fresh_blood", "inactive",
]


# ── Rate-limit tracker ─────────────────────────────────────────────────────────

class RateLimiter:
    """Thread-safe dual-window token bucket for Riot API dev keys.

    Dev key caps:  20 req / 1 s  AND  100 req / 2 min
    We target:     18 req / 1 s  AND   95 req / 2 min  (small safety buffer)

    Both windows must allow a request before it goes through; when either
    is saturated the caller sleeps only as long as needed to free a slot
    in the more restrictive bucket. This lets 10-thread bursts actually
    burst (up to 18 in-flight within any 1-second window) instead of being
    serialized to one-at-a-time by a fixed interval.
    """

    def __init__(self, per_sec: int = 18, per_two_min: int = 95):
        self._per_sec     = per_sec
        self._per_two_min = per_two_min
        self._short: deque[float] = deque()   # timestamps in last 1 s
        self._long:  deque[float] = deque()   # timestamps in last 120 s
        self._lock = threading.Lock()

    def wait(self):
        while True:
            with self._lock:
                now = time.time()
                # Drop timestamps that have aged out
                while self._short and now - self._short[0] >= 1.0:
                    self._short.popleft()
                while self._long and now - self._long[0] >= 120.0:
                    self._long.popleft()

                # If both buckets have room, take a token and return
                if len(self._short) < self._per_sec and len(self._long) < self._per_two_min:
                    self._short.append(now)
                    self._long.append(now)
                    return

                # Otherwise compute the shortest wait until *either* bucket frees a slot
                wait_short = 1.0   - (now - self._short[0]) if len(self._short) >= self._per_sec  else 0.0
                wait_long  = 120.0 - (now - self._long[0])  if len(self._long)  >= self._per_two_min else 0.0
                sleep_for = max(wait_short, wait_long, 0.001)

            time.sleep(sleep_for)


rl = RateLimiter()

# ── Sigma estimation ───────────────────────────────────────────────────────────

SIGMA_TIER_BASE = {
    "IRON": 210, "BRONZE": 195, "SILVER": 180, "GOLD": 165,
    "PLATINUM": 150, "EMERALD": 135, "DIAMOND": 120,
    "MASTER": 95, "GRANDMASTER": 75, "CHALLENGER": 60,
}

def estimate_sigma(tier: str, wins: int, losses: int) -> float:
    """Estimate TrueSkill-style sigma from tier, wins, and losses.

    Two signals:
      - games played  → more games = lower uncertainty (mirrors 1/√n convergence)
      - winrate deviation from 50% → player is still climbing/falling = higher uncertainty
    """
    games = wins + losses
    base  = SIGMA_TIER_BASE.get(tier.upper(), 170)
    games_factor = min(1.5, 50 / (games + 50))         # saturates near 1.0 after ~200 games
    wr = wins / games if games > 0 else 0.5
    wr_factor = 1.0 + abs(wr - 0.5) * 2.0              # 60% WR → +20%, 70% WR → +40%
    return round(base * games_factor * wr_factor, 2)


# ── Per-session player cache ────────────────────────────────────────────────────
# Stores full player data so the same PUUID is never re-fetched within a session.
# Value is None if the player is confirmed unranked (don't retry).
# Value is a dict with keys: mmr, sigma, winrate, tier, rank, lp, wins, losses.
_player_cache: dict[str, dict | None] = {}
_cache_lock = threading.Lock()


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def _get(url: str, params: dict = None, _retry: bool = True) -> dict | None:
    rl.wait()
    headers = {"X-Riot-Token": API_KEY}
    resp = requests.get(url, headers=headers, params=params, timeout=10)

    if resp.status_code == 200:
        return resp.json()
    if resp.status_code == 429 and _retry:
        retry_after = int(resp.headers.get("Retry-After", 10))
        print(f"  [429] Rate limited — sleeping {retry_after}s")
        time.sleep(retry_after)
        return _get(url, params, _retry=False)

    try:
        body = resp.json()
    except Exception:
        body = resp.text[:300]
    short_url = url.split(".api.riotgames.com")[-1]   # trim the host for readability
    print(f"  [HTTP {resp.status_code}] {short_url}  →  {body}")
    return None


# ── Riot API wrappers ──────────────────────────────────────────────────────────

def get_puuid_by_riot_id(summoner_name: str) -> str | None:
    """
    Resolve a Riot ID (GameName#TAG) to PUUID using the Account API.
    The legacy /by-name/ endpoint was removed by Riot in 2023.
    Summoner name must be in 'GameName#TAG' format, e.g. 'Faker#KR1'.
    """
    if "#" not in summoner_name:
        print("ERROR: Summoner must be in 'GameName#TAG' format (e.g. 'PlayerName#NA1').")
        print("       The old summoner name lookup was removed by Riot in 2023.")
        return None

    game_name, tag_line = summoner_name.split("#", 1)
    # URL-encode each component so spaces and special chars don't break the path
    url = f"{REGION_URL}/riot/account/v1/accounts/by-riot-id/{quote(game_name)}/{quote(tag_line)}"
    data = _get(url)
    if data:
        return data.get("puuid")
    print(f"  Could not find account '{summoner_name}'. Check the name and tag are exact.")
    return None


def get_match_ids(puuid: str, count: int = 20) -> list[str]:
    data = _get(
        f"{REGION_URL}/lol/match/v5/matches/by-puuid/{puuid}/ids",
        params={"queue": QUEUE_ID, "type": "ranked", "count": count},
    )
    return data if data else []


def get_match(match_id: str) -> dict | None:
    return _get(f"{REGION_URL}/lol/match/v5/matches/{match_id}")


def get_ranked_entries_by_puuid(puuid: str) -> list[dict]:
    data = _get(f"{PLATFORM_URL}/lol/league/v4/entries/by-puuid/{puuid}")
    return data if data else []


# ── MMR conversion ─────────────────────────────────────────────────────────────

def rank_to_mmr(tier: str, rank: str, lp: int) -> int:
    """Convert tier + division rank + LP to approximate MMR value.

    Diamond and below: base per tier + division offset (I=+300 … IV=+0) +
    LP fraction (100 LP = +100 MMR within the division).

    Apex tiers: Master/GM/Challenger are one continuous ladder.
      mmr = APEX_BASE + lp * APEX_LP_SCALE
    LP is compressed because apex players earn LP faster than lower tiers —
    raw LP addition would overstate the MMR gap between a fresh Master and
    a high-LP Challenger.
    """
    tier = tier.upper()
    if tier in APEX_TIERS:
        return int(APEX_BASE + lp * APEX_LP_SCALE)
    base        = TIER_BASE.get(tier, 1200)
    div_offset  = RANK_OFFSET.get(rank, 0)
    lp_fraction = (lp / 100) * 100    # 100 LP ≈ +100 MMR within a division
    return int(base + div_offset + lp_fraction)


def get_player_data(puuid: str) -> dict | None:
    """Return full solo/duo player data dict or None if unranked.

    Cached per session — same PUUID is never re-fetched.
    Returned dict keys: mmr, sigma, winrate, tier, rank, lp, wins, losses.
    """
    with _cache_lock:
        if puuid in _player_cache:
            return _player_cache[puuid]

    entries = get_ranked_entries_by_puuid(puuid)
    data = None
    for entry in entries:
        if entry.get("queueType") == "RANKED_SOLO_5x5":
            tier   = entry["tier"]
            rank   = entry["rank"]
            lp     = entry["leaguePoints"]
            wins   = entry.get("wins", 0)
            losses = entry.get("losses", 0)
            mmr    = rank_to_mmr(tier, rank, lp)
            wr     = round(wins / (wins + losses), 4) if (wins + losses) > 0 else 0.5
            sigma  = estimate_sigma(tier, wins, losses)
            data   = {
                "mmr": mmr, "sigma": sigma, "winrate": wr,
                "tier": tier, "rank": rank, "lp": lp,
                "wins": wins, "losses": losses,
                "hot_streak":  entry.get("hotStreak",  False),
                "veteran":     entry.get("veteran",    False),
                "fresh_blood": entry.get("freshBlood", False),
                "inactive":    entry.get("inactive",   False),
            }
            break

    with _cache_lock:
        _player_cache[puuid] = data
    return data


def fetch_match_player_data(puuids_a: list[str], puuids_b: list[str]) -> tuple[dict, dict] | None:
    """Fetch player data for all 10 players in a match using one shared thread pool.

    Both teams are submitted simultaneously so the rate limiter slots are consumed
    concurrently — team B's lookups start queuing while team A's are still running,
    saving one full rate-limiter interval (~1.26s) compared to two sequential pools.

    Returns (result_a, result_b) dicts or None if any player is unranked.
    """
    all_puuids = puuids_a + puuids_b
    results: dict[str, dict | None] = {}

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(get_player_data, p): p for p in all_puuids}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()

    if any(results[p] is None for p in all_puuids):
        return None

    def _extract(puuids: list[str]) -> dict:
        return {
            "mmrs":        [results[p]["mmr"]        for p in puuids],
            "sigmas":      [results[p]["sigma"]      for p in puuids],
            "winrates":    [results[p]["winrate"]    for p in puuids],
            "hot_streak":  [results[p]["hot_streak"] for p in puuids],
            "veteran":     [results[p]["veteran"]    for p in puuids],
            "fresh_blood": [results[p]["fresh_blood"]for p in puuids],
            "inactive":    [results[p]["inactive"]   for p in puuids],
        }

    return _extract(puuids_a), _extract(puuids_b)


# ── Match parsing ──────────────────────────────────────────────────────────────

def parse_match(match: dict) -> dict | None:
    """
    Extract features from a match object.
    Returns a row dict or None if the match is unusable (remakes, wrong queue, etc.).
    """
    info = match.get("info", {})

    # Hard filter: ranked solo/duo only (queueId 420)
    if info.get("queueId") != 420:
        return None

    # Skip remakes (< 5 minutes)
    duration = info.get("gameDuration", 0)
    if duration < 300:
        return None

    participants = info.get("participants", [])
    if len(participants) != 10:
        return None

    team100 = [p for p in participants if p["teamId"] == 100]
    team200 = [p for p in participants if p["teamId"] == 200]

    # Gold at end of game
    gold_100 = sum(p.get("goldEarned", 0) for p in team100)
    gold_200 = sum(p.get("goldEarned", 0) for p in team200)
    gold_diff = abs(gold_100 - gold_200)

    # Kill differential
    kills_100 = sum(p.get("kills", 0) for p in team100)
    kills_200 = sum(p.get("kills", 0) for p in team200)
    kill_diff = abs(kills_100 - kills_200)

    # Winner
    teams = info.get("teams", [])
    winner = next((t["teamId"] for t in teams if t.get("win")), None)

    # PUUIDs per team — for MMR lookup
    puuids_a = [p["puuid"] for p in team100]
    puuids_b = [p["puuid"] for p in team200]

    return {
        "puuids_a": puuids_a,
        "puuids_b": puuids_b,
        "game_duration_s": duration,
        "gold_diff_end": gold_diff,
        "kill_diff": kill_diff,
        "winner": winner,
        "match_id": match["metadata"]["matchId"],
        "game_version": info.get("gameVersion", ""),
    }


# ── Tier-based seed sampling ───────────────────────────────────────────────────

# Tiers with four divisions (I–IV)
_DIVISION_TIERS = ["IRON", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND"]
# Apex tiers use dedicated league-roster endpoints (no divisions)
_APEX_ENDPOINTS = {
    "MASTER":      "masterleagues",
    "GRANDMASTER": "grandmasterleagues",
    "CHALLENGER":  "challengerleagues",
}


def _puuid_from_entry(entry: dict) -> str | None:
    """Extract PUUID from a league entry. Tries the direct field first,
    falls back to a summoner lookup if the API hasn't added it yet."""
    puuid = entry.get("puuid")
    if puuid:
        return puuid
    sid = entry.get("summonerId")
    if sid:
        data = _get(f"{PLATFORM_URL}/lol/summoner/v4/summoners/{sid}")
        if data:
            return data.get("puuid")
    return None


def _cache_from_entry(puuid: str, entry: dict, tier: str):
    """Populate _player_cache from a league entry dict.

    League entry responses already carry wins, losses, LP, hotStreak, etc.
    Pre-populating the cache here means these players need zero ranked-lookup
    calls during match collection — the data is already in memory.
    """
    if not puuid or puuid in _player_cache:
        return
    wins   = entry.get("wins", 0)
    losses = entry.get("losses", 0)
    rank   = entry.get("rank", "I")
    lp     = entry.get("leaguePoints", 0)
    mmr    = rank_to_mmr(tier, rank, lp)
    wr     = round(wins / (wins + losses), 4) if (wins + losses) > 0 else 0.5
    sigma  = estimate_sigma(tier, wins, losses)
    with _cache_lock:
        _player_cache[puuid] = {
            "mmr": mmr, "sigma": sigma, "winrate": wr,
            "tier": tier, "rank": rank, "lp": lp,
            "wins": wins, "losses": losses,
            "hot_streak":  entry.get("hotStreak",  False),
            "veteran":     entry.get("veteran",    False),
            "fresh_blood": entry.get("freshBlood", False),
            "inactive":    entry.get("inactive",   False),
        }


def _fetch_division_puuids(tier: str, division: str, count: int) -> list[str]:
    """Return up to `count` PUUIDs from one tier/division page, pre-warming the cache."""
    data = _get(
        f"{PLATFORM_URL}/lol/league/v4/entries/RANKED_SOLO_5x5/{tier}/{division}",
        params={"page": 1},
    )
    if not data:
        return []
    random.shuffle(data)
    puuids = []
    for entry in data:
        if len(puuids) >= count:
            break
        p = _puuid_from_entry(entry)
        if p:
            _cache_from_entry(p, entry, tier)
            puuids.append(p)
    return puuids


def _fetch_apex_puuids(tier: str, count: int) -> list[str]:
    """Return up to `count` PUUIDs from a Master/Grandmaster/Challenger roster, pre-warming the cache."""
    endpoint = _APEX_ENDPOINTS.get(tier.upper())
    if not endpoint:
        return []
    data = _get(f"{PLATFORM_URL}/lol/league/v4/{endpoint}/by-queue/RANKED_SOLO_5x5")
    if not data:
        return []
    entries = data.get("entries", [])
    random.shuffle(entries)
    puuids = []
    for entry in entries:
        if len(puuids) >= count:
            break
        p = _puuid_from_entry(entry)
        if p:
            _cache_from_entry(p, entry, tier)
            puuids.append(p)
    return puuids


def build_tier_seeds(players_per_tier: int = 100) -> list[str]:
    """
    Sample ~players_per_tier PUUIDs from every rank tier and return them
    as a flat list. Covers the full MMR spectrum evenly, avoiding the
    single-seed BFS clustering problem.

    Tier strategy:
      Iron–Diamond  →  division II (representative middle of each tier)
      Master+       →  random sample from full league roster
    """
    all_puuids: list[str] = []

    for tier in _DIVISION_TIERS:
        print(f"  [{tier}] fetching ~{players_per_tier} players…", end=" ", flush=True)
        puuids = _fetch_division_puuids(tier, "II", players_per_tier)
        print(f"got {len(puuids)}")
        all_puuids.extend(puuids)

    for tier in _APEX_ENDPOINTS:
        print(f"  [{tier}] fetching ~{players_per_tier} players…", end=" ", flush=True)
        puuids = _fetch_apex_puuids(tier, players_per_tier)
        print(f"got {len(puuids)}")
        all_puuids.extend(puuids)

    random.shuffle(all_puuids)    # interleave tiers so BFS expands evenly
    print(f"  Total seed PUUIDs: {len(all_puuids)} across {len(_DIVISION_TIERS) + len(_APEX_ENDPOINTS)} tiers\n")
    return all_puuids


# ── Match quality labelling ────────────────────────────────────────────────────

def label_match_outcome(duration_s: int, gold_diff: int, kill_diff: int) -> str:
    """Derive a Close / Competitive / Stomp label from actual match outcome proxies.

    Each signal votes independently; the final label is the worst (most lopsided)
    of the three votes, so a game needs all three signals to be close before being
    called Close — conservative and avoids mislabelling edge cases.

    Thresholds chosen from typical ranked LoL distributions:
      Duration : < 22 min  → Stomp  |  > 35 min   → Close  |  else Competitive
      Gold diff: > 10 000  → Stomp  |  < 4 000    → Close  |  else Competitive
      Kill diff: > 15      → Stomp  |  < 7        → Close  |  else Competitive
    """
    LABEL_ORDER = {"Close": 0, "Competitive": 1, "Stomp": 2}

    def _vote_duration(s: int) -> str:
        if s < 1320:    return "Stomp"          # < 22 min
        if s > 2100:    return "Close"           # > 35 min
        return "Competitive"

    def _vote_gold(g: int) -> str:
        if g > 10_000:  return "Stomp"
        if g < 4_000:   return "Close"
        return "Competitive"

    def _vote_kills(k: int) -> str:
        if k > 15:      return "Stomp"
        if k < 7:       return "Close"
        return "Competitive"

    votes = [_vote_duration(duration_s), _vote_gold(gold_diff), _vote_kills(kill_diff)]
    return max(votes, key=lambda v: LABEL_ORDER[v])


# ── Main collection loop ───────────────────────────────────────────────────────

def load_seen() -> set:
    if os.path.exists(SEEN_FILE):
        with open(SEEN_FILE) as f:
            return set(json.load(f))
    return set()


def save_seen(seen: set):
    with open(SEEN_FILE, "w") as f:
        json.dump(list(seen), f)


def collect(seed_puuids: str | list[str], target: int,
            output_file: str | None = None, reset: bool = False):
    output_file = output_file or OUTPUT_FILE
    if reset:
        for f in [SEEN_FILE, output_file, PLAYERS_FILE]:
            if os.path.exists(f):
                os.remove(f)
                print(f"  Cleared {f}")

    seeds = [seed_puuids] if isinstance(seed_puuids, str) else seed_puuids
    seen_matches = load_seen()
    queue = deque(seeds)
    visited_puuids = set(seeds)
    collected = 0

    # Count existing rows
    write_header = not os.path.exists(output_file)
    if not write_header:
        with open(output_file) as f:
            collected = sum(1 for _ in f) - 1   # subtract header
        print(f"Resuming — {collected} rows saved, {len(seen_matches)} matches in seen cache.")
        if len(seen_matches) > collected * 12 and collected == 0:
            print("  WARNING: seen cache has entries but 0 rows saved — likely a stale cache from a previous crash.")
            print("  Run with --reset to clear and start fresh.")
    else:
        print(f"Starting fresh — seen cache: {len(seen_matches)} entries.")

    out_f = open(output_file, "a", newline="")
    writer = csv.DictWriter(out_f, fieldnames=CSV_HEADER)
    if write_header:
        writer.writeheader()

    write_player_header = not os.path.exists(PLAYERS_FILE)
    player_f  = open(PLAYERS_FILE, "a", newline="")
    player_w  = csv.DictWriter(player_f, fieldnames=PLAYER_CSV_HEADER)
    if write_player_header:
        player_w.writeheader()
    written_players: set[str] = set()   # track who's already in players.csv this session

    skips = {"fetch_failed": 0, "wrong_queue": 0, "remake": 0, "unranked": 0}
    match_times: list[float] = []   # seconds per saved match
    run_start = time.time()

    try:
        while collected < target and queue:
            puuid = queue.popleft()
            print(f"\n[PUUID] {puuid[:20]}…  queue={len(queue)}  saved={collected}/{target}")

            for match_id in [m for m in get_match_ids(puuid) if m not in seen_matches]:
                if collected >= target:
                    break

                seen_matches.add(match_id)

                t_match = time.time()
                match = get_match(match_id)
                if not match:
                    skips["fetch_failed"] += 1
                    continue

                parsed = parse_match(match)
                if not parsed:
                    info = match.get("info", {})
                    if info.get("queueId") != 420:
                        skips["wrong_queue"] += 1
                    elif info.get("gameDuration", 0) < 300:
                        skips["remake"] += 1
                    else:
                        skips["fetch_failed"] += 1
                    continue

                # Fetch solo/duo data for all 10 players in one shared pool
                all_puuids  = parsed["puuids_a"] + parsed["puuids_b"]
                team_result = fetch_match_player_data(parsed["puuids_a"], parsed["puuids_b"])

                if team_result is None:
                    skips["unranked"] += 1
                    continue

                result_a, result_b = team_result
                mmrs_a,     sigmas_a,     winrates_a     = result_a["mmrs"],     result_a["sigmas"],     result_a["winrates"]
                mmrs_b,     sigmas_b,     winrates_b     = result_b["mmrs"],     result_b["sigmas"],     result_b["winrates"]

                # Write new players to players.csv (deduplicated)
                for p_puuid in all_puuids:
                    if p_puuid not in written_players:
                        d = _player_cache.get(p_puuid)
                        if d:
                            player_w.writerow({
                                "puuid":      p_puuid,
                                "tier":       d["tier"],       "rank":       d["rank"],
                                "lp":         d["lp"],         "wins":       d["wins"],
                                "losses":     d["losses"],     "winrate":    d["winrate"],
                                "sigma":      d["sigma"],      "mmr":        d["mmr"],
                                "hot_streak": d["hot_streak"], "veteran":    d["veteran"],
                                "fresh_blood":d["fresh_blood"],"inactive":   d["inactive"],
                            })
                            written_players.add(p_puuid)
                player_f.flush()

                # BFS: enqueue newly discovered players
                for p_puuid in all_puuids:
                    if p_puuid not in visited_puuids:
                        visited_puuids.add(p_puuid)
                        queue.append(p_puuid)

                avg_wr_a = round(sum(winrates_a) / 5, 4)
                avg_wr_b = round(sum(winrates_b) / 5, 4)

                row = {
                    "mmr_a1": mmrs_a[0], "mmr_a2": mmrs_a[1], "mmr_a3": mmrs_a[2],
                    "mmr_a4": mmrs_a[3], "mmr_a5": mmrs_a[4],
                    "mmr_b1": mmrs_b[0], "mmr_b2": mmrs_b[1], "mmr_b3": mmrs_b[2],
                    "mmr_b4": mmrs_b[3], "mmr_b5": mmrs_b[4],
                    "avg_sigma_a":    round(sum(sigmas_a) / 5, 2),
                    "avg_sigma_b":    round(sum(sigmas_b) / 5, 2),
                    "avg_winrate_a":  avg_wr_a,
                    "avg_winrate_b":  avg_wr_b,
                    "hot_streak_count_a":  sum(result_a["hot_streak"]),
                    "hot_streak_count_b":  sum(result_b["hot_streak"]),
                    "fresh_blood_count_a": sum(result_a["fresh_blood"]),
                    "fresh_blood_count_b": sum(result_b["fresh_blood"]),
                    "inactive_count_a":    sum(result_a["inactive"]),
                    "inactive_count_b":    sum(result_b["inactive"]),
                    "veteran_count_a":     sum(result_a["veteran"]),
                    "veteran_count_b":     sum(result_b["veteran"]),
                    "winrate_gap":    round(abs(avg_wr_a - avg_wr_b), 4),
                    "max_mmr_spread_a": max(mmrs_a) - min(mmrs_a),
                    "max_mmr_spread_b": max(mmrs_b) - min(mmrs_b),
                    "game_duration_s":  parsed["game_duration_s"],
                    "gold_diff_end":    parsed["gold_diff_end"],
                    "kill_diff":        parsed["kill_diff"],
                    "winner":           parsed["winner"],
                    "actual_label":     label_match_outcome(
                                            parsed["game_duration_s"],
                                            parsed["gold_diff_end"],
                                            parsed["kill_diff"],
                                        ),
                    "split":            "validation",
                    "match_id":         parsed["match_id"],
                    "game_version":     parsed["game_version"],
                }
                writer.writerow(row)
                out_f.flush()
                collected += 1
                elapsed = time.time() - t_match
                match_times.append(elapsed)
                avg_recent = sum(match_times[-20:]) / len(match_times[-20:])
                label = label_match_outcome(
                    parsed["game_duration_s"], parsed["gold_diff_end"], parsed["kill_diff"]
                )
                print(f"  SAVED [{collected:>4}/{target}]  {match_id}  "
                      f"{parsed['game_duration_s']//60}m  "
                      f"gold±{parsed['gold_diff_end']:,}  "
                      f"A≈{sum(mmrs_a)//5}  B≈{sum(mmrs_b)//5}  "
                      f"[{label}]  "
                      f"{elapsed:.1f}s (avg {avg_recent:.1f}s)")

            save_seen(seen_matches)

    finally:
        out_f.close()
        player_f.close()
        save_seen(seen_matches)
        ranked_cached = sum(1 for v in _player_cache.values() if v is not None)
        run_elapsed = time.time() - run_start
        print(f"\nDone. {collected} matches saved to {output_file}")
        print(f"      {len(written_players)} players saved to {PLAYERS_FILE}")
        if match_times:
            avg = sum(match_times) / len(match_times)
            print(f"Timing  — total {run_elapsed:.1f}s  |  "
                  f"avg {avg:.2f}s/match  |  "
                  f"min {min(match_times):.2f}s  max {max(match_times):.2f}s  |  "
                  f"throughput {len(match_times) / run_elapsed * 60:.1f} matches/min")
        print(f"Skipped  — fetch-failed={skips['fetch_failed']}  "
              f"wrong-queue={skips['wrong_queue']}  "
              f"remake={skips['remake']}  "
              f"unranked={skips['unranked']}")
        print(f"Cache    — {len(_player_cache)} players looked up, {ranked_cached} ranked")


# ── Apex MMR rescale utility ───────────────────────────────────────────────────

def rescale_apex_mmr(
    csv_path: str = OUTPUT_FILE,
    old_scale: float = 0.5,
    new_scale: float = APEX_LP_SCALE,
) -> int:
    """Rewrite MMR columns in an existing CSV using a new APEX_LP_SCALE.

    For each player MMR >= APEX_BASE, back-calculates the original LP and
    recomputes MMR with new_scale.  Non-apex MMR values are unchanged.
    Also recomputes the derived max_mmr_spread_a/b columns.

    Returns the number of rows updated.
    """
    MMR_COLS = [f"mmr_a{i}" for i in range(1, 6)] + [f"mmr_b{i}" for i in range(1, 6)]

    def _rescale(val: float) -> int:
        if val >= APEX_BASE:
            lp = (val - APEX_BASE) / old_scale
            return int(APEX_BASE + lp * new_scale)
        return int(val)

    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found.")
        return 0

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    updated = 0
    for row in rows:
        new_vals = {}
        changed = False
        for col in MMR_COLS:
            if col not in row:
                continue
            old_val = float(row[col])
            new_val = _rescale(old_val)
            new_vals[col] = new_val
            if new_val != int(old_val):
                changed = True

        if changed:
            row.update({k: str(v) for k, v in new_vals.items()})
            # Recompute derived spread columns
            mmrs_a = [float(row[f"mmr_a{i}"]) for i in range(1, 6)]
            mmrs_b = [float(row[f"mmr_b{i}"]) for i in range(1, 6)]
            row["max_mmr_spread_a"] = str(int(max(mmrs_a) - min(mmrs_a)))
            row["max_mmr_spread_b"] = str(int(max(mmrs_b) - min(mmrs_b)))
            updated += 1

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Rescaled {updated}/{len(rows)} rows  ({old_scale}→{new_scale})  →  {csv_path}")
    return updated


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Collect Riot ranked match data")
    parser.add_argument("--summoner", type=str, default="",
                        help="Summoner name (or Name#TAG) to resolve seed PUUID")
    parser.add_argument("--puuid", type=str, default=SEED_PUUID,
                        help="Seed PUUID directly (overrides --summoner)")
    parser.add_argument("--target", type=int, default=TARGET,
                        help=f"Number of matches to collect (default: {TARGET})")
    parser.add_argument("--platform", type=str, default=PLATFORM,
                        help="Platform routing (default: na1)")
    parser.add_argument("--reset", action="store_true",
                        help="Clear seen cache and output file, start collection from scratch")
    parser.add_argument("--tier-seed", action="store_true",
                        help="Sample ~100 players from each tier (Iron→Challenger) as seeds instead of a single player")
    parser.add_argument("--players-per-tier", type=int, default=100,
                        help="Players to sample per tier when using --tier-seed (default: 100)")
    parser.add_argument("--rescale-apex", action="store_true",
                        help="Rewrite apex-tier MMR values in existing CSV using APEX_LP_SCALE=1.0 (old scale was 0.5)")
    parser.add_argument("--old-scale", type=float, default=0.5,
                        help="Previous APEX_LP_SCALE used when the CSV was collected (default: 0.5)")
    args = parser.parse_args()

    if args.rescale_apex:
        rescale_apex_mmr(csv_path=OUTPUT_FILE, old_scale=args.old_scale, new_scale=APEX_LP_SCALE)
        return

    if not API_KEY or API_KEY.startswith("RGAPI-xxx"):
        print("ERROR: Set RIOT_API_KEY in .env before running.")
        return

    # Resolve seed PUUID
    seed = args.puuid
    if not seed and args.summoner:
        print(f"Resolving PUUID for '{args.summoner}'…")
        seed = get_puuid_by_riot_id(args.summoner)
        if not seed:
            print("ERROR: Could not resolve Riot ID.")
            return
        print(f"  PUUID: {seed}")
        print(f"  Tip: add SEED_PUUID={seed} to .env to skip this step next time.\n")

    # Pick next free file name (real_matches.csv → real_matches2.csv → …)
    # When --reset is passed we overwrite the base file instead of auto-incrementing.
    output_file = OUTPUT_FILE if args.reset else next_output_file()

    if args.tier_seed:
        print(f"Building tier-based seeds ({args.players_per_tier} players/tier)…")
        seeds = build_tier_seeds(args.players_per_tier)
        if not seeds:
            print("ERROR: Could not fetch any seeds from tier endpoints.")
            return
        print(f"Starting collection — target: {args.target} matches, queue: {QUEUE_ID}")
        print(f"Output: {output_file}\n")
        collect(seeds, args.target, output_file=output_file, reset=args.reset)
        return

    if not seed:
        print("ERROR: Provide --summoner, set SEED_PUUID in .env, or use --tier-seed.")
        return

    print(f"Starting collection — target: {args.target} matches, queue: {QUEUE_ID}")
    print(f"Output: {output_file}\n")
    collect(seed, args.target, output_file=output_file, reset=args.reset)


if __name__ == "__main__":
    main()
