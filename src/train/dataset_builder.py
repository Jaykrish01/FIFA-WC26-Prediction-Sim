from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd

from tournament_weights import get_tournament_weight


DEFAULT_WARMUP_START = "2000-01-01"
DEFAULT_MODEL_START = "2009-01-01"
DEFAULT_CUTOFF_EXCLUSIVE = "2026-06-11"


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a neutral-venue World Cup training dataset with Elo and "
            "rolling team features from historical international results."
        )
    )
    parser.add_argument("--results", default="results.csv", help="Path to results.csv")
    parser.add_argument(
        "--output-dir",
        default="build",
        help="Directory where the generated CSV files will be written",
    )
    parser.add_argument(
        "--warmup-start",
        default=DEFAULT_WARMUP_START,
        help="Inclusive start date for the warm-up window",
    )
    parser.add_argument(
        "--model-start",
        default=DEFAULT_MODEL_START,
        help="Inclusive start date for the training dataset window",
    )
    parser.add_argument(
        "--cutoff-exclusive",
        default=DEFAULT_CUTOFF_EXCLUSIVE,
        help="Exclusive upper date bound for all processed matches",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=10,
        help="Recent-match window used for rolling form features",
    )
    parser.add_argument(
        "--base-elo",
        type=float,
        default=1500.0,
        help="Starting Elo for teams with no prior history in the selected window",
    )
    parser.add_argument(
        "--base-k",
        type=float,
        default=24.0,
        help="Base K-factor before tournament weighting and goal-margin adjustment",
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.01,
        help="Exclude matches whose tournament weight is below this threshold",
    )
    return parser.parse_args()


def expected_score(rating_for: float, rating_against: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_against - rating_for) / 400.0))


def result_value(goals_for: int, goals_against: int) -> float:
    if goals_for > goals_against:
        return 1.0
    if goals_for < goals_against:
        return 0.0
    return 0.5


def points_value(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for < goals_against:
        return 0
    return 1


def margin_multiplier(goal_diff: int, elo_diff: float) -> float:
    if goal_diff <= 1:
        return 1.0
    return math.log(goal_diff + 1.0) * (2.2 / (abs(elo_diff) * 0.001 + 2.2))


class TeamState:
    def __init__(self, base_elo: float, lookback: int) -> None:
        self.elo = float(base_elo)
        self.matches_played = 0
        self.wins = 0
        self.draws = 0
        self.losses = 0
        self.goals_for = 0
        self.goals_against = 0
        self.total_points = 0
        self.total_abs_goal_diff = 0
        self.total_match_goals = 0
        self.clean_sheets = 0
        self.failed_to_score = 0
        self.both_teams_scored_matches = 0
        self.low_scoring_matches = 0
        self.tight_matches = 0
        self.scoreless_draws = 0
        self.weighted_points_total = 0.0
        self.weighted_goal_diff_total = 0.0
        self.weighted_match_total = 0.0
        self.elo_delta_total = 0.0
        self.last_match_date: pd.Timestamp | None = None
        self.win_streak = 0
        self.unbeaten_streak = 0
        self.recent = deque(maxlen=lookback)


def build_team_features(state: TeamState, match_date: pd.Timestamp, prefix: str) -> dict[str, float]:
    recent_matches = len(state.recent)
    recent_points = sum(item["points"] for item in state.recent)
    recent_goal_diff = sum(item["goal_diff"] for item in state.recent)
    recent_abs_goal_diff = sum(item["abs_goal_diff"] for item in state.recent)
    recent_goals_for = sum(item["goals_for"] for item in state.recent)
    recent_goals_against = sum(item["goals_against"] for item in state.recent)
    recent_wins = sum(item["win"] for item in state.recent)
    recent_draws = sum(item["draw"] for item in state.recent)
    recent_losses = sum(item["loss"] for item in state.recent)
    recent_clean_sheets = sum(item["clean_sheet"] for item in state.recent)
    recent_failed_to_score = sum(item["failed_to_score"] for item in state.recent)
    recent_both_teams_scored = sum(item["both_teams_scored"] for item in state.recent)
    recent_low_scoring = sum(item["low_scoring"] for item in state.recent)
    recent_tight_matches = sum(item["tight_match"] for item in state.recent)
    recent_scoreless_draws = sum(item["scoreless_draw"] for item in state.recent)
    recent_total_goals = sum(item["total_goals"] for item in state.recent)
    recent_weight_sum = sum(item["weight"] for item in state.recent)
    recent_weighted_points = sum(item["points"] * item["weight"] for item in state.recent)
    recent_weighted_goal_diff = sum(item["goal_diff"] * item["weight"] for item in state.recent)
    recent_opponent_elo = sum(item["opponent_elo"] for item in state.recent)
    recent_elo_delta = sum(item["elo_delta"] for item in state.recent)

    days_since_last_match = (
        float((match_date - state.last_match_date).days)
        if state.last_match_date is not None
        else 9999.0
    )

    features = {
        f"{prefix}_elo": state.elo,
        f"{prefix}_matches_played": float(state.matches_played),
        f"{prefix}_days_since_last_match": days_since_last_match,
        f"{prefix}_has_history": 1.0 if state.matches_played > 0 else 0.0,
        f"{prefix}_points_per_match_all": safe_div(state.total_points, state.matches_played),
        f"{prefix}_goal_diff_all": safe_div(
            state.goals_for - state.goals_against, state.matches_played
        ),
        f"{prefix}_abs_goal_diff_all": safe_div(state.total_abs_goal_diff, state.matches_played),
        f"{prefix}_goals_scored_all": safe_div(state.goals_for, state.matches_played),
        f"{prefix}_goals_conceded_all": safe_div(state.goals_against, state.matches_played),
        f"{prefix}_total_goals_all": safe_div(state.total_match_goals, state.matches_played),
        f"{prefix}_win_rate_all": safe_div(state.wins, state.matches_played),
        f"{prefix}_draw_rate_all": safe_div(state.draws, state.matches_played),
        f"{prefix}_loss_rate_all": safe_div(state.losses, state.matches_played),
        f"{prefix}_clean_sheet_rate_all": safe_div(state.clean_sheets, state.matches_played),
        f"{prefix}_failed_to_score_rate_all": safe_div(state.failed_to_score, state.matches_played),
        f"{prefix}_both_teams_scored_rate_all": safe_div(
            state.both_teams_scored_matches, state.matches_played
        ),
        f"{prefix}_low_scoring_rate_all": safe_div(state.low_scoring_matches, state.matches_played),
        f"{prefix}_tight_match_rate_all": safe_div(state.tight_matches, state.matches_played),
        f"{prefix}_scoreless_draw_rate_all": safe_div(state.scoreless_draws, state.matches_played),
        f"{prefix}_weighted_points_per_match_all": safe_div(
            state.weighted_points_total, state.weighted_match_total
        ),
        f"{prefix}_weighted_goal_diff_all": safe_div(
            state.weighted_goal_diff_total, state.weighted_match_total
        ),
        f"{prefix}_elo_delta_avg_all": safe_div(state.elo_delta_total, state.matches_played),
        f"{prefix}_win_streak": float(state.win_streak),
        f"{prefix}_unbeaten_streak": float(state.unbeaten_streak),
        f"{prefix}_recent_matches": float(recent_matches),
        f"{prefix}_team_form_10": safe_div(recent_points, recent_matches),
        f"{prefix}_goal_diff_10": safe_div(recent_goal_diff, recent_matches),
        f"{prefix}_abs_goal_diff_10": safe_div(recent_abs_goal_diff, recent_matches),
        f"{prefix}_goals_scored_10": safe_div(recent_goals_for, recent_matches),
        f"{prefix}_goals_conceded_10": safe_div(recent_goals_against, recent_matches),
        f"{prefix}_total_goals_10": safe_div(recent_total_goals, recent_matches),
        f"{prefix}_win_rate_10": safe_div(recent_wins, recent_matches),
        f"{prefix}_draw_rate_10": safe_div(recent_draws, recent_matches),
        f"{prefix}_loss_rate_10": safe_div(recent_losses, recent_matches),
        f"{prefix}_clean_sheet_rate_10": safe_div(recent_clean_sheets, recent_matches),
        f"{prefix}_failed_to_score_rate_10": safe_div(recent_failed_to_score, recent_matches),
        f"{prefix}_both_teams_scored_rate_10": safe_div(
            recent_both_teams_scored, recent_matches
        ),
        f"{prefix}_low_scoring_rate_10": safe_div(recent_low_scoring, recent_matches),
        f"{prefix}_tight_match_rate_10": safe_div(recent_tight_matches, recent_matches),
        f"{prefix}_scoreless_draw_rate_10": safe_div(recent_scoreless_draws, recent_matches),
        f"{prefix}_weighted_form_10": safe_div(recent_weighted_points, recent_weight_sum),
        f"{prefix}_weighted_goal_diff_10": safe_div(
            recent_weighted_goal_diff, recent_weight_sum
        ),
        f"{prefix}_opponent_elo_avg_10": safe_div(recent_opponent_elo, recent_matches),
        f"{prefix}_elo_delta_avg_10": safe_div(recent_elo_delta, recent_matches),
    }
    features[f"{prefix}_momentum_10_vs_all"] = (
        features[f"{prefix}_team_form_10"] - features[f"{prefix}_points_per_match_all"]
    )
    return features


def build_h2h_features(
    pair_history: dict[tuple[str, str], deque],
    team_a: str,
    team_b: str,
    match_date: pd.Timestamp,
) -> dict[str, float]:
    key = tuple(sorted((team_a, team_b)))
    history = pair_history[key]
    if not history:
        return {
            "h2h_matches_5": 0.0,
            "h2h_team_a_points_per_match_5": 0.0,
            "h2h_team_a_goal_diff_5": 0.0,
            "h2h_team_a_goals_scored_5": 0.0,
            "h2h_team_a_goals_conceded_5": 0.0,
            "h2h_draw_rate_5": 0.0,
            "h2h_avg_total_goals_5": 0.0,
            "h2h_both_teams_scored_rate_5": 0.0,
            "h2h_scoreless_draw_rate_5": 0.0,
            "h2h_days_since_last_meeting": 9999.0,
        }

    points_total = 0.0
    goal_diff_total = 0.0
    goals_scored_total = 0.0
    goals_conceded_total = 0.0
    draw_total = 0.0
    total_goals_total = 0.0
    both_teams_scored_total = 0.0
    scoreless_draw_total = 0.0
    last_meeting_date = history[-1]["date"]

    for item in history:
        if item["team_a"] == team_a:
            goals_scored = item["score_a"]
            goals_conceded = item["score_b"]
        else:
            goals_scored = item["score_b"]
            goals_conceded = item["score_a"]

        points_total += points_value(goals_scored, goals_conceded)
        goal_diff_total += goals_scored - goals_conceded
        goals_scored_total += goals_scored
        goals_conceded_total += goals_conceded
        draw_total += 1.0 if goals_scored == goals_conceded else 0.0
        total_goals_total += goals_scored + goals_conceded
        both_teams_scored_total += 1.0 if goals_scored > 0 and goals_conceded > 0 else 0.0
        scoreless_draw_total += 1.0 if goals_scored == 0 and goals_conceded == 0 else 0.0

    matches = float(len(history))
    return {
        "h2h_matches_5": matches,
        "h2h_team_a_points_per_match_5": safe_div(points_total, matches),
        "h2h_team_a_goal_diff_5": safe_div(goal_diff_total, matches),
        "h2h_team_a_goals_scored_5": safe_div(goals_scored_total, matches),
        "h2h_team_a_goals_conceded_5": safe_div(goals_conceded_total, matches),
        "h2h_draw_rate_5": safe_div(draw_total, matches),
        "h2h_avg_total_goals_5": safe_div(total_goals_total, matches),
        "h2h_both_teams_scored_rate_5": safe_div(both_teams_scored_total, matches),
        "h2h_scoreless_draw_rate_5": safe_div(scoreless_draw_total, matches),
        "h2h_days_since_last_meeting": float((match_date - last_meeting_date).days),
    }


def build_diff_features(row: dict[str, float]) -> dict[str, float]:
    return {
        "elo_diff": row["team_a_elo"] - row["team_b_elo"],
        "matches_played_diff": row["team_a_matches_played"] - row["team_b_matches_played"],
        "rest_days_diff": row["team_a_days_since_last_match"]
        - row["team_b_days_since_last_match"],
        "team_form_10_diff": row["team_a_team_form_10"] - row["team_b_team_form_10"],
        "goal_diff_10_diff": row["team_a_goal_diff_10"] - row["team_b_goal_diff_10"],
        "abs_goal_diff_10_diff": row["team_a_abs_goal_diff_10"]
        - row["team_b_abs_goal_diff_10"],
        "goals_scored_diff_10": row["team_a_goals_scored_10"]
        - row["team_b_goals_scored_10"],
        "goals_conceded_diff_10": row["team_a_goals_conceded_10"]
        - row["team_b_goals_conceded_10"],
        "weighted_form_10_diff": row["team_a_weighted_form_10"]
        - row["team_b_weighted_form_10"],
        "weighted_goal_diff_10_diff": row["team_a_weighted_goal_diff_10"]
        - row["team_b_weighted_goal_diff_10"],
        "opponent_elo_avg_10_diff": row["team_a_opponent_elo_avg_10"]
        - row["team_b_opponent_elo_avg_10"],
        "win_rate_10_diff": row["team_a_win_rate_10"] - row["team_b_win_rate_10"],
        "draw_rate_10_diff": row["team_a_draw_rate_10"] - row["team_b_draw_rate_10"],
        "clean_sheet_rate_10_diff": row["team_a_clean_sheet_rate_10"]
        - row["team_b_clean_sheet_rate_10"],
        "failed_to_score_rate_10_diff": row["team_a_failed_to_score_rate_10"]
        - row["team_b_failed_to_score_rate_10"],
        "both_teams_scored_rate_10_diff": row["team_a_both_teams_scored_rate_10"]
        - row["team_b_both_teams_scored_rate_10"],
        "low_scoring_rate_10_diff": row["team_a_low_scoring_rate_10"]
        - row["team_b_low_scoring_rate_10"],
        "tight_match_rate_10_diff": row["team_a_tight_match_rate_10"]
        - row["team_b_tight_match_rate_10"],
        "scoreless_draw_rate_10_diff": row["team_a_scoreless_draw_rate_10"]
        - row["team_b_scoreless_draw_rate_10"],
        "momentum_10_vs_all_diff": row["team_a_momentum_10_vs_all"]
        - row["team_b_momentum_10_vs_all"],
        "points_per_match_all_diff": row["team_a_points_per_match_all"]
        - row["team_b_points_per_match_all"],
        "goal_diff_all_diff": row["team_a_goal_diff_all"] - row["team_b_goal_diff_all"],
        "win_rate_all_diff": row["team_a_win_rate_all"] - row["team_b_win_rate_all"],
        "draw_rate_all_diff": row["team_a_draw_rate_all"] - row["team_b_draw_rate_all"],
        "unbeaten_streak_diff": row["team_a_unbeaten_streak"] - row["team_b_unbeaten_streak"],
        "win_streak_diff": row["team_a_win_streak"] - row["team_b_win_streak"],
    }


def build_interaction_features(row: dict[str, float]) -> dict[str, float]:
    abs_elo_diff = abs(row["elo_diff"])
    abs_points_per_match_all_diff = abs(row["points_per_match_all_diff"])
    abs_goal_diff_all_diff = abs(row["goal_diff_all_diff"])
    abs_team_form_10_diff = abs(row["team_form_10_diff"])
    abs_goal_diff_10_diff = abs(row["goal_diff_10_diff"])
    abs_goals_scored_diff_10 = abs(row["goals_scored_diff_10"])
    abs_goals_conceded_diff_10 = abs(row["goals_conceded_diff_10"])
    abs_weighted_form_10_diff = abs(row["weighted_form_10_diff"])
    abs_weighted_goal_diff_10_diff = abs(row["weighted_goal_diff_10_diff"])
    abs_opponent_elo_avg_10_diff = abs(row["opponent_elo_avg_10_diff"])
    abs_win_rate_10_diff = abs(row["win_rate_10_diff"])
    abs_draw_rate_10_diff = abs(row["draw_rate_10_diff"])
    abs_clean_sheet_rate_10_diff = abs(row["clean_sheet_rate_10_diff"])
    abs_failed_to_score_rate_10_diff = abs(row["failed_to_score_rate_10_diff"])
    abs_both_teams_scored_rate_10_diff = abs(row["both_teams_scored_rate_10_diff"])
    abs_low_scoring_rate_10_diff = abs(row["low_scoring_rate_10_diff"])
    abs_tight_match_rate_10_diff = abs(row["tight_match_rate_10_diff"])
    abs_scoreless_draw_rate_10_diff = abs(row["scoreless_draw_rate_10_diff"])
    abs_draw_rate_all_diff = abs(row["draw_rate_all_diff"])

    combined_goals_scored_10 = row["team_a_goals_scored_10"] + row["team_b_goals_scored_10"]
    combined_goals_conceded_10 = (
        row["team_a_goals_conceded_10"] + row["team_b_goals_conceded_10"]
    )
    combined_total_goals_10 = row["team_a_total_goals_10"] + row["team_b_total_goals_10"]
    combined_goals_scored_all = (
        row["team_a_goals_scored_all"] + row["team_b_goals_scored_all"]
    )
    combined_goals_conceded_all = (
        row["team_a_goals_conceded_all"] + row["team_b_goals_conceded_all"]
    )
    combined_total_goals_all = row["team_a_total_goals_all"] + row["team_b_total_goals_all"]
    combined_draw_rate_10 = (row["team_a_draw_rate_10"] + row["team_b_draw_rate_10"]) / 2.0
    combined_draw_rate_all = (row["team_a_draw_rate_all"] + row["team_b_draw_rate_all"]) / 2.0
    combined_clean_sheet_rate_10 = (
        row["team_a_clean_sheet_rate_10"] + row["team_b_clean_sheet_rate_10"]
    ) / 2.0
    combined_failed_to_score_rate_10 = (
        row["team_a_failed_to_score_rate_10"] + row["team_b_failed_to_score_rate_10"]
    ) / 2.0
    combined_btts_rate_10 = (
        row["team_a_both_teams_scored_rate_10"] + row["team_b_both_teams_scored_rate_10"]
    ) / 2.0
    combined_low_scoring_rate_10 = (
        row["team_a_low_scoring_rate_10"] + row["team_b_low_scoring_rate_10"]
    ) / 2.0
    combined_tight_match_rate_10 = (
        row["team_a_tight_match_rate_10"] + row["team_b_tight_match_rate_10"]
    ) / 2.0
    combined_scoreless_draw_rate_10 = (
        row["team_a_scoreless_draw_rate_10"] + row["team_b_scoreless_draw_rate_10"]
    ) / 2.0
    combined_clean_sheet_rate_all = (
        row["team_a_clean_sheet_rate_all"] + row["team_b_clean_sheet_rate_all"]
    ) / 2.0
    combined_failed_to_score_rate_all = (
        row["team_a_failed_to_score_rate_all"] + row["team_b_failed_to_score_rate_all"]
    ) / 2.0
    combined_btts_rate_all = (
        row["team_a_both_teams_scored_rate_all"] + row["team_b_both_teams_scored_rate_all"]
    ) / 2.0
    combined_low_scoring_rate_all = (
        row["team_a_low_scoring_rate_all"] + row["team_b_low_scoring_rate_all"]
    ) / 2.0
    combined_tight_match_rate_all = (
        row["team_a_tight_match_rate_all"] + row["team_b_tight_match_rate_all"]
    ) / 2.0
    combined_scoreless_draw_rate_all = (
        row["team_a_scoreless_draw_rate_all"] + row["team_b_scoreless_draw_rate_all"]
    ) / 2.0

    expected_drawness = 1.0 - abs(row["elo_expected_a"] - 0.5) * 2.0
    low_event_draw_index = expected_drawness * (
        combined_clean_sheet_rate_10 + combined_failed_to_score_rate_10
    )
    draw_pressure_index = (
        0.35 * expected_drawness
        + 0.25 * combined_draw_rate_10
        + 0.15 * combined_clean_sheet_rate_10
        + 0.15 * combined_low_scoring_rate_10
        + 0.10 * row["h2h_draw_rate_5"]
    )

    return {
        "abs_elo_diff": abs_elo_diff,
        "expected_drawness": expected_drawness,
        "abs_points_per_match_all_diff": abs_points_per_match_all_diff,
        "abs_goal_diff_all_diff": abs_goal_diff_all_diff,
        "abs_team_form_10_diff": abs_team_form_10_diff,
        "abs_goal_diff_10_diff": abs_goal_diff_10_diff,
        "abs_goals_scored_diff_10": abs_goals_scored_diff_10,
        "abs_goals_conceded_diff_10": abs_goals_conceded_diff_10,
        "abs_weighted_form_10_diff": abs_weighted_form_10_diff,
        "abs_weighted_goal_diff_10_diff": abs_weighted_goal_diff_10_diff,
        "abs_opponent_elo_avg_10_diff": abs_opponent_elo_avg_10_diff,
        "abs_win_rate_10_diff": abs_win_rate_10_diff,
        "abs_draw_rate_10_diff": abs_draw_rate_10_diff,
        "abs_clean_sheet_rate_10_diff": abs_clean_sheet_rate_10_diff,
        "abs_failed_to_score_rate_10_diff": abs_failed_to_score_rate_10_diff,
        "abs_both_teams_scored_rate_10_diff": abs_both_teams_scored_rate_10_diff,
        "abs_low_scoring_rate_10_diff": abs_low_scoring_rate_10_diff,
        "abs_tight_match_rate_10_diff": abs_tight_match_rate_10_diff,
        "abs_scoreless_draw_rate_10_diff": abs_scoreless_draw_rate_10_diff,
        "abs_draw_rate_all_diff": abs_draw_rate_all_diff,
        "combined_draw_rate_10": combined_draw_rate_10,
        "combined_draw_rate_all": combined_draw_rate_all,
        "combined_goals_scored_10": combined_goals_scored_10,
        "combined_goals_conceded_10": combined_goals_conceded_10,
        "combined_total_goals_10": combined_total_goals_10,
        "combined_goals_scored_all": combined_goals_scored_all,
        "combined_goals_conceded_all": combined_goals_conceded_all,
        "combined_total_goals_all": combined_total_goals_all,
        "combined_clean_sheet_rate_10": combined_clean_sheet_rate_10,
        "combined_failed_to_score_rate_10": combined_failed_to_score_rate_10,
        "combined_both_teams_scored_rate_10": combined_btts_rate_10,
        "combined_low_scoring_rate_10": combined_low_scoring_rate_10,
        "combined_tight_match_rate_10": combined_tight_match_rate_10,
        "combined_scoreless_draw_rate_10": combined_scoreless_draw_rate_10,
        "combined_clean_sheet_rate_all": combined_clean_sheet_rate_all,
        "combined_failed_to_score_rate_all": combined_failed_to_score_rate_all,
        "combined_both_teams_scored_rate_all": combined_btts_rate_all,
        "combined_low_scoring_rate_all": combined_low_scoring_rate_all,
        "combined_tight_match_rate_all": combined_tight_match_rate_all,
        "combined_scoreless_draw_rate_all": combined_scoreless_draw_rate_all,
        "low_event_draw_index": low_event_draw_index,
        "draw_pressure_index": draw_pressure_index,
        "h2h_abs_goal_diff_5": abs(row["h2h_team_a_goal_diff_5"]),
        "h2h_draw_pressure": row["h2h_draw_rate_5"] * expected_drawness,
    }


def load_matches(
    results_path: Path,
    warmup_start: pd.Timestamp,
    cutoff_exclusive: pd.Timestamp,
    min_weight: float,
) -> pd.DataFrame:
    df = pd.read_csv(results_path)
    required_columns = {
        "date",
        "team_a",
        "team_b",
        "score_a",
        "score_b",
        "tournament",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"results.csv is missing required columns: {sorted(missing)}")

    df["row_id"] = range(len(df))
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["tournament_weight"] = df["tournament"].map(get_tournament_weight).astype(float)
    df = df[(df["date"] >= warmup_start) & (df["date"] < cutoff_exclusive)].copy()
    df = df[df["tournament_weight"] >= min_weight].copy()
    df = df.sort_values(["date", "row_id"]).reset_index(drop=True)
    return df


def update_state(
    state: TeamState,
    match_date: pd.Timestamp,
    goals_for: int,
    goals_against: int,
    weight: float,
    opponent_elo: float,
    elo_delta: float,
) -> None:
    points = points_value(goals_for, goals_against)
    goal_diff = goals_for - goals_against

    state.matches_played += 1
    state.goals_for += goals_for
    state.goals_against += goals_against
    state.total_points += points
    state.total_abs_goal_diff += abs(goal_diff)
    state.total_match_goals += goals_for + goals_against
    state.weighted_points_total += points * weight
    state.weighted_goal_diff_total += goal_diff * weight
    state.weighted_match_total += weight
    state.elo_delta_total += elo_delta
    state.clean_sheets += 1 if goals_against == 0 else 0
    state.failed_to_score += 1 if goals_for == 0 else 0
    state.both_teams_scored_matches += 1 if goals_for > 0 and goals_against > 0 else 0
    state.low_scoring_matches += 1 if goals_for + goals_against <= 2 else 0
    state.tight_matches += 1 if abs(goal_diff) <= 1 else 0
    state.scoreless_draws += 1 if goals_for == 0 and goals_against == 0 else 0

    if goals_for > goals_against:
        state.wins += 1
        state.win_streak += 1
        state.unbeaten_streak += 1
    elif goals_for == goals_against:
        state.draws += 1
        state.win_streak = 0
        state.unbeaten_streak += 1
    else:
        state.losses += 1
        state.win_streak = 0
        state.unbeaten_streak = 0

    state.recent.append(
        {
            "points": points,
            "goal_diff": goal_diff,
            "abs_goal_diff": abs(goal_diff),
            "goals_for": goals_for,
            "goals_against": goals_against,
            "weight": weight,
            "opponent_elo": opponent_elo,
            "elo_delta": elo_delta,
            "win": 1 if goals_for > goals_against else 0,
            "draw": 1 if goals_for == goals_against else 0,
            "loss": 1 if goals_for < goals_against else 0,
            "clean_sheet": 1 if goals_against == 0 else 0,
            "failed_to_score": 1 if goals_for == 0 else 0,
            "both_teams_scored": 1 if goals_for > 0 and goals_against > 0 else 0,
            "low_scoring": 1 if goals_for + goals_against <= 2 else 0,
            "tight_match": 1 if abs(goal_diff) <= 1 else 0,
            "scoreless_draw": 1 if goals_for == 0 and goals_against == 0 else 0,
            "total_goals": goals_for + goals_against,
        }
    )
    state.last_match_date = match_date
    state.elo += elo_delta


def build_datasets(
    matches: pd.DataFrame,
    model_start: pd.Timestamp,
    base_elo: float,
    base_k: float,
    lookback: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    team_states: dict[str, TeamState] = {}
    pair_history: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=5))
    output_rows: list[dict[str, object]] = []

    def get_state(team_name: str) -> TeamState:
        if team_name not in team_states:
            team_states[team_name] = TeamState(base_elo=base_elo, lookback=lookback)
        return team_states[team_name]

    for match_date, day_matches in matches.groupby("date", sort=True):
        pending_updates: list[dict[str, object]] = []

        for row in day_matches.itertuples(index=False):
            team_a_state = get_state(row.team_a)
            team_b_state = get_state(row.team_b)

            expected_a = expected_score(team_a_state.elo, team_b_state.elo)
            expected_b = 1.0 - expected_a
            actual_a = result_value(int(row.score_a), int(row.score_b))
            actual_b = 1.0 - actual_a
            margin = margin_multiplier(
                goal_diff=abs(int(row.score_a) - int(row.score_b)),
                elo_diff=team_a_state.elo - team_b_state.elo,
            )
            k_value = base_k * float(row.tournament_weight) * margin
            elo_delta_a = k_value * (actual_a - expected_a)
            elo_delta_b = k_value * (actual_b - expected_b)

            match_features: dict[str, object] = {
                "date": match_date.strftime("%Y-%m-%d"),
                "team_a": row.team_a,
                "team_b": row.team_b,
                "tournament": row.tournament,
                "tournament_weight": float(row.tournament_weight),
                "score_a": int(row.score_a),
                "score_b": int(row.score_b),
                "score_diff": int(row.score_a) - int(row.score_b),
                "abs_score_diff": abs(int(row.score_a) - int(row.score_b)),
                "total_goals": int(row.score_a) + int(row.score_b),
                "result_code": 1 if row.score_a > row.score_b else (-1 if row.score_a < row.score_b else 0),
                "target_a_win": 1 if row.score_a > row.score_b else 0,
                "target_draw": 1 if row.score_a == row.score_b else 0,
                "target_b_win": 1 if row.score_a < row.score_b else 0,
                "elo_expected_a": expected_a,
                "elo_expected_b": expected_b,
                "elo_delta_a_post_match": elo_delta_a,
                "elo_delta_b_post_match": elo_delta_b,
                "is_warmup": 1 if match_date < model_start else 0,
            }
            match_features.update(build_team_features(team_a_state, match_date, "team_a"))
            match_features.update(build_team_features(team_b_state, match_date, "team_b"))
            match_features.update(build_h2h_features(pair_history, row.team_a, row.team_b, match_date))
            match_features.update(build_diff_features(match_features))
            match_features.update(build_interaction_features(match_features))
            output_rows.append(match_features)

            pending_updates.append(
                {
                    "team_a": row.team_a,
                    "team_b": row.team_b,
                    "score_a": int(row.score_a),
                    "score_b": int(row.score_b),
                    "weight": float(row.tournament_weight),
                    "elo_a_before": team_a_state.elo,
                    "elo_b_before": team_b_state.elo,
                    "elo_delta_a": elo_delta_a,
                    "elo_delta_b": elo_delta_b,
                    "date": match_date,
                }
            )

        for update in pending_updates:
            team_a_state = get_state(str(update["team_a"]))
            team_b_state = get_state(str(update["team_b"]))
            update_state(
                state=team_a_state,
                match_date=update["date"],
                goals_for=int(update["score_a"]),
                goals_against=int(update["score_b"]),
                weight=float(update["weight"]),
                opponent_elo=float(update["elo_b_before"]),
                elo_delta=float(update["elo_delta_a"]),
            )
            update_state(
                state=team_b_state,
                match_date=update["date"],
                goals_for=int(update["score_b"]),
                goals_against=int(update["score_a"]),
                weight=float(update["weight"]),
                opponent_elo=float(update["elo_a_before"]),
                elo_delta=float(update["elo_delta_b"]),
            )

            pair_key = tuple(sorted((str(update["team_a"]), str(update["team_b"]))))
            pair_history[pair_key].append(
                {
                    "date": update["date"],
                    "team_a": str(update["team_a"]),
                    "team_b": str(update["team_b"]),
                    "score_a": int(update["score_a"]),
                    "score_b": int(update["score_b"]),
                }
            )

    features_df = pd.DataFrame(output_rows)
    warmup_df = features_df[features_df["is_warmup"] == 1].copy()
    model_df = features_df[features_df["is_warmup"] == 0].copy()

    ratings_rows = []
    for team_name, state in sorted(team_states.items(), key=lambda item: item[1].elo, reverse=True):
        recent_len = len(state.recent)
        recent_points = sum(item["points"] for item in state.recent)
        recent_goal_diff = sum(item["goal_diff"] for item in state.recent)
        recent_goals_for = sum(item["goals_for"] for item in state.recent)
        recent_goals_against = sum(item["goals_against"] for item in state.recent)
        recent_wins = sum(item["win"] for item in state.recent)
        recent_draws = sum(item["draw"] for item in state.recent)
        recent_losses = sum(item["loss"] for item in state.recent)
        recent_clean_sheets = sum(item["clean_sheet"] for item in state.recent)
        recent_failed_to_score = sum(item["failed_to_score"] for item in state.recent)
        recent_both_teams_scored = sum(item["both_teams_scored"] for item in state.recent)
        recent_low_scoring = sum(item["low_scoring"] for item in state.recent)
        recent_tight_matches = sum(item["tight_match"] for item in state.recent)
        recent_scoreless_draws = sum(item["scoreless_draw"] for item in state.recent)
        recent_weight_sum = sum(item["weight"] for item in state.recent)
        recent_weighted_points = sum(item["points"] * item["weight"] for item in state.recent)
        recent_weighted_goal_diff = sum(item["goal_diff"] * item["weight"] for item in state.recent)
        recent_opponent_elo = sum(item["opponent_elo"] for item in state.recent)
        recent_elo_delta = sum(item["elo_delta"] for item in state.recent)
        ratings_rows.append(
            {
                "team": team_name,
                "elo": state.elo,
                "matches_played": state.matches_played,
                "wins": state.wins,
                "draws": state.draws,
                "losses": state.losses,
                "points_per_match_all": safe_div(state.total_points, state.matches_played),
                "goals_scored_all": safe_div(state.goals_for, state.matches_played),
                "goals_conceded_all": safe_div(state.goals_against, state.matches_played),
                "goal_diff_all": safe_div(
                    state.goals_for - state.goals_against, state.matches_played
                ),
                "abs_goal_diff_all": safe_div(state.total_abs_goal_diff, state.matches_played),
                "total_goals_all": safe_div(state.total_match_goals, state.matches_played),
                "clean_sheet_rate_all": safe_div(state.clean_sheets, state.matches_played),
                "failed_to_score_rate_all": safe_div(state.failed_to_score, state.matches_played),
                "both_teams_scored_rate_all": safe_div(
                    state.both_teams_scored_matches, state.matches_played
                ),
                "low_scoring_rate_all": safe_div(state.low_scoring_matches, state.matches_played),
                "tight_match_rate_all": safe_div(state.tight_matches, state.matches_played),
                "scoreless_draw_rate_all": safe_div(state.scoreless_draws, state.matches_played),
                "win_rate_all": safe_div(state.wins, state.matches_played),
                "draw_rate_all": safe_div(state.draws, state.matches_played),
                "loss_rate_all": safe_div(state.losses, state.matches_played),
                "weighted_points_per_match_all": safe_div(
                    state.weighted_points_total, state.weighted_match_total
                ),
                "weighted_goal_diff_all": safe_div(
                    state.weighted_goal_diff_total, state.weighted_match_total
                ),
                "elo_delta_avg_all": safe_div(state.elo_delta_total, state.matches_played),
                "win_streak": float(state.win_streak),
                "unbeaten_streak": float(state.unbeaten_streak),
                "recent_matches": float(recent_len),
                "recent_form_10": safe_div(recent_points, recent_len),
                "recent_goal_diff_10": safe_div(recent_goal_diff, recent_len),
                "recent_abs_goal_diff_10": safe_div(
                    sum(item["abs_goal_diff"] for item in state.recent), recent_len
                ),
                "recent_goals_scored_10": safe_div(recent_goals_for, recent_len),
                "recent_goals_conceded_10": safe_div(recent_goals_against, recent_len),
                "recent_total_goals_10": safe_div(recent_goals_for + recent_goals_against, recent_len),
                "recent_win_rate_10": safe_div(recent_wins, recent_len),
                "recent_draw_rate_10": safe_div(recent_draws, recent_len),
                "recent_loss_rate_10": safe_div(recent_losses, recent_len),
                "recent_clean_sheet_rate_10": safe_div(recent_clean_sheets, recent_len),
                "recent_failed_to_score_rate_10": safe_div(recent_failed_to_score, recent_len),
                "recent_both_teams_scored_rate_10": safe_div(
                    recent_both_teams_scored, recent_len
                ),
                "recent_low_scoring_rate_10": safe_div(recent_low_scoring, recent_len),
                "recent_tight_match_rate_10": safe_div(recent_tight_matches, recent_len),
                "recent_scoreless_draw_rate_10": safe_div(recent_scoreless_draws, recent_len),
                "recent_weighted_form_10": safe_div(recent_weighted_points, recent_weight_sum),
                "recent_weighted_goal_diff_10": safe_div(
                    recent_weighted_goal_diff, recent_weight_sum
                ),
                "recent_opponent_elo_avg_10": safe_div(recent_opponent_elo, recent_len),
                "recent_elo_delta_avg_10": safe_div(recent_elo_delta, recent_len),
                "last_match_date": (
                    state.last_match_date.strftime("%Y-%m-%d")
                    if state.last_match_date is not None
                    else ""
                ),
            }
        )

    ratings_df = pd.DataFrame(ratings_rows)
    return warmup_df, model_df, ratings_df


def main() -> None:
    args = parse_args()
    results_path = Path(args.results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    warmup_start = pd.Timestamp(args.warmup_start)
    model_start = pd.Timestamp(args.model_start)
    cutoff_exclusive = pd.Timestamp(args.cutoff_exclusive)

    if model_start <= warmup_start:
        raise ValueError("--model-start must be later than --warmup-start")
    if cutoff_exclusive <= model_start:
        raise ValueError("--cutoff-exclusive must be later than --model-start")

    matches = load_matches(
        results_path=results_path,
        warmup_start=warmup_start,
        cutoff_exclusive=cutoff_exclusive,
        min_weight=args.min_weight,
    )
    warmup_df, model_df, ratings_df = build_datasets(
        matches=matches,
        model_start=model_start,
        base_elo=args.base_elo,
        base_k=args.base_k,
        lookback=args.lookback,
    )

    warmup_path = output_dir / "warmup_matches_features.csv"
    model_path = output_dir / "model_matches_features.csv"
    ratings_path = output_dir / "team_elo_snapshot.csv"
    metadata_path = output_dir / "dataset_metadata.json"

    warmup_df.to_csv(warmup_path, index=False)
    model_df.to_csv(model_path, index=False)
    ratings_df.to_csv(ratings_path, index=False)

    metadata = {
        "results_path": str(results_path),
        "warmup_start": warmup_start.strftime("%Y-%m-%d"),
        "model_start": model_start.strftime("%Y-%m-%d"),
        "cutoff_exclusive": cutoff_exclusive.strftime("%Y-%m-%d"),
        "lookback": args.lookback,
        "base_elo": args.base_elo,
        "base_k": args.base_k,
        "min_weight": args.min_weight,
        "processed_matches": int(len(matches)),
        "warmup_rows": int(len(warmup_df)),
        "model_rows": int(len(model_df)),
        "teams_with_ratings": int(len(ratings_df)),
        "outputs": {
            "warmup": str(warmup_path),
            "model": str(model_path),
            "ratings": str(ratings_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Processed matches: {len(matches)}")
    print(f"Warm-up rows: {len(warmup_df)} -> {warmup_path}")
    print(f"Model rows: {len(model_df)} -> {model_path}")
    print(f"Team rating snapshot: {len(ratings_df)} -> {ratings_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
