from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from predict import (
    DRAW_CLASS,
    apply_market_adjustment,
    align_to_feature_order,
    build_feature_snapshot,
    load_feature_order,
    load_market_values,
    load_team_snapshots,
    load_model,
    resolve_team_name,
)
from tournament_weights import get_tournament_weight


TARGET_LABELS = {0: "team_b_win", 1: "draw", 2: "team_a_win"}
TARGET_ORDER = [0, 1, 2]
WORLD_CUP_TOURNAMENT_WEIGHT = get_tournament_weight("FIFA World Cup")
ROUND_DAY_OFFSETS = (0, 5, 10)
ROUND_PAIRINGS = (
    ((0, 1), (2, 3)),
    ((0, 2), (1, 3)),
    ((0, 3), (1, 2)),
)
DEFAULT_MATCH_DATE = "2026-06-11"
MAX_SCORELINE_GOALS = 6


@dataclass
class FixtureResult:
    group: str
    round_number: int
    match_date: str
    team_a: str
    team_b: str
    team_a_win_prob: float
    draw_prob: float
    team_b_win_prob: float
    expected_goals_a: float
    expected_goals_b: float
    expected_goal_diff: float
    most_likely_score_a: int
    most_likely_score_b: int
    most_likely_winner: str
    sample_score_a: int
    sample_score_b: int
    sample_winner: str
    sample_goal_diff: int
    sample_team_a_points: int
    sample_team_b_points: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate World Cup group stage with selectable model mode and optional market-value adjustment."
        )
    )
    parser.add_argument(
        "--model",
        choices=["xgboost", "xgboost_market"],
        default="xgboost",
        help="Simulation mode to run. Use xgboost_market to include market-value adjustment.",
    )
    parser.add_argument(
        "--groups-file",
        default="world_cup_groups.csv",
        help="CSV with columns group,team describing the World Cup groups",
    )
    parser.add_argument(
        "--snapshot-file",
        default="build/team_elo_snapshot.csv",
        help="Team snapshot CSV produced by dataset_builder.py",
    )
    parser.add_argument(
        "--feature-order",
        default="build/training/feature_order.json",
        help="Saved feature order JSON from training",
    )
    parser.add_argument(
        "--model-dir",
        default="build/training/models",
        help="Directory containing the trained xgboost model",
    )
    parser.add_argument(
        "--output-dir",
        default="build/group_stage",
        help="Base directory where group-stage outputs will be written",
    )
    parser.add_argument(
        "--market-values",
        default="market_values.csv",
        help="CSV with team market values used when market adjustment is enabled",
    )
    parser.add_argument(
        "--match-date",
        default=DEFAULT_MATCH_DATE,
        help="Base World Cup start date used for the round schedule",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=10000,
        help="Monte Carlo runs per group",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed for sampling scorelines",
    )
    parser.add_argument(
        "--group",
        default=None,
        help="Optional single group label to simulate",
    )
    return parser.parse_args()


def load_groups(groups_path: Path) -> dict[str, list[str]]:
    df = pd.read_csv(groups_path)
    required = {"group", "team"}
    if not required.issubset(df.columns):
        raise ValueError(f"{groups_path} must contain columns: {sorted(required)}")

    df["group"] = df["group"].astype(str).str.strip().str.upper()
    groups: dict[str, list[str]] = {}
    for group_name, group_df in df.groupby("group", sort=False):
        teams = [resolve_team_name(str(team)) for team in group_df["team"].tolist()]
        if len(teams) != 4:
            raise ValueError(f"Group {group_name} must contain exactly 4 teams")
        if len(set(teams)) != 4:
            raise ValueError(f"Group {group_name} contains duplicate teams after normalization")
        groups[group_name] = teams
    return groups


def apply_draw_boost(proba: np.ndarray, draw_boost: float) -> np.ndarray:
    adjusted = np.asarray(proba, dtype=float).copy()
    adjusted[DRAW_CLASS] *= draw_boost
    adjusted /= adjusted.sum()
    return adjusted


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def value(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    raw = row.get(key, default)
    if pd.isna(raw):
        return default
    return float(raw)


def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0.0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam**k) / math.factorial(k)


def most_likely_scoreline(lambda_a: float, lambda_b: float, max_goals: int = MAX_SCORELINE_GOALS) -> tuple[int, int, float]:
    best_score = (0, 0)
    best_probability = -1.0
    pmf_a = [poisson_pmf(goals, lambda_a) for goals in range(max_goals + 1)]
    pmf_b = [poisson_pmf(goals, lambda_b) for goals in range(max_goals + 1)]
    for goals_a in range(max_goals + 1):
        for goals_b in range(max_goals + 1):
            probability = pmf_a[goals_a] * pmf_b[goals_b]
            if probability > best_probability:
                best_score = (goals_a, goals_b)
                best_probability = probability
    return best_score[0], best_score[1], best_probability


def estimate_goal_lambdas(row: dict[str, Any], probs: np.ndarray) -> tuple[float, float, float]:
    p_b_win, p_draw, p_a_win = map(float, probs)
    elo_gap = value(row, "team_a_elo") - value(row, "team_b_elo")
    form_gap = value(row, "team_a_team_form_10") - value(row, "team_b_team_form_10")
    goal_diff_gap = value(row, "team_a_goal_diff_10") - value(row, "team_b_goal_diff_10")
    points_gap = value(row, "team_a_points_per_match_all") - value(row, "team_b_points_per_match_all")
    weighted_form_gap = value(row, "team_a_weighted_form_10") - value(row, "team_b_weighted_form_10")

    recent_total_avg = 0.5 * (value(row, "team_a_total_goals_10") + value(row, "team_b_total_goals_10"))
    all_total_avg = 0.5 * (value(row, "team_a_total_goals_all") + value(row, "team_b_total_goals_all"))
    low_scoring_avg = 0.5 * (value(row, "team_a_low_scoring_rate_10") + value(row, "team_b_low_scoring_rate_10"))
    failed_to_score_avg = 0.5 * (
        value(row, "team_a_failed_to_score_rate_10") + value(row, "team_b_failed_to_score_rate_10")
    )
    both_teams_scored_avg = 0.5 * (
        value(row, "team_a_both_teams_scored_rate_10") + value(row, "team_b_both_teams_scored_rate_10")
    )
    clean_sheet_avg = 0.5 * (value(row, "team_a_clean_sheet_rate_10") + value(row, "team_b_clean_sheet_rate_10"))
    scoreless_draw_avg = 0.5 * (
        value(row, "team_a_scoreless_draw_rate_10") + value(row, "team_b_scoreless_draw_rate_10")
    )

    base_total = 0.60 * recent_total_avg + 0.25 * all_total_avg + 0.15 * 2.35
    base_total += 0.12 * (both_teams_scored_avg - 0.45)
    base_total -= 0.20 * (low_scoring_avg - 0.40)
    base_total -= 0.16 * (failed_to_score_avg - 0.16)
    base_total -= 0.10 * (clean_sheet_avg - 0.50)
    base_total -= 0.12 * scoreless_draw_avg
    base_total -= 0.08 * (p_draw - 0.22)
    base_total = clamp(base_total, 1.40, 4.10)

    gap_signal = (
        0.45 * math.tanh(elo_gap / 320.0)
        + 0.20 * math.tanh(form_gap / 1.0)
        + 0.15 * math.tanh(goal_diff_gap / 1.7)
        + 0.10 * math.tanh(points_gap / 0.45)
        + 0.10 * math.tanh(weighted_form_gap / 0.9)
        + 0.90 * (p_a_win - p_b_win)
    )
    gap_signal *= max(0.60, 1.0 - 0.30 * p_draw)
    gap_signal = clamp(gap_signal, -1.75, 1.75)

    lambda_a = max(0.20, (base_total + gap_signal) / 2.0)
    lambda_b = max(0.20, (base_total - gap_signal) / 2.0)
    return base_total, lambda_a, lambda_b


def build_fixture_row(
    team_a: str,
    team_b: str,
    match_date: pd.Timestamp,
    snapshots: dict[str, dict[str, Any]],
    feature_order: list[str],
    market_values: dict[str, float],
) -> tuple[dict[str, float], pd.DataFrame, dict[str, Any]]:
    row, snapshot_meta = build_feature_snapshot(
        team_a_name=team_a,
        team_b_name=team_b,
        match_date=match_date,
        snapshots=snapshots,
        market_values=market_values,
    )
    row["tournament_weight"] = WORLD_CUP_TOURNAMENT_WEIGHT
    feature_df = align_to_feature_order(row, feature_order)
    return row, feature_df, snapshot_meta


def simulate_fixture(
    group_name: str,
    round_number: int,
    match_date: pd.Timestamp,
    team_a: str,
    team_b: str,
    snapshots: dict[str, dict[str, Any]],
    market_values: dict[str, float],
    feature_order: list[str],
    model: Any,
    draw_boost: float,
    use_market_values: bool,
    market_strength: float,
    draw_suppression: float,
    rng: np.random.Generator,
    simulations: int,
) -> tuple[FixtureResult, np.ndarray, np.ndarray]:
    row, feature_df, snapshot_meta = build_fixture_row(
        team_a=team_a,
        team_b=team_b,
        match_date=match_date,
        snapshots=snapshots,
        feature_order=feature_order,
        market_values=market_values,
    )
    proba = model.predict_proba(feature_df)[0]
    proba = apply_draw_boost(proba, draw_boost)
    if use_market_values:
        proba, _ = apply_market_adjustment(
            proba,
            snapshot_meta.get("team_a_market_value_cr_inr"),
            snapshot_meta.get("team_b_market_value_cr_inr"),
            market_strength,
            draw_suppression,
        )
    base_total, lambda_a, lambda_b = estimate_goal_lambdas(row, proba)
    most_likely_a, most_likely_b, _ = most_likely_scoreline(lambda_a, lambda_b)

    goals_a = rng.poisson(lambda_a, size=simulations)
    goals_b = rng.poisson(lambda_b, size=simulations)

    sample_a = int(goals_a[0])
    sample_b = int(goals_b[0])
    sample_goal_diff = sample_a - sample_b
    if sample_a > sample_b:
        sample_winner = team_a
        sample_team_a_points, sample_team_b_points = 3, 0
    elif sample_b > sample_a:
        sample_winner = team_b
        sample_team_a_points, sample_team_b_points = 0, 3
    else:
        sample_winner = "draw"
        sample_team_a_points, sample_team_b_points = 1, 1

    if most_likely_a > most_likely_b:
        most_likely_winner = team_a
    elif most_likely_b > most_likely_a:
        most_likely_winner = team_b
    else:
        most_likely_winner = "draw"

    result = FixtureResult(
        group=group_name,
        round_number=round_number,
        match_date=match_date.strftime("%Y-%m-%d"),
        team_a=team_a,
        team_b=team_b,
        team_a_win_prob=float(proba[2]),
        draw_prob=float(proba[1]),
        team_b_win_prob=float(proba[0]),
        expected_goals_a=float(lambda_a),
        expected_goals_b=float(lambda_b),
        expected_goal_diff=float(lambda_a - lambda_b),
        most_likely_score_a=most_likely_a,
        most_likely_score_b=most_likely_b,
        most_likely_winner=most_likely_winner,
        sample_score_a=sample_a,
        sample_score_b=sample_b,
        sample_winner=sample_winner,
        sample_goal_diff=sample_goal_diff,
        sample_team_a_points=sample_team_a_points,
        sample_team_b_points=sample_team_b_points,
    )
    return result, goals_a, goals_b


def update_team_arrays(
    team_index: dict[str, int],
    team_points: np.ndarray,
    team_goals_for: np.ndarray,
    team_goals_against: np.ndarray,
    goals_a: np.ndarray,
    goals_b: np.ndarray,
    team_a: str,
    team_b: str,
) -> None:
    a_idx = team_index[team_a]
    b_idx = team_index[team_b]

    team_goals_for[a_idx] += goals_a
    team_goals_against[a_idx] += goals_b
    team_goals_for[b_idx] += goals_b
    team_goals_against[b_idx] += goals_a

    team_points[a_idx] += np.where(goals_a > goals_b, 3, np.where(goals_a == goals_b, 1, 0))
    team_points[b_idx] += np.where(goals_b > goals_a, 3, np.where(goals_a == goals_b, 1, 0))


def rank_teams(
    teams: list[str],
    points: dict[str, int],
    goals_for: dict[str, int],
    goals_against: dict[str, int],
    snapshots: dict[str, dict[str, Any]],
) -> list[str]:
    return sorted(
        teams,
        key=lambda team: (
            -points[team],
            -(goals_for[team] - goals_against[team]),
            -goals_for[team],
            -float(snapshots[team].get("elo", 0.0)),
            team,
        ),
    )


def simulate_group(
    group_name: str,
    teams: list[str],
    base_snapshots: dict[str, dict[str, Any]],
    market_values: dict[str, float],
    feature_order: list[str],
    model: Any,
    draw_boost: float,
    use_market_values: bool,
    market_strength: float,
    draw_suppression: float,
    match_date: pd.Timestamp,
    simulations: int,
    rng: np.random.Generator,
) -> tuple[list[FixtureResult], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fixture_results: list[FixtureResult] = []
    working_snapshots = deepcopy({team: base_snapshots[team] for team in teams})

    team_index = {team: idx for idx, team in enumerate(teams)}
    points = np.zeros((len(teams), simulations), dtype=np.int16)
    goals_for = np.zeros((len(teams), simulations), dtype=np.int16)
    goals_against = np.zeros((len(teams), simulations), dtype=np.int16)

    sample_points = {team: 0 for team in teams}
    sample_goals_for = {team: 0 for team in teams}
    sample_goals_against = {team: 0 for team in teams}

    for round_number, pairings in enumerate(ROUND_PAIRINGS, start=1):
        round_date = match_date + pd.Timedelta(days=ROUND_DAY_OFFSETS[round_number - 1])
        for left_idx, right_idx in pairings:
            team_a = teams[left_idx]
            team_b = teams[right_idx]
            result, goals_a, goals_b = simulate_fixture(
                group_name=group_name,
                round_number=round_number,
                match_date=round_date,
                team_a=team_a,
                team_b=team_b,
                snapshots=working_snapshots,
                market_values=market_values,
                feature_order=feature_order,
                model=model,
                draw_boost=draw_boost,
                use_market_values=use_market_values,
                market_strength=market_strength,
                draw_suppression=draw_suppression,
                rng=rng,
                simulations=simulations,
            )
            fixture_results.append(result)
            update_team_arrays(
                team_index=team_index,
                team_points=points,
                team_goals_for=goals_for,
                team_goals_against=goals_against,
                goals_a=goals_a,
                goals_b=goals_b,
                team_a=team_a,
                team_b=team_b,
            )

            sample_goals_for[team_a] += result.sample_score_a
            sample_goals_against[team_a] += result.sample_score_b
            sample_goals_for[team_b] += result.sample_score_b
            sample_goals_against[team_b] += result.sample_score_a
            sample_points[team_a] += result.sample_team_a_points
            sample_points[team_b] += result.sample_team_b_points

        for team in teams:
            working_snapshots[team]["last_match_date"] = round_date

    sample_rows = []
    expected_rows = []
    winner_prob_rows = []

    winner_counts = {team: 0 for team in teams}
    runner_up_counts = {team: 0 for team in teams}
    total_rank = {team: 0.0 for team in teams}

    for sim_index in range(simulations):
        per_team_points = {team: int(points[team_index[team], sim_index]) for team in teams}
        per_team_gf = {team: int(goals_for[team_index[team], sim_index]) for team in teams}
        per_team_ga = {team: int(goals_against[team_index[team], sim_index]) for team in teams}
        ranking = rank_teams(teams, per_team_points, per_team_gf, per_team_ga, base_snapshots)
        winner_counts[ranking[0]] += 1
        runner_up_counts[ranking[1]] += 1
        for rank_idx, team in enumerate(ranking, start=1):
            total_rank[team] += rank_idx

    sample_ranking = rank_teams(teams, sample_points, sample_goals_for, sample_goals_against, base_snapshots)
    for rank_idx, team in enumerate(sample_ranking, start=1):
        sample_rows.append(
            {
                "group": group_name,
                "rank": rank_idx,
                "team": team,
                "points": int(sample_points[team]),
                "goal_difference": int(sample_goals_for[team] - sample_goals_against[team]),
                "goals_for": int(sample_goals_for[team]),
                "goals_against": int(sample_goals_against[team]),
            }
        )

    for team in teams:
        mean_points = float(points[team_index[team]].mean())
        mean_goal_difference = float((goals_for[team_index[team]] - goals_against[team_index[team]]).mean())
        mean_goals_for = float(goals_for[team_index[team]].mean())
        mean_goals_against = float(goals_against[team_index[team]].mean())
        winner_prob = winner_counts[team] / simulations
        runner_up_prob = runner_up_counts[team] / simulations
        top2_prob = winner_prob + runner_up_prob
        expected_rows.append(
            {
                "group": group_name,
                "team": team,
                "expected_rank": total_rank[team] / simulations,
                "expected_points": mean_points,
                "expected_goal_difference": mean_goal_difference,
                "expected_goals_for": mean_goals_for,
                "expected_goals_against": mean_goals_against,
                "group_win_probability": winner_prob,
                "runner_up_probability": runner_up_prob,
                "top2_probability": top2_prob,
            }
        )
        winner_prob_rows.append(
            {
                "group": group_name,
                "team": team,
                "group_win_probability": winner_prob,
                "runner_up_probability": runner_up_prob,
                "top2_probability": top2_prob,
                "mean_points": mean_points,
                "mean_goal_difference": mean_goal_difference,
                "mean_goals_for": mean_goals_for,
                "mean_goals_against": mean_goals_against,
                "mean_rank": total_rank[team] / simulations,
            }
        )

    match_rows = [result.__dict__ for result in fixture_results]
    match_df = pd.DataFrame(match_rows).sort_values(["group", "round_number", "team_a", "team_b"])
    sample_df = pd.DataFrame(sample_rows).sort_values(["rank"])
    expected_df = pd.DataFrame(expected_rows).sort_values(
        ["group_win_probability", "expected_points", "expected_goal_difference"],
        ascending=[False, False, False],
    )
    winners_df = pd.DataFrame(winner_prob_rows).sort_values(
        ["group", "group_win_probability", "top2_probability"], ascending=[True, False, False]
    )
    return fixture_results, match_df, sample_df, expected_df, winners_df


def format_group_block(group_name: str, sample_df: pd.DataFrame) -> str:
    lines = [f"GROUP {group_name}"]
    for _, row in sample_df.iterrows():
        team = str(row["team"])
        points = int(row["points"])
        goal_difference = int(row["goal_difference"])
        lines.append(f"{int(row['rank'])}. {team:<20} {points:>2}  GD {goal_difference:+d}")
    return "\n".join(lines)


def plot_group_standings_chart(
    group_name: str,
    standings_df: pd.DataFrame,
    output_path: Path,
    title_suffix: str,
) -> None:
    if standings_df.empty:
        return

    ordered = standings_df.sort_values(["rank", "team"]).reset_index(drop=True)
    teams = ordered["team"].astype(str).tolist()
    points = ordered["points"].astype(float).to_numpy()
    goal_diff = ordered["goal_difference"].astype(int).to_numpy()

    colors = ["#1d3557", "#457b9d", "#a8dadc", "#e63946"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(teams, points, color=colors[: len(teams)], alpha=0.9)
    ax.invert_yaxis()
    ax.set_xlabel("Points")
    ax.set_title(f"Group {group_name} {title_suffix}")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.set_xlim(0, max(9, int(points.max()) + 2))

    for bar, gd, pts in zip(bars, goal_diff, points, strict=False):
        ax.text(
            bar.get_width() + 0.08,
            bar.get_y() + bar.get_height() / 2,
            f"{pts:.0f} pts  GD {gd:+d}",
            va="center",
            ha="left",
            fontsize=10,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_expected_group_rankings(
    expected_df: pd.DataFrame,
    snapshots: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    ranked = expected_df.copy()
    ranked["elo"] = ranked["team"].map(lambda team: float(snapshots[str(team)].get("elo", 0.0)))
    ranked = ranked.sort_values(
        ["group", "expected_points", "expected_goal_difference", "expected_goals_for", "elo", "team"],
        ascending=[True, False, False, False, False, True],
    ).reset_index(drop=True)
    ranked["rank"] = ranked.groupby("group").cumcount() + 1
    return ranked


def build_third_place_rankings(
    expected_ranked_df: pd.DataFrame,
    snapshots: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    third_place_rows = expected_ranked_df[expected_ranked_df["rank"] == 3].copy()
    if third_place_rows.empty:
        return pd.DataFrame(
            columns=[
                "qualification_rank",
                "group",
                "team",
                "points",
                "goal_difference",
                "goals_for",
                "goals_against",
                "elo",
                "qualified",
            ]
        )

    third_place_rows = third_place_rows.rename(
        columns={
            "expected_points": "points",
            "expected_goal_difference": "goal_difference",
            "expected_goals_for": "goals_for",
            "expected_goals_against": "goals_against",
        }
    )
    third_place_rows["elo"] = third_place_rows["team"].map(
        lambda team: float(snapshots[str(team)].get("elo", 0.0))
    )
    third_place_rows = third_place_rows.sort_values(
        ["points", "goal_difference", "goals_for", "elo", "team"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    third_place_rows.insert(0, "qualification_rank", np.arange(1, len(third_place_rows) + 1))
    third_place_rows["qualified"] = third_place_rows["qualification_rank"] <= 8
    return third_place_rows[
        [
            "qualification_rank",
            "group",
            "team",
            "points",
            "goal_difference",
            "goals_for",
            "goals_against",
            "elo",
            "qualified",
        ]
    ]


def build_round_of_32_qualifiers(
    summary: dict[str, Any],
    third_place_rankings: pd.DataFrame,
) -> pd.DataFrame:
    qualifiers: list[dict[str, Any]] = []
    for group_name, group_info in summary["groups"].items():
        qualifiers.append(
            {
                "source": "expected_group_winner",
                "group": group_name,
                "team": group_info["expected_winner"],
                "position_in_group": 1,
            }
        )
        qualifiers.append(
            {
                "source": "expected_group_runner_up",
                "group": group_name,
                "team": group_info["expected_runner_up"],
                "position_in_group": 2,
            }
        )

    for _, row in third_place_rankings[third_place_rankings["qualified"]].iterrows():
        qualifiers.append(
            {
                "source": "best_third_place",
                "group": str(row["group"]),
                "team": str(row["team"]),
                "position_in_group": 3,
            }
        )

    qualifiers_df = pd.DataFrame(qualifiers)
    if not qualifiers_df.empty:
        qualifiers_df = qualifiers_df.sort_values(
            ["source", "group", "team"],
            ascending=[True, True, True],
        ).reset_index(drop=True)
    return qualifiers_df


def main() -> None:
    args = parse_args()

    groups_path = Path(args.groups_file)
    snapshot_path = Path(args.snapshot_file)
    feature_order_path = Path(args.feature_order)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    mode_dir_name = args.model
    output_dir = output_dir / mode_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = output_dir / "charts"

    groups = load_groups(groups_path)
    if args.group is not None:
        selected_group = str(args.group).strip().upper()
        if selected_group not in groups:
            raise KeyError(f"Group {selected_group} not found in {groups_path}")
        groups = {selected_group: groups[selected_group]}

    feature_order = load_feature_order(feature_order_path, Path("build/model_matches_features.csv"))
    snapshots = load_team_snapshots(snapshot_path)
    use_market_values = args.model == "xgboost_market"
    market_values = load_market_values(Path(args.market_values)) if use_market_values else {}
    xgb_model, xgb_draw_boost = load_model(model_dir, "xgboost")

    rng = np.random.default_rng(args.random_state)
    match_date = pd.Timestamp(args.match_date)

    all_match_rows: list[pd.DataFrame] = []
    all_sample_rows: list[pd.DataFrame] = []
    all_expected_rows: list[pd.DataFrame] = []
    all_winner_rows: list[pd.DataFrame] = []
    group_text_blocks: list[str] = []
    summary: dict[str, Any] = {
        "groups_file": str(groups_path),
        "snapshot_file": str(snapshot_path),
        "feature_order_file": str(feature_order_path),
        "model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "model_mode": args.model,
        "use_market_values": use_market_values,
        "simulations": int(args.simulations),
        "random_state": int(args.random_state),
        "match_date": str(match_date.date()),
        "tournament_weight": WORLD_CUP_TOURNAMENT_WEIGHT,
        "xgboost_draw_boost": float(xgb_draw_boost),
        "groups": {},
        "charts": {},
    }

    for group_name, teams in groups.items():
        _, match_df, sample_df, expected_df, winners_df = simulate_group(
            group_name=group_name,
            teams=teams,
            base_snapshots=snapshots,
            market_values=market_values,
            feature_order=feature_order,
            model=xgb_model,
            draw_boost=xgb_draw_boost,
            use_market_values=use_market_values,
            market_strength=0.06,
            draw_suppression=0.03,
            match_date=match_date,
            simulations=args.simulations,
            rng=rng,
        )
        all_match_rows.append(match_df)
        all_sample_rows.append(sample_df)
        all_expected_rows.append(expected_df)
        all_winner_rows.append(winners_df)
        group_text_blocks.append(format_group_block(group_name, sample_df))

        sample_chart_path = charts_dir / f"group_{group_name}_sample_standings.png"
        expected_chart_path = charts_dir / f"group_{group_name}_expected_standings.png"
        plot_group_standings_chart(
            group_name=group_name,
            standings_df=sample_df,
            output_path=sample_chart_path,
            title_suffix="Sample Standings",
        )
        plot_group_standings_chart(
            group_name=group_name,
            standings_df=expected_df.rename(
                columns={
                    "expected_points": "points",
                    "expected_goal_difference": "goal_difference",
                }
            ).assign(rank=lambda frame: np.arange(1, len(frame) + 1)),
            output_path=expected_chart_path,
            title_suffix="Expected Standings",
        )
        summary["groups"][group_name] = {
            "sample_winner": str(sample_df.iloc[0]["team"]),
            "sample_runner_up": str(sample_df.iloc[1]["team"]),
        }
        summary["charts"][group_name] = {
            "sample_standings": str(sample_chart_path),
            "expected_standings": str(expected_chart_path),
        }

    match_df = pd.concat(all_match_rows, ignore_index=True).sort_values(
        ["group", "round_number", "team_a", "team_b"]
    ).reset_index(drop=True)
    sample_df = pd.concat(all_sample_rows, ignore_index=True).sort_values(
        ["group", "rank", "team"]
    ).reset_index(drop=True)
    expected_df = pd.concat(all_expected_rows, ignore_index=True).sort_values(
        ["group", "expected_rank", "team"]
    ).reset_index(drop=True)
    winners_df = pd.concat(all_winner_rows, ignore_index=True).sort_values(
        ["group", "group_win_probability", "top2_probability"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    expected_ranked_df = build_expected_group_rankings(expected_df=expected_df, snapshots=snapshots)
    expected_df = expected_ranked_df

    for group_name in sorted(groups.keys()):
        expected_group = expected_ranked_df[expected_ranked_df["group"] == group_name]
        summary["groups"][group_name]["expected_winner"] = str(expected_group.iloc[0]["team"])
        summary["groups"][group_name]["expected_runner_up"] = str(expected_group.iloc[1]["team"])

    third_place_df = build_third_place_rankings(expected_ranked_df=expected_ranked_df, snapshots=snapshots)
    round_of_32_df = build_round_of_32_qualifiers(summary=summary, third_place_rankings=third_place_df)

    match_path = output_dir / "group_stage_matches.csv"
    sample_path = output_dir / "group_stage_sample_standings.csv"
    expected_path = output_dir / "group_stage_expected_standings.csv"
    winners_path = output_dir / "group_winners_probability.csv"
    third_place_path = output_dir / "group_stage_third_place_rankings.csv"
    best_third_place_path = output_dir / "group_stage_best_third_place.csv"
    round_of_32_path = output_dir / "round_of_32_qualifiers.csv"
    summary_path = output_dir / "group_stage_summary.json"

    match_df.to_csv(match_path, index=False)
    sample_df.to_csv(sample_path, index=False)
    expected_df.to_csv(expected_path, index=False)
    winners_df.to_csv(winners_path, index=False)
    third_place_df.to_csv(third_place_path, index=False)
    third_place_df[third_place_df["qualified"]].to_csv(best_third_place_path, index=False)
    round_of_32_df.to_csv(round_of_32_path, index=False)
    summary["match_results_path"] = str(match_path)
    summary["sample_standings_path"] = str(sample_path)
    summary["expected_standings_path"] = str(expected_path)
    summary["group_winners_probability_path"] = str(winners_path)
    summary["third_place_rankings_path"] = str(third_place_path)
    summary["best_third_place_path"] = str(best_third_place_path)
    summary["round_of_32_qualifiers_path"] = str(round_of_32_path)
    summary["charts_dir"] = str(charts_dir)
    summary["third_place_qualifiers"] = third_place_df[third_place_df["qualified"]][
        ["group", "team", "points", "goal_difference", "goals_for"]
    ].to_dict(orient="records")
    summary["round_of_32_qualifiers"] = round_of_32_df.to_dict(orient="records")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if use_market_values:
        print("XGBoost with market value adjustment.")
    else:
        print("XGBoost only. No market value adjustment.")
    print(f"World Cup tournament weight: {WORLD_CUP_TOURNAMENT_WEIGHT}")
    print(f"Monte Carlo simulations per group: {args.simulations}")
    print()
    for block in group_text_blocks:
        print(block)
        print()

    print("Group winners probability summary:")
    for group_name in sorted(groups.keys()):
        group_winners = winners_df[winners_df["group"] == group_name].sort_values(
            "group_win_probability", ascending=False
        )
        leader = group_winners.iloc[0]
        print(
            f"GROUP {group_name}: {leader['team']} {leader['group_win_probability']:.1%} "
            f"(top 2: {leader['top2_probability']:.1%})"
        )

    print()
    print(f"Match results written to: {match_path}")
    print(f"Sample standings written to: {sample_path}")
    print(f"Expected standings written to: {expected_path}")
    print(f"Group winners probability written to: {winners_path}")
    print(f"Third-place rankings written to: {third_place_path}")
    print(f"Best third-place teams written to: {best_third_place_path}")
    print(f"Round of 32 qualifiers written to: {round_of_32_path}")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
