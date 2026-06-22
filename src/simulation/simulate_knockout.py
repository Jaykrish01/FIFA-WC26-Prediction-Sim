import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from predict import (
    apply_market_adjustment,
    align_to_feature_order,
    build_feature_snapshot,
    load_feature_order,
    load_market_values,
    load_model,
    load_team_snapshots,
    resolve_team_name,
)
from simulate_group_stage import (
    WORLD_CUP_TOURNAMENT_WEIGHT,
    apply_draw_boost,
    estimate_goal_lambdas,
    most_likely_scoreline,
)


DEFAULT_MATCH_DATE = "2026-06-28"
MAX_SCORELINE_GOALS = 6


@dataclass(frozen=True)
class ThirdPlaceSlotSpec:
    match_id: str
    label: str
    allowed_groups: tuple[str, ...]
    preferred_groups: tuple[str, ...]


THIRD_PLACE_SLOTS: dict[str, ThirdPlaceSlotSpec] = {
    # Preferences mirror the simplified image mapping first, then fall back to
    # other valid groups from the provided FIFA-style slot labels.
    "M74": ThirdPlaceSlotSpec("M74", "Best 3rd ABCDF", tuple("ABCDF"), ("D", "A", "C", "B", "F")),
    "M77": ThirdPlaceSlotSpec("M77", "Best 3rd CDFGH", tuple("CDFGH"), ("G", "C", "D", "F", "H")),
    "M79": ThirdPlaceSlotSpec("M79", "Best 3rd CEFHI", tuple("CEFHI"), ("C", "E", "F", "H", "I")),
    "M80": ThirdPlaceSlotSpec("M80", "Best 3rd EHIJK", tuple("EHIJK"), ("I", "E", "H", "J", "K")),
    "M81": ThirdPlaceSlotSpec("M81", "Best 3rd BEFIJ", tuple("BEFIJ"), ("B", "E", "F", "I", "J")),
    "M82": ThirdPlaceSlotSpec("M82", "Best 3rd AEHIJ", tuple("AEHIJ"), ("A", "E", "H", "I", "J")),
    "M85": ThirdPlaceSlotSpec("M85", "Best 3rd EFGIJ", tuple("EFGIJ"), ("J", "E", "F", "G", "I")),
    "M87": ThirdPlaceSlotSpec("M87", "Best 3rd DEIJL", tuple("DEIJL"), ("L", "D", "E", "I", "J")),
}


ROUND_OF_32_FIXTURES: list[dict[str, Any]] = [
    {"match_id": "M73", "date": "2026-06-28", "a": ("position", "A", 2, "2A"), "b": ("position", "B", 2, "2B")},
    {"match_id": "M74", "date": "2026-06-29", "a": ("position", "E", 1, "1E"), "b": ("third", "M74")},
    {"match_id": "M75", "date": "2026-06-29", "a": ("position", "F", 1, "1F"), "b": ("position", "C", 2, "2C")},
    {"match_id": "M76", "date": "2026-06-29", "a": ("position", "C", 1, "1C"), "b": ("position", "F", 2, "2F")},
    {"match_id": "M77", "date": "2026-06-30", "a": ("position", "I", 1, "1I"), "b": ("third", "M77")},
    {"match_id": "M78", "date": "2026-06-30", "a": ("position", "E", 2, "2E"), "b": ("position", "I", 2, "2I")},
    {"match_id": "M79", "date": "2026-06-30", "a": ("position", "A", 1, "1A"), "b": ("third", "M79")},
    {"match_id": "M80", "date": "2026-07-01", "a": ("position", "L", 1, "1L"), "b": ("third", "M80")},
    {"match_id": "M81", "date": "2026-07-01", "a": ("position", "D", 1, "1D"), "b": ("third", "M81")},
    {"match_id": "M82", "date": "2026-07-01", "a": ("position", "G", 1, "1G"), "b": ("third", "M82")},
    {"match_id": "M83", "date": "2026-07-02", "a": ("position", "K", 2, "2K"), "b": ("position", "L", 2, "2L")},
    {"match_id": "M84", "date": "2026-07-02", "a": ("position", "H", 1, "1H"), "b": ("position", "J", 2, "2J")},
    {"match_id": "M85", "date": "2026-07-02", "a": ("position", "B", 1, "1B"), "b": ("third", "M85")},
    {"match_id": "M86", "date": "2026-07-03", "a": ("position", "J", 1, "1J"), "b": ("position", "H", 2, "2H")},
    {"match_id": "M87", "date": "2026-07-03", "a": ("position", "K", 1, "1K"), "b": ("third", "M87")},
    {"match_id": "M88", "date": "2026-07-03", "a": ("position", "D", 2, "2D"), "b": ("position", "G", 2, "2G")},
]


NEXT_ROUND_FIXTURES: list[dict[str, Any]] = [
    {"round": "round_of_16", "match_id": "M89", "date": "2026-07-04", "a": ("winner", "M74", "W74"), "b": ("winner", "M77", "W77")},
    {"round": "round_of_16", "match_id": "M90", "date": "2026-07-04", "a": ("winner", "M73", "W73"), "b": ("winner", "M75", "W75")},
    {"round": "round_of_16", "match_id": "M91", "date": "2026-07-05", "a": ("winner", "M76", "W76"), "b": ("winner", "M78", "W78")},
    {"round": "round_of_16", "match_id": "M92", "date": "2026-07-05", "a": ("winner", "M79", "W79"), "b": ("winner", "M80", "W80")},
    {"round": "round_of_16", "match_id": "M93", "date": "2026-07-06", "a": ("winner", "M83", "W83"), "b": ("winner", "M84", "W84")},
    {"round": "round_of_16", "match_id": "M94", "date": "2026-07-06", "a": ("winner", "M81", "W81"), "b": ("winner", "M82", "W82")},
    {"round": "round_of_16", "match_id": "M95", "date": "2026-07-07", "a": ("winner", "M86", "W86"), "b": ("winner", "M88", "W88")},
    {"round": "round_of_16", "match_id": "M96", "date": "2026-07-07", "a": ("winner", "M85", "W85"), "b": ("winner", "M87", "W87")},
    {"round": "quarter_final", "match_id": "M97", "date": "2026-07-09", "a": ("winner", "M89", "W89"), "b": ("winner", "M90", "W90")},
    {"round": "quarter_final", "match_id": "M98", "date": "2026-07-09", "a": ("winner", "M93", "W93"), "b": ("winner", "M94", "W94")},
    {"round": "quarter_final", "match_id": "M99", "date": "2026-07-10", "a": ("winner", "M91", "W91"), "b": ("winner", "M92", "W92")},
    {"round": "quarter_final", "match_id": "M100", "date": "2026-07-10", "a": ("winner", "M95", "W95"), "b": ("winner", "M96", "W96")},
    {"round": "semi_final", "match_id": "M101", "date": "2026-07-14", "a": ("winner", "M97", "W97"), "b": ("winner", "M98", "W98")},
    {"round": "semi_final", "match_id": "M102", "date": "2026-07-15", "a": ("winner", "M99", "W99"), "b": ("winner", "M100", "W100")},
    {"round": "third_place_playoff", "match_id": "M103", "date": "2026-07-18", "a": ("loser", "M101", "L101"), "b": ("loser", "M102", "L102")},
    {"round": "final", "match_id": "M104", "date": "2026-07-19", "a": ("winner", "M101", "W101"), "b": ("winner", "M102", "W102")},
]


FEATURE_SNAPSHOT_COLUMNS = [
    "elo_diff",
    "team_form_10_diff",
    "goal_diff_10_diff",
    "goals_scored_10_diff",
    "goals_conceded_10_diff",
    "points_per_match_all_diff",
    "clean_sheet_rate_10_diff",
    "failed_to_score_rate_10_diff",
    "draw_rate_10_diff",
    "low_scoring_rate_10_diff",
    "team_a_elo",
    "team_b_elo",
    "team_a_team_form_10",
    "team_b_team_form_10",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and simulate the World Cup knockout bracket from expected group-stage outputs."
    )
    parser.add_argument(
        "--model",
        choices=["xgboost", "xgboost_market"],
        default="xgboost",
        help="Use plain XGBoost or XGBoost with market-value adjustment.",
    )
    parser.add_argument(
        "--group-stage-dir",
        default=None,
        help="Directory containing group_stage_expected_standings.csv. Defaults to build/group_stage/<model>.",
    )
    parser.add_argument(
        "--snapshot-file",
        default="build/team_elo_snapshot.csv",
        help="Team snapshot CSV produced by dataset_builder.py.",
    )
    parser.add_argument(
        "--feature-order",
        default="build/training/feature_order.json",
        help="Saved feature order JSON from training.",
    )
    parser.add_argument(
        "--model-dir",
        default="build/training/models",
        help="Directory containing the trained xgboost model.",
    )
    parser.add_argument(
        "--market-values",
        default="market_values.csv",
        help="CSV with team market values used only for --model xgboost_market.",
    )
    parser.add_argument(
        "--output-dir",
        default="build/knockout",
        help="Base directory where knockout outputs will be written.",
    )
    parser.add_argument(
        "--winner-mode",
        choices=["expected", "sample"],
        default="expected",
        help="expected chooses higher advance probability; sample draws one scoreline path.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed used for sample knockout paths.",
    )
    parser.add_argument(
        "--market-strength",
        type=float,
        default=0.06,
        help="How strongly market value nudges win probabilities.",
    )
    parser.add_argument(
        "--draw-suppression",
        type=float,
        default=0.03,
        help="How strongly market imbalance suppresses draw probability.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def normalize_team_column(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized["team"] = normalized["team"].map(lambda value: resolve_team_name(str(value)))
    return normalized


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().casefold() in {"true", "1", "yes", "y"}


def rank_expected_standings(expected_df: pd.DataFrame, snapshots: dict[str, dict[str, Any]]) -> pd.DataFrame:
    required = {"group", "team", "expected_points", "expected_goal_difference", "expected_goals_for"}
    missing = required - set(expected_df.columns)
    if missing:
        raise ValueError(f"group_stage_expected_standings.csv is missing columns: {sorted(missing)}")

    ranked = normalize_team_column(expected_df)
    ranked["group"] = ranked["group"].astype(str).str.strip().str.upper()
    ranked["elo"] = ranked["team"].map(lambda team: float(snapshots[str(team)].get("elo", 0.0)))
    ranked = ranked.sort_values(
        ["group", "expected_points", "expected_goal_difference", "expected_goals_for", "elo", "team"],
        ascending=[True, False, False, False, False, True],
    ).reset_index(drop=True)
    ranked["rank"] = ranked.groupby("group").cumcount() + 1
    return ranked


def load_expected_positions(
    group_stage_dir: Path,
    snapshots: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, dict[tuple[str, int], str], dict[str, str]]:
    expected_path = group_stage_dir / "group_stage_expected_standings.csv"
    expected_df = rank_expected_standings(read_csv(expected_path), snapshots)

    positions: dict[tuple[str, int], str] = {}
    team_groups: dict[str, str] = {}
    for row in expected_df.to_dict(orient="records"):
        group = str(row["group"])
        rank = int(row["rank"])
        team = str(row["team"])
        positions[(group, rank)] = team
        team_groups[team] = group

    return expected_df, positions, team_groups


def build_third_place_rankings(
    group_stage_dir: Path,
    expected_df: pd.DataFrame,
) -> pd.DataFrame:
    third_place_path = group_stage_dir / "group_stage_third_place_rankings.csv"
    if third_place_path.exists():
        third_df = normalize_team_column(read_csv(third_place_path))
        third_df["group"] = third_df["group"].astype(str).str.strip().str.upper()
        if "qualified" in third_df.columns:
            third_df = third_df[third_df["qualified"].map(as_bool)].copy()
        elif "qualification_rank" in third_df.columns:
            third_df = third_df[third_df["qualification_rank"].astype(int) <= 8].copy()
        else:
            third_df = third_df.head(8).copy()
    else:
        third_df = expected_df[expected_df["rank"] == 3].copy()
        third_df = third_df.rename(
            columns={
                "expected_points": "points",
                "expected_goal_difference": "goal_difference",
                "expected_goals_for": "goals_for",
                "expected_goals_against": "goals_against",
            }
        )
        third_df = third_df.sort_values(
            ["points", "goal_difference", "goals_for", "elo", "team"],
            ascending=[False, False, False, False, True],
        ).reset_index(drop=True)
        third_df.insert(0, "qualification_rank", np.arange(1, len(third_df) + 1))
        third_df = third_df[third_df["qualification_rank"] <= 8].copy()

    if len(third_df) != 8:
        raise ValueError(f"Expected 8 best third-place teams, found {len(third_df)}")

    if "qualification_rank" not in third_df.columns:
        third_df.insert(0, "qualification_rank", np.arange(1, len(third_df) + 1))

    return third_df.sort_values("qualification_rank").reset_index(drop=True)


def third_place_cost(slot: ThirdPlaceSlotSpec, group: str, qualification_rank: int) -> int:
    if group in slot.preferred_groups:
        preference_cost = slot.preferred_groups.index(group)
    else:
        preference_cost = len(slot.preferred_groups) + slot.allowed_groups.index(group)
    return preference_cost * 100 + qualification_rank


def assign_third_place_slots(third_df: pd.DataFrame) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    third_by_group = {str(row["group"]): row for row in third_df.to_dict(orient="records")}
    qualified_groups = set(third_by_group)
    slots = list(THIRD_PLACE_SLOTS.values())
    candidates = {
        slot.match_id: [group for group in slot.allowed_groups if group in qualified_groups]
        for slot in slots
    }

    empty_slots = [match_id for match_id, groups in candidates.items() if not groups]
    if empty_slots:
        raise ValueError(
            "No qualified third-place candidate for slot(s): "
            + ", ".join(empty_slots)
            + ". Re-run group-stage outputs or update the simplified third-place mapping."
        )

    best_assignment: dict[str, str] | None = None
    best_cost = math.inf

    def recurse(
        remaining_slots: list[ThirdPlaceSlotSpec],
        remaining_groups: set[str],
        assignment: dict[str, str],
        total_cost: int,
    ) -> None:
        nonlocal best_assignment, best_cost
        if total_cost >= best_cost:
            return
        if not remaining_slots:
            if not remaining_groups:
                best_assignment = dict(assignment)
                best_cost = total_cost
            return

        slot = min(
            remaining_slots,
            key=lambda item: len([group for group in candidates[item.match_id] if group in remaining_groups]),
        )
        viable_groups = [group for group in candidates[slot.match_id] if group in remaining_groups]
        if not viable_groups:
            return

        next_slots = [item for item in remaining_slots if item.match_id != slot.match_id]
        viable_groups = sorted(
            viable_groups,
            key=lambda group: third_place_cost(
                slot,
                group,
                int(third_by_group[group].get("qualification_rank", 99)),
            ),
        )
        for group in viable_groups:
            assignment[slot.match_id] = group
            recurse(
                next_slots,
                remaining_groups - {group},
                assignment,
                total_cost
                + third_place_cost(
                    slot,
                    group,
                    int(third_by_group[group].get("qualification_rank", 99)),
                ),
            )
            assignment.pop(slot.match_id, None)

    recurse(slots, qualified_groups, {}, 0)
    if best_assignment is None:
        raise ValueError(
            "Could not map the eight best third-place groups into the simplified Round of 32 slots."
        )

    assignment: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for slot in slots:
        group = best_assignment[slot.match_id]
        row = third_by_group[group]
        assigned = {
            "match_id": slot.match_id,
            "slot_label": slot.label,
            "allowed_groups": "".join(slot.allowed_groups),
            "preferred_groups": "".join(slot.preferred_groups),
            "assigned_group": group,
            "assigned_team": str(row["team"]),
            "qualification_rank": int(row.get("qualification_rank", 0)),
            "points": float(row.get("points", row.get("expected_points", 0.0))),
            "goal_difference": float(row.get("goal_difference", row.get("expected_goal_difference", 0.0))),
            "goals_for": float(row.get("goals_for", row.get("expected_goals_for", 0.0))),
            "assignment_reason": "image_preference"
            if group == slot.preferred_groups[0]
            else "best_available_allowed_group",
        }
        assignment[slot.match_id] = assigned
        rows.append(assigned)

    return assignment, pd.DataFrame(rows)


def resolve_fixture_ref(
    ref: tuple[Any, ...],
    positions: dict[tuple[str, int], str],
    third_assignment: dict[str, dict[str, Any]],
    winners: dict[str, str],
    losers: dict[str, str],
) -> tuple[str, str]:
    ref_type = str(ref[0])
    if ref_type == "position":
        group = str(ref[1])
        rank = int(ref[2])
        label = str(ref[3])
        return positions[(group, rank)], label
    if ref_type == "third":
        match_id = str(ref[1])
        assigned = third_assignment[match_id]
        label = f"3{assigned['assigned_group']} ({assigned['slot_label']})"
        return str(assigned["assigned_team"]), label
    if ref_type == "winner":
        match_id = str(ref[1])
        return winners[match_id], str(ref[2])
    if ref_type == "loser":
        match_id = str(ref[1])
        return losers[match_id], str(ref[2])
    raise ValueError(f"Unknown fixture reference: {ref}")


def poisson_probability(score_a: int, score_b: int, lambda_a: float, lambda_b: float) -> float:
    prob_a = math.exp(-lambda_a) * (lambda_a**score_a) / math.factorial(score_a)
    prob_b = math.exp(-lambda_b) * (lambda_b**score_b) / math.factorial(score_b)
    return float(prob_a * prob_b)


def best_scoreline_for_winner(
    lambda_a: float,
    lambda_b: float,
    team_a_wins: bool,
    max_goals: int = MAX_SCORELINE_GOALS,
) -> tuple[int, int]:
    best_score = (1, 0) if team_a_wins else (0, 1)
    best_probability = -1.0
    for score_a in range(max_goals + 1):
        for score_b in range(max_goals + 1):
            if team_a_wins and score_a <= score_b:
                continue
            if not team_a_wins and score_b <= score_a:
                continue
            probability = poisson_probability(score_a, score_b, lambda_a, lambda_b)
            if probability > best_probability:
                best_score = (score_a, score_b)
                best_probability = probability
    return best_score


def penalty_probability_a(row: dict[str, Any], probs: np.ndarray) -> float:
    p_b_win, _, p_a_win = map(float, probs)
    non_draw_total = max(p_a_win + p_b_win, 1e-9)
    regulation_edge = p_a_win / non_draw_total
    elo_gap = float(row.get("team_a_elo", 0.0)) - float(row.get("team_b_elo", 0.0))
    form_gap = float(row.get("team_a_team_form_10", 0.0)) - float(row.get("team_b_team_form_10", 0.0))
    elo_edge = 1.0 / (1.0 + 10.0 ** (-(elo_gap) / 600.0))
    form_edge = 0.5 + 0.12 * math.tanh(form_gap / 1.1)
    penalty_edge = 0.62 * regulation_edge + 0.28 * elo_edge + 0.10 * form_edge
    return max(0.28, min(0.72, penalty_edge))


def build_match_prediction(
    match_id: str,
    round_name: str,
    match_date: pd.Timestamp,
    team_a: str,
    team_b: str,
    slot_a: str,
    slot_b: str,
    team_groups: dict[str, str],
    snapshots: dict[str, dict[str, Any]],
    market_values: dict[str, float],
    feature_order: list[str],
    model: Any,
    draw_boost: float,
    use_market_values: bool,
    market_strength: float,
    draw_suppression: float,
    winner_mode: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    row, snapshot_meta = build_feature_snapshot(
        team_a_name=team_a,
        team_b_name=team_b,
        match_date=match_date,
        snapshots=snapshots,
        market_values=market_values,
    )
    row["tournament_weight"] = WORLD_CUP_TOURNAMENT_WEIGHT
    feature_df = align_to_feature_order(row, feature_order)
    probs = model.predict_proba(feature_df)[0]
    probs = apply_draw_boost(probs, draw_boost)

    if use_market_values:
        probs, _ = apply_market_adjustment(
            probs,
            snapshot_meta.get("team_a_market_value_cr_inr"),
            snapshot_meta.get("team_b_market_value_cr_inr"),
            market_strength,
            draw_suppression,
        )

    _, lambda_a, lambda_b = estimate_goal_lambdas(row, probs)
    likely_a, likely_b, likely_score_prob = most_likely_scoreline(lambda_a, lambda_b)

    p_b_win, p_draw, p_a_win = map(float, probs)
    penalty_a = penalty_probability_a(row, probs)
    team_a_advance_prob = p_a_win + p_draw * penalty_a
    team_b_advance_prob = p_b_win + p_draw * (1.0 - penalty_a)
    advance_total = team_a_advance_prob + team_b_advance_prob
    team_a_advance_prob /= advance_total
    team_b_advance_prob /= advance_total

    scoreline_method = "sample"
    if winner_mode == "sample":
        score_a = int(rng.poisson(lambda_a))
        score_b = int(rng.poisson(lambda_b))
        if score_a > score_b:
            winner, loser = team_a, team_b
            resolution = "regular_time"
            penalty_score = ""
        elif score_b > score_a:
            winner, loser = team_b, team_a
            resolution = "regular_time"
            penalty_score = ""
        elif rng.random() <= penalty_a:
            winner, loser = team_a, team_b
            resolution = "penalties"
            penalty_score = "5-4"
            scoreline_method = "sample_penalties"
        else:
            winner, loser = team_b, team_a
            resolution = "penalties"
            penalty_score = "4-5"
            scoreline_method = "sample_penalties"
    else:
        team_a_advances = team_a_advance_prob >= team_b_advance_prob
        winner, loser = (team_a, team_b) if team_a_advances else (team_b, team_a)
        draw_is_dominant = p_draw >= max(p_a_win, p_b_win) + 0.06
        if likely_a == likely_b and draw_is_dominant:
            score_a, score_b = likely_a, likely_b
            resolution = "penalties"
            penalty_score = "5-4" if team_a_advances else "4-5"
            scoreline_method = "dominant_draw"
        elif (likely_a > likely_b and team_a_advances) or (likely_b > likely_a and not team_a_advances):
            score_a, score_b = likely_a, likely_b
            resolution = "regular_time"
            penalty_score = ""
            scoreline_method = "most_likely"
        else:
            score_a, score_b = best_scoreline_for_winner(lambda_a, lambda_b, team_a_advances)
            resolution = "regular_time"
            penalty_score = ""
            scoreline_method = "winner_conditioned"

    team_a_group = team_groups.get(resolve_team_name(team_a), "")
    team_b_group = team_groups.get(resolve_team_name(team_b), "")
    feature_snapshot = {key: float(row.get(key, 0.0)) for key in FEATURE_SNAPSHOT_COLUMNS}

    return {
        "round": round_name,
        "match_id": match_id,
        "match_date": match_date.strftime("%Y-%m-%d"),
        "slot_a": slot_a,
        "slot_b": slot_b,
        "team_a": team_a,
        "team_b": team_b,
        "team_a_group": team_a_group,
        "team_b_group": team_b_group,
        "team_a_win_prob": float(p_a_win),
        "draw_prob": float(p_draw),
        "team_b_win_prob": float(p_b_win),
        "team_a_advance_prob": float(team_a_advance_prob),
        "team_b_advance_prob": float(team_b_advance_prob),
        "team_a_penalty_prob_if_draw": float(penalty_a),
        "expected_goals_a": float(lambda_a),
        "expected_goals_b": float(lambda_b),
        "expected_goal_diff": float(lambda_a - lambda_b),
        "most_likely_score_a": int(likely_a),
        "most_likely_score_b": int(likely_b),
        "most_likely_score_probability": float(likely_score_prob),
        "score_a": int(score_a),
        "score_b": int(score_b),
        "scoreline": f"{score_a}-{score_b}",
        "resolution": resolution,
        "penalty_score": penalty_score,
        "scoreline_method": scoreline_method,
        "winner": winner,
        "loser": loser,
        "same_group_match": bool(team_a_group and team_a_group == team_b_group),
        **feature_snapshot,
    }


def simulate_knockout(
    positions: dict[tuple[str, int], str],
    third_assignment: dict[str, dict[str, Any]],
    team_groups: dict[str, str],
    snapshots: dict[str, dict[str, Any]],
    market_values: dict[str, float],
    feature_order: list[str],
    model: Any,
    draw_boost: float,
    use_market_values: bool,
    market_strength: float,
    draw_suppression: float,
    winner_mode: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    winners: dict[str, str] = {}
    losers: dict[str, str] = {}
    rows: list[dict[str, Any]] = []

    for fixture in ROUND_OF_32_FIXTURES:
        team_a, slot_a = resolve_fixture_ref(fixture["a"], positions, third_assignment, winners, losers)
        team_b, slot_b = resolve_fixture_ref(fixture["b"], positions, third_assignment, winners, losers)
        if team_groups.get(team_a) == team_groups.get(team_b):
            raise ValueError(
                f"{fixture['match_id']} creates an immediate same-group rematch: {team_a} vs {team_b}"
            )
        row = build_match_prediction(
            match_id=fixture["match_id"],
            round_name="round_of_32",
            match_date=pd.Timestamp(fixture["date"]),
            team_a=team_a,
            team_b=team_b,
            slot_a=slot_a,
            slot_b=slot_b,
            team_groups=team_groups,
            snapshots=snapshots,
            market_values=market_values,
            feature_order=feature_order,
            model=model,
            draw_boost=draw_boost,
            use_market_values=use_market_values,
            market_strength=market_strength,
            draw_suppression=draw_suppression,
            winner_mode=winner_mode,
            rng=rng,
        )
        winners[fixture["match_id"]] = str(row["winner"])
        losers[fixture["match_id"]] = str(row["loser"])
        rows.append(row)

    for fixture in NEXT_ROUND_FIXTURES:
        team_a, slot_a = resolve_fixture_ref(fixture["a"], positions, third_assignment, winners, losers)
        team_b, slot_b = resolve_fixture_ref(fixture["b"], positions, third_assignment, winners, losers)
        row = build_match_prediction(
            match_id=fixture["match_id"],
            round_name=fixture["round"],
            match_date=pd.Timestamp(fixture["date"]),
            team_a=team_a,
            team_b=team_b,
            slot_a=slot_a,
            slot_b=slot_b,
            team_groups=team_groups,
            snapshots=snapshots,
            market_values=market_values,
            feature_order=feature_order,
            model=model,
            draw_boost=draw_boost,
            use_market_values=use_market_values,
            market_strength=market_strength,
            draw_suppression=draw_suppression,
            winner_mode=winner_mode,
            rng=rng,
        )
        winners[fixture["match_id"]] = str(row["winner"])
        losers[fixture["match_id"]] = str(row["loser"])
        rows.append(row)

    return pd.DataFrame(rows)


def format_bracket_line(row: pd.Series) -> str:
    return (
        f"{row['match_id']}: {row['team_a']} {row['scoreline']} {row['team_b']} "
        f"-> {row['winner']} ({row['resolution']})"
    )


def main() -> None:
    args = parse_args()
    group_stage_dir = Path(args.group_stage_dir) if args.group_stage_dir else Path("build/group_stage") / args.model
    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshots = load_team_snapshots(Path(args.snapshot_file))
    feature_order = load_feature_order(Path(args.feature_order), Path("build/model_matches_features.csv"))
    xgb_model, xgb_draw_boost = load_model(Path(args.model_dir), "xgboost")
    use_market_values = args.model == "xgboost_market"
    market_values = load_market_values(Path(args.market_values)) if use_market_values else {}
    rng = np.random.default_rng(args.random_state)

    expected_df, positions, team_groups = load_expected_positions(group_stage_dir, snapshots)
    third_df = build_third_place_rankings(group_stage_dir, expected_df)
    third_assignment, third_assignment_df = assign_third_place_slots(third_df)

    knockout_df = simulate_knockout(
        positions=positions,
        third_assignment=third_assignment,
        team_groups=team_groups,
        snapshots=snapshots,
        market_values=market_values,
        feature_order=feature_order,
        model=xgb_model,
        draw_boost=xgb_draw_boost,
        use_market_values=use_market_values,
        market_strength=args.market_strength,
        draw_suppression=args.draw_suppression,
        winner_mode=args.winner_mode,
        rng=rng,
    )

    round32_df = knockout_df[knockout_df["round"] == "round_of_32"].copy()
    knockout_path = output_dir / "knockout_matches.csv"
    round32_path = output_dir / "round_of_32_bracket.csv"
    third_assignment_path = output_dir / "third_place_slot_assignment.csv"
    summary_path = output_dir / "knockout_summary.json"

    knockout_df.to_csv(knockout_path, index=False)
    round32_df.to_csv(round32_path, index=False)
    third_assignment_df.to_csv(third_assignment_path, index=False)

    final_row = knockout_df[knockout_df["match_id"] == "M104"].iloc[0]
    third_place_row = knockout_df[knockout_df["match_id"] == "M103"].iloc[0]
    summary = {
        "model_mode": args.model,
        "winner_mode": args.winner_mode,
        "group_stage_dir": str(group_stage_dir),
        "output_dir": str(output_dir),
        "use_market_values": use_market_values,
        "xgboost_draw_boost": float(xgb_draw_boost),
        "tournament_weight": float(WORLD_CUP_TOURNAMENT_WEIGHT),
        "champion": str(final_row["winner"]),
        "runner_up": str(final_row["loser"]),
        "third_place": str(third_place_row["winner"]),
        "fourth_place": str(third_place_row["loser"]),
        "knockout_matches_path": str(knockout_path),
        "round_of_32_bracket_path": str(round32_path),
        "third_place_slot_assignment_path": str(third_assignment_path),
        "third_place_assignment": third_assignment_df.to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Knockout simulation mode: {args.model} ({args.winner_mode} winners)")
    print(f"Group-stage source: {group_stage_dir}")
    print()
    print("Round of 32")
    for _, row in round32_df.iterrows():
        print(format_bracket_line(row))
    print()
    print("Final path")
    for match_id in ["M97", "M98", "M99", "M100", "M101", "M102", "M103", "M104"]:
        row = knockout_df[knockout_df["match_id"] == match_id].iloc[0]
        print(format_bracket_line(row))
    print()
    print(f"Champion: {final_row['winner']}")
    print(f"Runner-up: {final_row['loser']}")
    print(f"Third place: {third_place_row['winner']}")
    print()
    print(f"All knockout matches written to: {knockout_path}")
    print(f"Round of 32 bracket written to: {round32_path}")
    print(f"Third-place slot assignment written to: {third_assignment_path}")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
