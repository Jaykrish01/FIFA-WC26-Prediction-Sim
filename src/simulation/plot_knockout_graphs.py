import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

try:
    import networkx as nx
    from networkx.drawing.nx_pydot import graphviz_layout

    HAS_NETWORKX = True
except ImportError:
    nx = None
    graphviz_layout = None
    HAS_NETWORKX = False


PARENTS: dict[str, tuple[str, str]] = {
    "M89": ("M74", "M77"),
    "M90": ("M73", "M75"),
    "M91": ("M76", "M78"),
    "M92": ("M79", "M80"),
    "M93": ("M83", "M84"),
    "M94": ("M81", "M82"),
    "M95": ("M86", "M88"),
    "M96": ("M85", "M87"),
    "M97": ("M89", "M90"),
    "M98": ("M93", "M94"),
    "M99": ("M91", "M92"),
    "M100": ("M95", "M96"),
    "M101": ("M97", "M98"),
    "M102": ("M99", "M100"),
    "M103": ("M101", "M102"),
    "M104": ("M101", "M102"),
}

FINAL_TREE_ROOT = "M104"
THIRD_PLACE_ROOT = "M103"
ROUND_ORDER = {
    "round_of_32": 0,
    "round_of_16": 1,
    "quarter_final": 2,
    "semi_final": 3,
    "third_place_playoff": 4,
    "final": 4,
}
ROUND_LABELS = {
    "round_of_32": "Round of 32",
    "round_of_16": "Round of 16",
    "quarter_final": "Quarter Final",
    "semi_final": "Semi Final",
    "third_place_playoff": "Third Place",
    "final": "Final",
}
ROUND_COLORS = {
    "round_of_32": "#d8f3dc",
    "round_of_16": "#b7e4c7",
    "quarter_final": "#95d5b2",
    "semi_final": "#74c69d",
    "third_place_playoff": "#ffd6a5",
    "final": "#52b788",
}
FIG_BG = "#f7f3ea"
INK = "#20302a"
MUTED = "#58645d"
EDGE = "#9be7a3"
CHAMPION = "#1b7f46"
UPSET = "#d95d39"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate readable knockout flow graphs from knockout_matches.csv."
    )
    parser.add_argument(
        "--model",
        choices=["xgboost", "xgboost_market"],
        default="xgboost",
        help="Which knockout output folder to plot.",
    )
    parser.add_argument(
        "--knockout-dir",
        default=None,
        help="Directory containing knockout_matches.csv. Defaults to build/knockout/<model>.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for graph outputs. Defaults to <knockout-dir>/graphs.",
    )
    parser.add_argument(
        "--layout",
        choices=["auto", "manual", "graphviz"],
        default="auto",
        help="auto uses graphviz_layout when networkx/pydot/Graphviz are installed, otherwise manual.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="PNG export resolution.",
    )
    return parser.parse_args()


def read_knockout_matches(knockout_path: Path) -> pd.DataFrame:
    if not knockout_path.exists():
        raise FileNotFoundError(
            f"Missing {knockout_path}. Run simulate_knockout.py before plotting graphs."
        )
    df = pd.read_csv(knockout_path, encoding="utf-8-sig")
    required = {
        "round",
        "match_id",
        "team_a",
        "team_b",
        "winner",
        "loser",
        "team_a_advance_prob",
        "team_b_advance_prob",
        "scoreline",
        "expected_goals_a",
        "expected_goals_b",
        "elo_diff",
        "draw_prob",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{knockout_path} is missing columns: {sorted(missing)}")

    df["match_number"] = df["match_id"].str.extract(r"(\d+)").astype(int)
    df["round_order"] = df["round"].map(ROUND_ORDER)
    return df.sort_values(["round_order", "match_number"]).reset_index(drop=True)


def winner_probability(row: pd.Series) -> float:
    if str(row["winner"]) == str(row["team_a"]):
        return float(row["team_a_advance_prob"])
    return float(row["team_b_advance_prob"])


def loser_probability(row: pd.Series) -> float:
    if str(row["winner"]) == str(row["team_a"]):
        return float(row["team_b_advance_prob"])
    return float(row["team_a_advance_prob"])


def winner_xg_diff(row: pd.Series) -> float:
    diff = float(row["expected_goals_a"]) - float(row["expected_goals_b"])
    return diff if str(row["winner"]) == str(row["team_a"]) else -diff


def winner_elo_diff(row: pd.Series) -> float:
    diff = float(row["elo_diff"])
    return diff if str(row["winner"]) == str(row["team_a"]) else -diff


def enrich_stats(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["winner_advance_prob"] = enriched.apply(winner_probability, axis=1)
    enriched["loser_advance_prob"] = enriched.apply(loser_probability, axis=1)
    enriched["confidence_margin"] = (
        enriched["winner_advance_prob"] - enriched["loser_advance_prob"]
    )
    enriched["winner_xg_diff"] = enriched.apply(winner_xg_diff, axis=1)
    enriched["winner_elo_diff"] = enriched.apply(winner_elo_diff, axis=1)
    enriched["lower_elo_winner"] = enriched["winner_elo_diff"] < 0
    enriched["winner_regulation_prob"] = np.where(
        enriched["winner"] == enriched["team_a"],
        enriched["team_a_win_prob"],
        enriched["team_b_win_prob"],
    )
    return enriched


def confidence_color(probability: float) -> str:
    if probability >= 0.68:
        return "#1b7f46"
    if probability >= 0.61:
        return "#2f9e44"
    if probability >= 0.56:
        return "#74c69d"
    if probability >= 0.52:
        return "#b7e4c7"
    return "#ffe8a3"


def wrap_line(value: Any, width: int = 18) -> str:
    text = str(value)
    if len(text) <= width:
        return text
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def match_label(row: pd.Series) -> str:
    winner = wrap_line(row["winner"], 18)
    loser = wrap_line(row["loser"], 18)
    winner_prob = float(row["winner_advance_prob"])
    loser_prob = float(row["loser_advance_prob"])
    xg_a = float(row["expected_goals_a"])
    xg_b = float(row["expected_goals_b"])
    draw_prob = float(row["draw_prob"])
    upset = " | lower Elo" if bool(row["lower_elo_winner"]) else ""
    return (
        f"{row['match_id']}  {row['scoreline']}\n"
        f"{winner} {winner_prob:.0%}\n"
        f"over {loser} {loser_prob:.0%}\n"
        f"xG {xg_a:.1f}-{xg_b:.1f} | draw {draw_prob:.0%}{upset}"
    )


def build_edges(include_third_place: bool = True) -> list[tuple[str, str, str]]:
    edges: list[tuple[str, str, str]] = []
    for parent, children in PARENTS.items():
        if parent == THIRD_PLACE_ROOT and not include_third_place:
            continue
        edge_type = "loser_path" if parent == THIRD_PLACE_ROOT else "winner_path"
        edges.append((children[0], parent, edge_type))
        edges.append((children[1], parent, edge_type))
    edges.append((FINAL_TREE_ROOT, "Champion", "champion"))
    return edges


def leaf_order(root: str = FINAL_TREE_ROOT) -> list[str]:
    if root not in PARENTS:
        return [root]
    left, right = PARENTS[root]
    return leaf_order(left) + leaf_order(right)


def manual_positions(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    by_id = {str(row["match_id"]): row for _, row in df.iterrows()}
    leaves = leaf_order()
    positions: dict[str, tuple[float, float]] = {}
    vertical_gap = 1.35
    for index, match_id in enumerate(leaves):
        positions[match_id] = (0.0, (len(leaves) - 1 - index) * vertical_gap)

    def place_parent(match_id: str) -> tuple[float, float]:
        if match_id in positions:
            return positions[match_id]
        left, right = PARENTS[match_id]
        left_pos = place_parent(left)
        right_pos = place_parent(right)
        round_name = str(by_id[match_id]["round"])
        x = ROUND_ORDER[round_name] * 3.15
        y = (left_pos[1] + right_pos[1]) / 2.0
        positions[match_id] = (x, y)
        return positions[match_id]

    place_parent(FINAL_TREE_ROOT)
    final_x, final_y = positions[FINAL_TREE_ROOT]
    positions["Champion"] = (final_x + 2.8, final_y)
    if THIRD_PLACE_ROOT in by_id:
        positions[THIRD_PLACE_ROOT] = (final_x, min(y for _, y in positions.values()) - 2.1)
    return positions


def graphviz_positions(df: pd.DataFrame) -> dict[str, tuple[float, float]] | None:
    if not HAS_NETWORKX:
        return None
    graph = nx.DiGraph()
    graph.graph["graph"] = {"rankdir": "LR", "splines": "ortho"}
    for match_id in df["match_id"].tolist():
        graph.add_node(match_id)
    graph.add_node("Champion")
    for source, target, _ in build_edges(include_third_place=False):
        graph.add_edge(source, target)
    try:
        raw = graphviz_layout(graph, prog="dot")
    except Exception:
        return None
    xs = np.array([value[0] for value in raw.values()], dtype=float)
    ys = np.array([value[1] for value in raw.values()], dtype=float)
    x_span = max(xs.max() - xs.min(), 1.0)
    y_span = max(ys.max() - ys.min(), 1.0)
    return {
        node: ((coords[0] - xs.min()) / x_span * 12.5, (coords[1] - ys.min()) / y_span * 20.0)
        for node, coords in raw.items()
    }


def choose_positions(df: pd.DataFrame, layout: str) -> tuple[dict[str, tuple[float, float]], str]:
    if layout in {"auto", "graphviz"}:
        positions = graphviz_positions(df)
        if positions is not None:
            return positions, "graphviz_layout"
        if layout == "graphviz":
            print("graphviz_layout unavailable; falling back to manual bracket layout.")
    return manual_positions(df), "manual_bracket"


def draw_round_labels(ax: plt.Axes, positions: dict[str, tuple[float, float]], df: pd.DataFrame) -> None:
    ymax = max(y for _, y in positions.values()) + 0.95
    for round_name, order in ROUND_ORDER.items():
        if round_name == "third_place_playoff":
            continue
        round_rows = df[df["round"] == round_name]
        if round_rows.empty:
            continue
        x_values = [positions[str(match_id)][0] for match_id in round_rows["match_id"] if str(match_id) in positions]
        if not x_values:
            continue
        ax.text(
            float(np.mean(x_values)),
            ymax,
            ROUND_LABELS[round_name],
            ha="center",
            va="bottom",
            fontsize=16,
            color=INK,
            fontweight="bold",
        )


def plot_flow_bracket(
    df: pd.DataFrame,
    output_path: Path,
    layout: str,
    dpi: int,
) -> Path:
    positions, layout_used = choose_positions(df, layout)
    by_id = {str(row["match_id"]): row for _, row in df.iterrows()}
    champion = str(df[df["match_id"] == FINAL_TREE_ROOT].iloc[0]["winner"])
    champion_path = set(df[df["winner"] == champion]["match_id"].astype(str).tolist())
    champion_path.add("Champion")

    fig, ax = plt.subplots(figsize=(24, 18), facecolor=FIG_BG)
    fig.subplots_adjust(top=0.88, bottom=0.07, left=0.03, right=0.98)
    ax.set_facecolor(FIG_BG)

    for source, target, edge_type in build_edges(include_third_place=True):
        if source not in positions or target not in positions:
            continue
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        source_row = by_id.get(source)
        is_champion_path = source in champion_path and target in champion_path
        if edge_type == "loser_path":
            color = "#c58c45"
            style = "--"
            alpha = 0.55
            width = 2.1
        elif is_champion_path:
            color = CHAMPION
            style = "-"
            alpha = 0.95
            width = 4.2
        else:
            color = EDGE
            style = "-"
            alpha = 0.52
            width = 1.5 + 3.0 * max(float(source_row["winner_advance_prob"]) - 0.5, 0.0) if source_row is not None else 2.0
        ax.plot([x1, x2], [y1, y2], color=color, lw=width, alpha=alpha, linestyle=style, zorder=1)

    for _, row in df.iterrows():
        match_id = str(row["match_id"])
        if match_id not in positions:
            continue
        x, y = positions[match_id]
        winner_prob = float(row["winner_advance_prob"])
        face = confidence_color(winner_prob)
        edge = UPSET if bool(row["lower_elo_winner"]) else "#2d6a4f"
        line_width = 3.4 if match_id in champion_path else 1.8
        size = 1450 + 4200 * max(winner_prob - 0.5, 0.0)
        ax.scatter(
            [x],
            [y],
            s=size,
            c=[face],
            edgecolors=edge,
            linewidths=line_width,
            zorder=3,
        )
        ax.text(
            x,
            y - 0.75,
            match_label(row),
            ha="center",
            va="top",
            fontsize=8.6,
            color=INK,
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "#fffdf7",
                "edgecolor": "#8f978f",
                "linewidth": 0.8,
                "alpha": 0.96,
            },
            zorder=5,
        )

    final_x, final_y = positions["Champion"]
    final_row = df[df["match_id"] == FINAL_TREE_ROOT].iloc[0]
    champion_prob = float(final_row["winner_advance_prob"])
    ax.scatter(
        [final_x],
        [final_y],
        s=4300,
        c=[CHAMPION],
        edgecolors="#0b3d25",
        linewidths=3.5,
        zorder=4,
    )
    ax.text(
        final_x,
        final_y,
        f"Champion\n{wrap_line(champion, 16)}\nFinal {champion_prob:.0%}",
        ha="center",
        va="center",
        color="white",
        fontsize=12,
        fontweight="bold",
        zorder=6,
    )

    draw_round_labels(ax, positions, df)
    fig.text(
        0.035,
        0.968,
        "Knockout probability flow\n"
        "Node size/color = winner advance probability | red outline = lower-Elo winner | thick green path = champion path",
        ha="left",
        va="top",
        fontsize=17,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.01,
        0.035,
        f"Layout: {layout_used}. Probabilities are knockout advance probabilities, so draw probability is redistributed through extra-time/penalty edge.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color=MUTED,
    )

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ffe8a3", markeredgecolor="#2d6a4f", markersize=12, label="Narrow winner"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#74c69d", markeredgecolor="#2d6a4f", markersize=12, label="Solid winner"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CHAMPION, markeredgecolor="#0b3d25", markersize=12, label="High confidence / champion"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#b7e4c7", markeredgecolor=UPSET, markersize=12, label="Lower-Elo winner"),
        Line2D([0], [0], color=CHAMPION, lw=4, label="Champion path"),
        Line2D([0], [0], color="#c58c45", lw=2, linestyle="--", label="Third-place path"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, framealpha=0.92, fontsize=10)

    ax.axis("off")
    x_values = [x for x, _ in positions.values()]
    y_values = [y for _, y in positions.values()]
    ax.set_xlim(min(x_values) - 1.0, max(x_values) + 1.6)
    ax.set_ylim(min(y_values) - 2.3, max(y_values) + 2.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_winner_probability_bars(df: pd.DataFrame, output_path: Path, dpi: int) -> Path:
    ordered = df.sort_values(["round_order", "match_number"], ascending=[False, True]).reset_index(drop=True)
    labels = [
        f"{row.match_id}: {row.winner} over {row.loser} ({row.scoreline})"
        for row in ordered.itertuples(index=False)
    ]
    probs = ordered["winner_advance_prob"].to_numpy(dtype=float)
    margins = ordered["confidence_margin"].to_numpy(dtype=float)
    colors = [confidence_color(prob) for prob in probs]

    fig_height = max(9, len(ordered) * 0.34)
    fig, ax = plt.subplots(figsize=(16, fig_height), facecolor=FIG_BG)
    ax.set_facecolor(FIG_BG)
    y_pos = np.arange(len(ordered))
    bars = ax.barh(y_pos, probs, color=colors, edgecolor="#2d6a4f", linewidth=0.9)
    ax.axvline(0.50, color="#636363", lw=1.0, linestyle="--", alpha=0.75)
    ax.axvline(float(probs.mean()), color=CHAMPION, lw=1.4, linestyle="-", alpha=0.9)

    for index, (bar, row, margin) in enumerate(zip(bars, ordered.itertuples(index=False), margins, strict=False)):
        upset_marker = " lower Elo" if bool(row.lower_elo_winner) else ""
        ax.text(
            min(float(row.winner_advance_prob) + 0.006, 0.91),
            bar.get_y() + bar.get_height() / 2,
            f"{row.winner_advance_prob:.1%} | margin {margin:.1%} | xG edge {row.winner_xg_diff:+.2f}{upset_marker}",
            va="center",
            fontsize=8.5,
            color=INK,
        )
        if bool(row.lower_elo_winner):
            ax.scatter([0.497], [index], color=UPSET, marker="*", s=80, zorder=4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlim(0.48, max(0.78, probs.max() + 0.08))
    ax.set_xlabel("Winner advance probability", fontsize=12, color=INK)
    ax.set_title(
        "Knockout winner probability ladder\n"
        "Stars mark lower-Elo winners; green vertical line is average winner confidence.",
        fontsize=17,
        fontweight="bold",
        color=INK,
        loc="left",
    )
    ax.grid(axis="x", color="#d7d0c2", linestyle="-", linewidth=0.8, alpha=0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", colors=MUTED)
    ax.tick_params(axis="y", colors=INK)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_champion_path(df: pd.DataFrame, output_path: Path, dpi: int) -> Path:
    final_row = df[df["match_id"] == FINAL_TREE_ROOT].iloc[0]
    champion = str(final_row["winner"])
    path_df = df[df["winner"] == champion].sort_values(["round_order", "match_number"]).copy()
    path_df = path_df[path_df["round"] != "third_place_playoff"].reset_index(drop=True)
    path_df["cumulative_path_probability"] = path_df["winner_advance_prob"].cumprod()

    fig, ax = plt.subplots(figsize=(18, 7), facecolor=FIG_BG)
    ax.set_facecolor(FIG_BG)
    x_positions = np.arange(len(path_df)) * 2.25
    y = np.zeros(len(path_df))
    ax.plot(x_positions, y, color=CHAMPION, lw=5, alpha=0.68, zorder=1)

    for index, row in path_df.iterrows():
        prob = float(row["winner_advance_prob"])
        cumulative = float(row["cumulative_path_probability"])
        color = confidence_color(prob)
        ax.scatter(
            [x_positions[index]],
            [0],
            s=2600 + 4200 * max(prob - 0.5, 0.0),
            c=[color],
            edgecolors=CHAMPION,
            linewidths=2.4,
            zorder=3,
        )
        label = (
            f"{row['match_id']} | {ROUND_LABELS[row['round']]}\n"
            f"{champion} {prob:.1%}\n"
            f"vs {wrap_line(row['loser'], 18)}\n"
            f"{row['scoreline']} | xG edge {row['winner_xg_diff']:+.2f}\n"
            f"path {cumulative:.2%}"
        )
        ax.text(
            x_positions[index],
            -0.72,
            label,
            ha="center",
            va="top",
            fontsize=9.5,
            color=INK,
            bbox={
                "boxstyle": "round,pad=0.33",
                "facecolor": "#fffdf7",
                "edgecolor": "#8f978f",
                "linewidth": 0.8,
                "alpha": 0.96,
            },
        )

    trophy = FancyBboxPatch(
        (x_positions[-1] + 1.15, -0.34),
        2.25,
        0.68,
        boxstyle="round,pad=0.22,rounding_size=0.18",
        facecolor=CHAMPION,
        edgecolor="#0b3d25",
        linewidth=2.0,
        zorder=4,
    )
    ax.add_patch(trophy)
    ax.text(
        x_positions[-1] + 2.275,
        0,
        f"Champion\n{champion}",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="white",
        zorder=5,
    )
    ax.set_title(
        f"{champion}'s title path\nMatch probability and cumulative path probability across the knockout bracket.",
        fontsize=17,
        fontweight="bold",
        color=INK,
        loc="left",
    )
    ax.axis("off")
    ax.set_xlim(-0.9, x_positions[-1] + 3.75)
    ax.set_ylim(-2.55, 1.15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_stats(df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    stats_cols = [
        "round",
        "match_id",
        "team_a",
        "team_b",
        "winner",
        "loser",
        "scoreline",
        "winner_advance_prob",
        "loser_advance_prob",
        "confidence_margin",
        "winner_regulation_prob",
        "draw_prob",
        "winner_xg_diff",
        "winner_elo_diff",
        "lower_elo_winner",
        "expected_goals_a",
        "expected_goals_b",
    ]
    stats_path = output_dir / "knockout_graph_stats.csv"
    df[stats_cols].to_csv(stats_path, index=False)

    closest = df.sort_values("confidence_margin").iloc[0]
    strongest = df.sort_values("winner_advance_prob", ascending=False).iloc[0]
    upsets = df[df["lower_elo_winner"]].sort_values("winner_elo_diff")
    final = df[df["match_id"] == FINAL_TREE_ROOT].iloc[0]
    summary = {
        "champion": str(final["winner"]),
        "runner_up": str(final["loser"]),
        "average_winner_advance_probability": float(df["winner_advance_prob"].mean()),
        "average_draw_probability": float(df["draw_prob"].mean()),
        "closest_match": {
            "match_id": str(closest["match_id"]),
            "winner": str(closest["winner"]),
            "loser": str(closest["loser"]),
            "winner_advance_probability": float(closest["winner_advance_prob"]),
            "margin": float(closest["confidence_margin"]),
        },
        "strongest_favorite_to_advance": {
            "match_id": str(strongest["match_id"]),
            "winner": str(strongest["winner"]),
            "loser": str(strongest["loser"]),
            "winner_advance_probability": float(strongest["winner_advance_prob"]),
        },
        "lower_elo_winners": int(len(upsets)),
        "biggest_lower_elo_winner": None
        if upsets.empty
        else {
            "match_id": str(upsets.iloc[0]["match_id"]),
            "winner": str(upsets.iloc[0]["winner"]),
            "loser": str(upsets.iloc[0]["loser"]),
            "elo_gap": float(upsets.iloc[0]["winner_elo_diff"]),
            "winner_advance_probability": float(upsets.iloc[0]["winner_advance_prob"]),
        },
    }
    summary_path = output_dir / "knockout_graph_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return stats_path, summary_path


def main() -> None:
    args = parse_args()
    knockout_dir = Path(args.knockout_dir) if args.knockout_dir else Path("build/knockout") / args.model
    output_dir = Path(args.output_dir) if args.output_dir else knockout_dir / "graphs"
    output_dir.mkdir(parents=True, exist_ok=True)

    knockout_path = knockout_dir / "knockout_matches.csv"
    df = enrich_stats(read_knockout_matches(knockout_path))

    flow_path = plot_flow_bracket(
        df=df,
        output_path=output_dir / "knockout_probability_flow.png",
        layout=args.layout,
        dpi=args.dpi,
    )
    bars_path = plot_winner_probability_bars(
        df=df,
        output_path=output_dir / "winner_probability_ladder.png",
        dpi=args.dpi,
    )
    champion_path = plot_champion_path(
        df=df,
        output_path=output_dir / "champion_path.png",
        dpi=args.dpi,
    )
    stats_path, summary_path = write_stats(df, output_dir)

    final = df[df["match_id"] == FINAL_TREE_ROOT].iloc[0]
    print(f"Graph source: {knockout_path}")
    print(f"Champion path: {final['winner']} over {final['loser']} in the final")
    print(f"NetworkX available: {HAS_NETWORKX}")
    print(f"Flow graph written to: {flow_path}")
    print(f"Winner probability ladder written to: {bars_path}")
    print(f"Champion path graph written to: {champion_path}")
    print(f"Graph stats written to: {stats_path}")
    print(f"Graph summary written to: {summary_path}")


if __name__ == "__main__":
    main()
