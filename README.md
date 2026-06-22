# FIFA World Cup 2026 Prediction Simulator

This project builds a neutral-venue football prediction pipeline for the 2026 FIFA World Cup. It creates Elo ratings from historical international results, engineers team-form features, trains match outcome models, simulates the group stage, builds the new 48-team knockout bracket, and generates readable probability flow graphs for the tournament path.

The current recommended tournament mode is:

```text
XGBoost + no market-value adjustment
```

Market value support is included as an optional comparison mode, but the main simulation path uses the trained XGBoost model directly.

## What The Project Does

- Builds chess-style Elo ratings from `results.csv`.
- Uses tournament weights from `tournament_weights.py`.
- Uses 2000-2009 as an Elo/model warm-up period.
- Uses 2009-2026, before the World Cup start, as the model training window.
- Creates rolling features such as form over 10 matches, goal difference, goals scored/conceded trends, draw tendency, low-scoring tendency, clean sheets, and failed-to-score rates.
- Trains Logistic Regression, XGBoost, Random Forest, and ensemble comparison outputs.
- Uses XGBoost for World Cup simulation.
- Simulates the 48-team group stage with 10,000 Monte Carlo iterations.
- Selects 24 top-two teams plus the 8 best third-place teams.
- Builds a simplified Round of 32 bracket.
- Simulates knockouts from expected group-stage standings.
- Generates bracket flow graphs, winner probability ladders, and champion path visuals.

## Repository Structure

```text
.
├── results.csv
├── world_cup_groups.csv
├── market_values.csv
├── tournament_weights.py
├── dataset_builder.py
├── train.py
├── predict.py
├── simulate_group_stage.py
├── simulate_knockout.py
├── plot_knockout_graphs.py
└── build/
    ├── model_matches_features.csv
    ├── warmup_matches_features.csv
    ├── team_elo_snapshot.csv
    ├── training/
    ├── group_stage/
    ├── knockout/
    └── predictions/
```

## Installation

Use Python 3.10+.

Install the core dependencies:

```bash
pip install pandas numpy scikit-learn xgboost joblib matplotlib
```

Optional graph dependencies:

```bash
pip install networkx pydot
```

If Graphviz is installed on your system, `plot_knockout_graphs.py --layout graphviz` can use `networkx.drawing.nx_pydot.graphviz_layout`. If not, the graph script automatically falls back to a manual bracket layout.

## Full Pipeline

Run these commands from the repository root.

### 1. Build The Dataset

```bash
python dataset_builder.py
```

Main outputs:

```text
build/warmup_matches_features.csv
build/model_matches_features.csv
build/team_elo_snapshot.csv
build/dataset_metadata.json
```

### 2. Train Models

```bash
python train.py
```

Main outputs:

```text
build/training/model_comparison.csv
build/training/feature_order.json
build/training/models/xgboost/model.joblib
build/training/models/logistic_regression/model.joblib
build/training/models/random_forest/model.joblib
```

The project compares models using accuracy, balanced accuracy, macro F1, weighted F1, log loss, confusion matrices, and classification reports.

### 3. Predict A Single Match

```bash
python predict.py --team-a Spain --team-b Argentina
```

This prints base model probabilities and market-adjusted probabilities, and writes a JSON artifact to:

```text
build/predictions/
```

### 4. Simulate Group Stage

Recommended no-market XGBoost mode:

```bash
python simulate_group_stage.py --model xgboost --simulations 10000
```

Optional market-value comparison mode:

```bash
python simulate_group_stage.py --model xgboost_market --simulations 10000
```

Main no-market outputs:

```text
build/group_stage/xgboost/group_stage_matches.csv
build/group_stage/xgboost/group_stage_sample_standings.csv
build/group_stage/xgboost/group_stage_expected_standings.csv
build/group_stage/xgboost/group_winners_probability.csv
build/group_stage/xgboost/group_stage_third_place_rankings.csv
build/group_stage/xgboost/group_stage_best_third_place.csv
build/group_stage/xgboost/round_of_32_qualifiers.csv
build/group_stage/xgboost/charts/
```

The knockout stage uses expected standings because they are more stable and credible than a single sampled Monte Carlo path.

### 5. Simulate Knockout Stage

Recommended:

```bash
python simulate_knockout.py --model xgboost --winner-mode expected
```

Optional sampled path:

```bash
python simulate_knockout.py --model xgboost --winner-mode sample
```

Optional market-value comparison:

```bash
python simulate_knockout.py --model xgboost_market --winner-mode expected
```

Main no-market outputs:

```text
build/knockout/xgboost/knockout_matches.csv
build/knockout/xgboost/round_of_32_bracket.csv
build/knockout/xgboost/third_place_slot_assignment.csv
build/knockout/xgboost/knockout_summary.json
```

## Generate Graphs

Generate no-market XGBoost graphs:

```bash
python plot_knockout_graphs.py --model xgboost
```

Generate market-value comparison graphs:

```bash
python plot_knockout_graphs.py --model xgboost_market
```

Generated graph outputs:

```text
build/knockout/xgboost/graphs/knockout_probability_flow.png
build/knockout/xgboost/graphs/winner_probability_ladder.png
build/knockout/xgboost/graphs/champion_path.png
build/knockout/xgboost/graphs/knockout_graph_stats.csv
build/knockout/xgboost/graphs/knockout_graph_summary.json
```

The graph script includes:

- Winner advance probability for each knockout match.
- Scoreline and expected goals.
- Draw probability from the base model.
- Lower-Elo winner markers.
- Champion path highlighting.
- Confidence margins.
- Cumulative champion path probability.

## One-Command Recipe

For the recommended no-market XGBoost tournament:

```bash
python dataset_builder.py
python train.py
python simulate_group_stage.py --model xgboost --simulations 10000
python simulate_knockout.py --model xgboost --winner-mode expected
python plot_knockout_graphs.py --model xgboost
```

For the XGBoost + market-value comparison:

```bash
python simulate_group_stage.py --model xgboost_market --simulations 10000
python simulate_knockout.py --model xgboost_market --winner-mode expected
python plot_knockout_graphs.py --model xgboost_market
```

## Model Notes

### Elo

Elo starts from a base rating and updates match-by-match using:

- Expected result from Elo difference.
- Actual result from win/draw/loss.
- Tournament weight.
- Goal margin adjustment.

Tournament weights are defined in `tournament_weights.py`, with higher weights for competitions like the FIFA World Cup, continental championships, and World Cup qualification.

### Features

The training dataset includes neutral-venue features only. Home and away advantages are intentionally excluded because the World Cup finals are treated as neutral-site matches.

Useful engineered features include:

- Elo difference.
- Team form over the last 10 matches.
- Recent goal difference.
- Goals scored and conceded trends.
- Clean sheet rate.
- Failed-to-score rate.
- Draw rate.
- Low-scoring match rate.
- Tight-match rate.
- Weighted recent form.
- Opponent strength in recent matches.
- Head-to-head placeholders.
- Market value fields for optional adjustment.

### Draw Handling

Football draws are common in normal matches, but knockout games require a winner. In knockout simulation:

- The model still predicts win/draw/loss probabilities.
- Draw probability is redistributed into advancement probability using penalty/extra-time edge.
- The CSV preserves both base draw probability and final advance probability.

## Important Outputs

### Training

```text
build/training/model_comparison.csv
build/training/models/xgboost/metrics.json
build/training/models/xgboost/confusion_matrix_test.png
build/training/models/xgboost/feature_importance.csv
```

### Group Stage

```text
build/group_stage/xgboost/group_stage_expected_standings.csv
build/group_stage/xgboost/group_stage_third_place_rankings.csv
build/group_stage/xgboost/round_of_32_qualifiers.csv
```

### Knockouts

```text
build/knockout/xgboost/knockout_matches.csv
build/knockout/xgboost/knockout_summary.json
```

### Graphs

```text
build/knockout/xgboost/graphs/
```

## Data Files

Required:

- `results.csv`: historical international match results.
- `world_cup_groups.csv`: World Cup group definitions with columns `group,team`.

Optional:

- `market_values.csv`: total team market value, used only in `xgboost_market` mode.

## Caveats

This is a statistical simulation, not a guarantee. International football has high randomness, low scoring, tactical matchup effects, injuries, squad selection uncertainty, penalties, and tournament pressure that cannot be fully captured from historical results alone.

The model is best used to compare relative strengths, bracket paths, and probability-weighted outcomes rather than as an exact forecast.

## License

Add a license file before publishing if you want others to reuse or modify the project under clear terms.
