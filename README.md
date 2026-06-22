# FIFA World Cup 2026 Prediction Simulator

This project builds a neutral-venue football prediction pipeline for the 2026 FIFA World Cup. It creates Elo ratings from historical international results, engineers team-form features, trains match outcome models, simulates the group stage, builds the new 48-team knockout bracket, and generates readable probability flow graphs for the tournament path.


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
FIFA-WC26-Prediction-Sim
├── data/                      # Input datasets
├── results/
│   ├── group stage/           # Group stage outputs
│   └── knockouts and final/   # Knockout stage outputs and visualizations
├── src/
│   ├── simulation/            # Tournament simulation engine
│   └── train/                 # Model training and dataset generation
├── README.md
├── requirements.txt
└── .gitignore
```

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

## Results & Visualizations

### Champion Probabilities
<img width="3182" height="3318" alt="image" src="https://github.com/user-attachments/assets/89c287a2-314b-402f-9815-7c91d89e0931" />

### Group Stage Final Standings
## Projected Group Stage Standings

## Projected Group Stage Standings

🟢 = Qualified (Top 2)  
🟡 = Third Place (Best Third-Place qualification possible)  
🔴 = Eliminated

| Group | 1st | 2nd | 3rd | 4th |
|---------|---------|---------|---------|---------|
| A | 🟢 Mexico | 🟢 South Korea | 🟡 Czech Republic | 🔴 South Africa |
| B | 🟢 Switzerland | 🟢 Canada | 🟡 Bosnia and Herzegovina | 🔴 Qatar |
| C | 🟢 Brazil | 🟢 Morocco | 🟡 Scotland | 🔴 Haiti |
| D | 🟢 Paraguay | 🟢 Turkey | 🟡 United States | 🔴 Australia |
| E | 🟢 Ivory Coast | 🟢 Germany | 🟡 Ecuador | 🔴 Curaçao |
| F | 🟢 Netherlands | 🟢 Japan | 🟡 Sweden | 🔴 Tunisia |
| G | 🟢 Belgium | 🟢 Iran | 🟡 New Zealand | 🔴 Egypt |
| H | 🟢 Uruguay | 🟢 Spain | 🟡 Saudi Arabia | 🔴 Cape Verde |
| I | 🟢 France | 🟢 Norway | 🟡 Senegal | 🔴 Iraq |
| J | 🟢 Austria | 🟢 Argentina | 🟡 Algeria | 🔴 Jordan |
| K | 🟢 Portugal | 🟢 Colombia | 🟡 DR Congo | 🔴 Uzbekistan |
| L | 🟢 England | 🟢 Croatia | 🟡 Panama | 🔴 Ghana |

*Projected standings from a representative tournament simulation generated using the trained XGBoost model.*

### Knockout Bracket
<img width="5059" height="3600" alt="knockout_probability_flow" src="https://github.com/user-attachments/assets/d6c738a3-c1fa-4411-a5da-d200d8fc32b8" />

### Winner Path
<img width="3113" height="1344" alt="champion_path" src="https://github.com/user-attachments/assets/774ad6bd-fde1-4970-8ced-806c10db0292" />

## Caveats

This is a statistical simulation, not a guarantee. International football has high randomness, low scoring, tactical matchup effects, injuries, squad selection uncertainty, penalties, and tournament pressure that cannot be fully captured from historical results alone.

The model is best used to compare relative strengths, bracket paths, and probability-weighted outcomes rather than as an exact forecast.
