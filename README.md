# 🏆 Fair Tournament Scheduler

Generate mathematically balanced tournament schedules that ensure fair play across multiple dimensions.

## Overview

BQBrackets is an intelligent tournament scheduling tool that uses **constraint programming** to create fair tournament schedules. Instead of simple rotation systems, it optimizes 8 different fairness factors simultaneously:

1. **Pair Balance** - Prevents any two teams from playing drastically more/less than others
2. **Pair Consistency** - Smooths out pair encounter frequencies across the entire tournament
3. **Referee Balance** - Ensures teams see each referee roughly the same number of times
4. **Referee Consistency** - Smooths referee assignments across all team-ref combinations
5. **Team Match Fairness** - Ensures all teams get equal playing opportunities (intelligently handles unavoidable imbalances)
6. **Rematch Spacing** - Spreads rematches apart instead of clustering them
7. **Bye Fairness** - Distributes rest rounds equitably among teams
8. **Bye Distribution** - Prevents clustered bye rounds, spreading rest opportunities

## Features

- 🎯 **Interactive Web UI** - Streamlit-based interface for real-time schedule generation
- ⚙️ **Configurable Weights** - Adjust the importance of each fairness factor with sliders
- 🔄 **Multi-Attempt Solver** - Runs multiple optimization passes to find the best solution
- 📊 **Visual Heatmaps** - Referee balance, pair encounters, team match counts, and bye patterns
- 📥 **CSV Export** - Download schedules for use in other tools
- 🎲 **Smart Configuration** - Suggestions for optimal parameters based on your tournament size
- 💡 **Intelligent Penalty Scaling** - Recognizes mathematically unavoidable imbalances (doesn't penalize them)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/bqbrackets.git
cd bqbrackets
```

2. Create a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Run the Web App
```bash
streamlit run app.py
```
Visit `http://localhost:8501` and configure your tournament parameters using the interactive sliders.

### Run a Demo
```bash
python3 demo.py
```
Shows a quick example with 4 teams in 3 rounds.

### Run the Test Suite
```bash
python3 test_3team.py
```
Tests a more complex scenario: 14 teams, 3-team matches, 4 refs, 8 rounds.

### Score an Existing Schedule
```bash
python3 score_bracket.py --file tournament_schedule.csv
```
Analyze any CSV tournament schedule with custom weights:
```bash
python3 score_bracket.py --file schedule.csv \
  --pair-weight 20.0 \
  --team-match-weight 300.0 \
  --rematch-delay-weight 1.0
```

## How It Works

**Constraint Solver**: Uses Google OR-Tools' CP-SAT solver to find schedules that satisfy all constraints while optimizing the 8-factor score.

**Score Calculation**: Each schedule receives a composite score (lower is better). The final score is the weighted sum of all 8 fairness factors. Typical excellent scores range from 20-40, good scores from 40-60.

**Intelligent Penalty Scaling**: For scenarios where perfect balance is mathematically impossible (e.g., 14 teams with 3-per-match), the system calculates the minimum unavoidable imbalance and only penalizes differences beyond that threshold.

## Input Parameters

- **Number of Teams**: 2-100
- **Teams per Match**: 2-10
- **Number of Referees**: 1-50
- **Number of Rounds**: 1-20
- **Scoring Weights**: Configure importance of each fairness factor (0-500 scale)
- **Solver Time Limit**: 1-120 seconds per attempt
- **Max Attempts**: 1-50 scheduling attempts (higher = better quality, slower)
- **Score Threshold**: Stop when a schedule scores below this value (optimization early-stopping)

## Output

The app generates:
- Complete match schedule by round and referee
- Referee balance table (team × referee encounters)
- Pair encounter matrix (how many times each pair played)
- Team match counts (participation fairness)
- Team bye patterns (rest round distribution)
- Detailed score breakdown of all 8 factors

## Project Structure

```
├── app.py                 # Main Streamlit web interface
├── src/
│   ├── datamodel.py      # Tournament data structures
│   ├── scheduler.py      # Constraint model & solver
├── score_bracket.py      # CLI tool for scoring existing schedules
├── demo.py               # Quick demo script
├── test_3team.py         # Comprehensive test case
└── requirements.txt      # Python dependencies
```

## Example: Optimal Settings

For a typical tournament:
- **12 teams, 3-per-match, 8 rounds**: Use 4 referees (one per concurrent match)
- **Score target**: 40-50 for good balance
- **Weight focus**: Team match fairness (300) + Pair balance (20) are usually most important

## License

MIT License - see LICENSE file for details

## Contributing

Found a bug? Have a feature request? Issues and pull requests welcome!

---

Built with [Streamlit](https://streamlit.io) + [OR-Tools](https://developers.google.com/optimization)

