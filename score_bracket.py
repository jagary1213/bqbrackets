#!/usr/bin/env python3
"""Score a bracket CSV: print pair-encounter matrix, referee matrix, and scalar score.

CSV format expected (columns): Round,Match ID,Teams,Referee
`Teams` is a comma-separated list of team names in the match.

Usage:
  python3 score_bracket.py --file tournament_schedule.csv
  python3 score_bracket.py --file my.csv --pair-weight 20.0 --pair-var-weight 5.0 --ref-weight 10.0 --team-match-weight 300.0
"""
import argparse
import pandas as pd
import itertools
import statistics
import sys


def compute_pair_and_ref_counts(df):
    teams = set()
    pair_counts = {}
    ref_counts = {}  # (team, ref) -> count
    team_match_counts = {}  # team -> count of matches

    for _, row in df.iterrows():
        teams_in_match = [s.strip() for s in str(row.get("Teams", "")).split(",") if s.strip()]
        ref = row.get("Referee") if pd.notna(row.get("Referee")) else "(unassigned)"
        for t in teams_in_match:
            teams.add(t)
            ref_counts[(t, ref)] = ref_counts.get((t, ref), 0) + 1
            team_match_counts[t] = team_match_counts.get(t, 0) + 1
        for a, b in itertools.combinations(sorted(teams_in_match), 2):
            key = (a, b) if a < b else (b, a)
            pair_counts[key] = pair_counts.get(key, 0) + 1

    team_list = sorted(teams)
    return team_list, pair_counts, ref_counts, team_match_counts


def build_pair_matrix(team_list, pair_counts):
    matrix = {t: {u: 0 for u in team_list} for t in team_list}
    for (a, b), c in pair_counts.items():
        matrix[a][b] = c
        matrix[b][a] = c
    return pd.DataFrame(matrix).loc[team_list, team_list]


def build_ref_matrix(team_list, ref_counts):
    refs = sorted({r for (_, r) in ref_counts.keys()})
    mat = {t: {r: 0 for r in refs} for t in team_list}
    for (t, r), c in ref_counts.items():
        if t in mat and r in mat[t]:
            mat[t][r] = c
    return pd.DataFrame(mat).T


def score(df, pair_counts, ref_counts, team_match_counts, pair_weight, pair_var_weight, ref_slack_weight, ref_var_weight, team_match_weight, rematch_delay_weight, bye_balance_weight, bye_spread_weight, num_teams=None, num_rounds=None, num_refs=None, teams_per_match=None):
    # Include ALL possible pairs (even 0-count) for accurate slack calculation
    teams = sorted(team_match_counts.keys())
    all_pair_counts = []
    for t1, t2 in itertools.combinations(teams, 2):
        key = (t1, t2) if t1 < t2 else (t2, t1)
        all_pair_counts.append(pair_counts.get(key, 0))
    
    if all_pair_counts:
        pair_slack = max(all_pair_counts) - min(all_pair_counts)
        pair_var = statistics.pvariance(all_pair_counts)
    else:
        pair_slack = 0
        pair_var = 0.0

    # Include ALL possible (team, ref) combinations for accurate slack calculation
    teams = sorted(team_match_counts.keys())
    refs = sorted(set(r for (_, r) in ref_counts.keys()))
    all_ref_counts = []
    for team in teams:
        for ref in refs:
            all_ref_counts.append(ref_counts.get((team, ref), 0))
    
    if all_ref_counts:
        ref_slack = max(all_ref_counts) - min(all_ref_counts)
        ref_var = statistics.pvariance(all_ref_counts)
    else:
        ref_slack = 0
        ref_var = 0.0

    team_match_vals = list(team_match_counts.values()) if team_match_counts else [0]
    team_match_slack = max(team_match_vals) - min(team_match_vals) if team_match_vals else 0
    team_match_var = statistics.pvariance(team_match_vals) if len(team_match_vals) > 1 else 0.0
    
    # Intelligent penalty scaling: only penalize imbalance beyond mathematical minimum
    # If parameters provided, calculate adjusted penalty
    if num_teams and num_rounds and num_refs and teams_per_match:
        total_team_slots = num_rounds * num_refs * teams_per_match
        min_imbalance = total_team_slots % num_teams
        # If slack <= 1 and min_imbalance > 0: 1-diff is unavoidable, don't penalize
        if team_match_slack <= 1 and min_imbalance > 0:
            adjusted_team_match_slack = 0.0
        else:
            adjusted_team_match_slack = team_match_slack
    else:
        adjusted_team_match_slack = team_match_slack

    # Rematch delay: for pairs that meet multiple times, calculate average gap penalty
    pair_meetings = {}  # (t1,t2) -> [list of round numbers]
    num_rounds = df['Round'].max() if 'Round' in df.columns and not df.empty else 1
    for _, row in df.iterrows():
        round_num = row.get('Round', 1)
        teams_in_match = [s.strip() for s in str(row.get("Teams", "")).split(",") if s.strip()]
        for t1, t2 in itertools.combinations(sorted(teams_in_match), 2):
            key = (t1, t2) if t1 < t2 else (t2, t1)
            if key not in pair_meetings:
                pair_meetings[key] = []
            pair_meetings[key].append(round_num)
    
    rematch_gaps = []
    ideal_gap = num_rounds / 2.0
    for pair, rounds in pair_meetings.items():
        if len(rounds) > 1:
            sorted_rounds = sorted(rounds)
            gaps = [sorted_rounds[i+1] - sorted_rounds[i] for i in range(len(sorted_rounds)-1)]
            avg_gap = sum(gaps) / len(gaps)
            gap_penalty = max(0, (ideal_gap - avg_gap) / ideal_gap * 5.0)
            rematch_gaps.append(gap_penalty)
    rematch_delay_score = statistics.mean(rematch_gaps) if rematch_gaps else 0.0

    # Bye balance: count bye rounds per team
    teams = sorted(team_match_counts.keys())
    team_play_rounds = {t: [] for t in teams}
    for _, row in df.iterrows():
        round_num = row.get('Round', 1)
        teams_in_match = [s.strip() for s in str(row.get("Teams", "")).split(",") if s.strip()]
        for t in teams_in_match:
            if t in team_play_rounds:
                team_play_rounds[t].append(round_num)
    
    team_bye_counts = {t: num_rounds - len(set(rounds)) for t, rounds in team_play_rounds.items()}
    bye_vals = list(team_bye_counts.values())
    bye_balance_slack = max(bye_vals) - min(bye_vals) if bye_vals else 0

    # Bye spread: variance of bye round positions per team
    bye_spread_vars = []
    for team, play_rounds in team_play_rounds.items():
        all_rounds = set(range(1, num_rounds + 1))
        bye_rounds = sorted(all_rounds - set(play_rounds))
        if len(bye_rounds) > 1:
            bye_spread_vars.append(statistics.pvariance(bye_rounds))
        elif len(bye_rounds) == 1:
            bye_spread_vars.append(0.0)
    bye_spread_score = statistics.mean(bye_spread_vars) if bye_spread_vars else 0.0

    score_val = (pair_slack * pair_weight + 
                 pair_var * pair_var_weight + 
                 ref_slack * ref_slack_weight +
                 ref_var * ref_var_weight +
                 adjusted_team_match_slack * team_match_weight +
                 rematch_delay_score * rematch_delay_weight +
                 bye_balance_slack * bye_balance_weight +
                 bye_spread_score * bye_spread_weight)
    return score_val, pair_slack, pair_var, ref_slack, ref_var, team_match_slack, team_match_var, rematch_delay_score, bye_balance_slack, bye_spread_score


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--file", "-f", required=True, help="CSV file with schedule")
    p.add_argument("--pair-weight", type=float, default=20.0)
    p.add_argument("--pair-var-weight", type=float, default=5.0)
    p.add_argument("--ref-slack-weight", type=float, default=10.0)
    p.add_argument("--ref-var-weight", type=float, default=5.0)
    p.add_argument("--team-match-weight", type=float, default=300.0)
    p.add_argument("--rematch-delay-weight", type=float, default=1.0)
    p.add_argument("--bye-balance-weight", type=float, default=10.0)
    p.add_argument("--bye-spread-weight", type=float, default=3.0)
    args = p.parse_args(argv)

    try:
        df = pd.read_csv(args.file)
    except Exception as e:
        print(f"Failed to read '{args.file}': {e}")
        sys.exit(2)

    team_list, pair_counts, ref_counts, team_match_counts = compute_pair_and_ref_counts(df)

    print("Teams:", team_list)
    print("\nPair-encounters (list):")
    for k in sorted(pair_counts.keys()):
        print(f"  {k[0]} vs {k[1]}: {pair_counts[k]}")

    print("\nPair-encounter matrix:")
    print(build_pair_matrix(team_list, pair_counts).to_string())

    print("\nReferee counts per team:")
    print(build_ref_matrix(team_list, ref_counts).to_string())

    print("\nTeam match counts:")
    for team in team_list:
        print(f"  {team}: {team_match_counts.get(team, 0)} matches")

    score_val, pair_slack, pair_var, ref_slack, ref_var, team_match_slack, team_match_var, rematch_delay, bye_balance, bye_spread = score(
        df, pair_counts, ref_counts, team_match_counts, 
        args.pair_weight, args.pair_var_weight, args.ref_slack_weight, args.ref_var_weight, args.team_match_weight,
        args.rematch_delay_weight, args.bye_balance_weight, args.bye_spread_weight
    )
    print(f"\nScore: {score_val:.3f}")
    print(f"  • Pair balance (extremes): {pair_slack} × {args.pair_weight} = {pair_slack * args.pair_weight:.2f}")
    print(f"  • Pair consistency: {pair_var:.3f} × {args.pair_var_weight} = {pair_var * args.pair_var_weight:.2f}")
    print(f"  • Referee balance (extremes): {ref_slack} × {args.ref_slack_weight} = {ref_slack * args.ref_slack_weight:.2f}")
    print(f"  • Referee consistency: {ref_var:.3f} × {args.ref_var_weight} = {ref_var * args.ref_var_weight:.2f}")
    print(f"  • Team match fairness: {team_match_slack} × {args.team_match_weight} = {team_match_slack * args.team_match_weight:.2f}")
    print(f"  • Rematch spacing: {rematch_delay:.3f} × {args.rematch_delay_weight} = {rematch_delay * args.rematch_delay_weight:.2f}")
    print(f"  • Bye fairness: {bye_balance} × {args.bye_balance_weight} = {bye_balance * args.bye_balance_weight:.2f}")
    print(f"  • Bye distribution: {bye_spread:.3f} × {args.bye_spread_weight} = {bye_spread * args.bye_spread_weight:.2f}")


if __name__ == "__main__":
    main()
