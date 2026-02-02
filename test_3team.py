from src.datamodel import DataModel
from src.scheduler import build_round_robin_model, solve_and_extract
import pandas as pd
import statistics
import itertools

dm = DataModel()
# 14 teams
for name in ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "M", "N", "O", "P"]:
    dm.add_team(name)
# 4 referees
for r in ["Douglas", "Scott", "Baggett", "Jono"]: dm.add_referee(r)


print(f"Round-robin schedule: 14 teams, 3-team matches, 4 matches/round, 8 rounds")
dm.set_rounds(8)
dm.generate_round_robin_matches(teams_per_match=3, matches_per_round=4)
print(f"Generated {len(dm.matches)} matches using round-robin rotation")

model, solver, vars = build_round_robin_model(dm, dm.rounds, matches_per_round=4)
schedule, counts = solve_and_extract(model, solver, vars, dm, dm.rounds, time_limit_seconds=30)

if schedule is None:
    print("No feasible solution found.")
    diag = counts.get("_diag") if isinstance(counts, dict) else None
    if diag:
        print("\nDiagnostics:")
        for k, v in diag.items():
            if k == "reasons":
                print("  Reasons:")
                for r in v:
                    print("   -", r)
            else:
                print(f"  {k}: {v}")
else:
    total_scheduled = len([m for r in schedule.values() for m in r])
    print(f'\nSchedule ({total_scheduled} total matches scheduled):')
    for r in sorted(schedule.keys()):
        matches_in_round = schedule[r]
        if matches_in_round:
            print(f"Round {r}:")
            for entry in matches_in_round:
                mid, team_names, ref = entry
                print(f"  {ref:12s}: {team_names}")

    print(f'\nReferee utilization (matches per ref, per round):')
    for r in sorted(schedule.keys()):
        ref_counts = {}
        for entry in schedule[r]:
            mid, team_names, ref = entry
            ref_counts[ref] = ref_counts.get(ref, 0) + 1
        if ref_counts:
            print(f"  Round {r}: {ref_counts}")

    print(f'\nTeam play frequency (how many matches per team):')
    team_plays = {t.name: 0 for t in dm.teams}
    for r in schedule.values():
        for mid, team_names, ref in r:
            for team_name in team_names.split(", "):
                team_plays[team_name] += 1
    for team, count in sorted(team_plays.items()):
        print(f"  {team}: {count} matches")

    print(f'\nPair encounters (how many times each pair faced each other):')
    pair_encounters = {}
    pair_meetings = {}  # Track which rounds pairs meet for rematch delay calculation
    for r in schedule.values():
        for mid, team_names, ref in r:
            teams = team_names.split(", ")
            # Record all pairs in this match
            for i in range(len(teams)):
                for j in range(i+1, len(teams)):
                    pair = tuple(sorted([teams[i], teams[j]]))
                    pair_encounters[pair] = pair_encounters.get(pair, 0) + 1
                    if pair not in pair_meetings:
                        pair_meetings[pair] = []
                    # Extract round number from schedule dict key
                    for round_num in schedule.keys():
                        if (mid, team_names, ref) in schedule[round_num]:
                            pair_meetings[pair].append(round_num)
                            break
    
    # Sort and display
    for pair, count in sorted(pair_encounters.items()):
        print(f"  {pair[0]} vs {pair[1]}: {count}")
    
    # Also print a readable pair-encounter matrix (teams x teams)
    team_list = [t.name for t in dm.teams]
    matrix = {t: {u: 0 for u in team_list} for t in team_list}
    for (a, b), c in pair_encounters.items():
        matrix[a][b] = c
        matrix[b][a] = c

    df_pairs = pd.DataFrame(matrix).loc[team_list, team_list]
    print('\nPair-encounter matrix:')
    print(df_pairs.to_string())

    # Summary stats
    min_enc = min(pair_encounters.values()) if pair_encounters else 0
    max_enc = max(pair_encounters.values()) if pair_encounters else 0
    avg_enc = sum(pair_encounters.values()) / len(pair_encounters) if pair_encounters else 0
    print(f'\nPair balance stats: min={min_enc}, max={max_enc}, avg={avg_enc:.1f}, variance={max_enc - min_enc}')
    
    # Calculate and display score breakdown
    print("\n" + "="*60)
    print("SCORE BREAKDOWN")
    print("="*60)
    
    # 1. Pair balance (extremes) and consistency
    # Include ALL possible pairs (even 0-count) for accurate slack calculation
    team_list = [t.name for t in dm.teams]
    all_pair_counts = []
    for t1, t2 in itertools.combinations(sorted(team_list), 2):
        key = tuple(sorted([t1, t2]))
        all_pair_counts.append(pair_encounters.get(key, 0))
    
    if all_pair_counts:
        pair_slack = max(all_pair_counts) - min(all_pair_counts)
        pair_mean = sum(all_pair_counts) / len(all_pair_counts)
        pair_var = sum((x - pair_mean) ** 2 for x in all_pair_counts) / len(all_pair_counts)
    else:
        pair_slack, pair_var = 0, 0.0
    
    # 2. Referee balance (extremes) and consistency
    # Reconstruct referee counts from schedule
    ref_counts = {}
    for r in schedule.values():
        for mid, team_names, ref in r:
            for team_name in team_names.split(", "):
                key = (team_name, ref)
                ref_counts[key] = ref_counts.get(key, 0) + 1
    
    # Include ALL possible (team, ref) combinations
    team_list = [t.name for t in dm.teams]
    refs_list = sorted(set(ref for (_, ref) in ref_counts.keys()))
    all_ref_counts = []
    for team in team_list:
        for ref in refs_list:
            all_ref_counts.append(ref_counts.get((team, ref), 0))
    
    if all_ref_counts:
        ref_slack = max(all_ref_counts) - min(all_ref_counts)
        ref_mean = sum(all_ref_counts) / len(all_ref_counts)
        ref_var = sum((x - ref_mean) ** 2 for x in all_ref_counts) / len(all_ref_counts)
    else:
        ref_slack, ref_var = 0, 0.0
    
    # 3. Team match fairness
    match_vals = list(team_plays.values())
    if match_vals:
        team_match_slack = max(match_vals) - min(match_vals)
        match_mean = sum(match_vals) / len(match_vals)
        team_match_var = sum((x - match_mean) ** 2 for x in match_vals) / len(match_vals)
    else:
        team_match_slack, team_match_var = 0, 0.0
    
    # 4. Rematch delay: for pairs that meet multiple times
    num_rounds = max(schedule.keys()) if schedule else 1
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
    
    # 5. Bye balance and spread
    team_list = [t.name for t in dm.teams]
    team_play_rounds = {t: [] for t in team_list}
    for round_num in sorted(schedule.keys()):
        for mid, team_names, ref in schedule[round_num]:
            for team in team_names.split(", "):
                if team in team_play_rounds:
                    team_play_rounds[team].append(round_num)
    
    team_bye_counts = {t: num_rounds - len(set(rounds)) for t, rounds in team_play_rounds.items()}
    bye_vals = list(team_bye_counts.values())
    bye_balance_slack = max(bye_vals) - min(bye_vals) if bye_vals else 0
    
    bye_spread_vars = []
    for team, play_rounds in team_play_rounds.items():
        all_rounds = set(range(1, num_rounds + 1))
        bye_rounds = sorted(all_rounds - set(play_rounds))
        if len(bye_rounds) > 1:
            bye_spread_vars.append(statistics.pvariance(bye_rounds))
        elif len(bye_rounds) == 1:
            bye_spread_vars.append(0.0)
    bye_spread_score = statistics.mean(bye_spread_vars) if bye_spread_vars else 0.0
    
    # Default weights
    pair_slack_weight = 20.0
    pair_var_weight = 5.0
    ref_slack_weight = 10.0
    ref_var_weight = 5.0
    team_match_weight = 300.0
    rematch_delay_weight = 1.0
    bye_balance_weight = 10.0
    bye_spread_weight = 3.0
    
    # Intelligent penalty scaling: only penalize imbalance beyond mathematical minimum
    num_teams_count = len([t.name for t in dm.teams])
    num_refs_count = 4  # 4 referees in this test
    teams_per_match_test = 3  # 3-team matches in this test
    total_team_slots = num_rounds * num_refs_count * teams_per_match_test
    min_imbalance = total_team_slots % num_teams_count if num_teams_count > 0 else 0
    
    adjusted_team_match_slack = team_match_slack
    if team_match_slack <= 1 and min_imbalance > 0:
        adjusted_team_match_slack = 0.0
    
    # Calculate total score
    total_score = (pair_slack * pair_slack_weight + 
                  pair_var * pair_var_weight + 
                  ref_slack * ref_slack_weight + 
                  ref_var * ref_var_weight + 
                  adjusted_team_match_slack * team_match_weight +
                  rematch_delay_score * rematch_delay_weight +
                  bye_balance_slack * bye_balance_weight +
                  bye_spread_score * bye_spread_weight)
    
    print(f"Total Score: {total_score:.2f}")
    print(f"\nBreakdown:")
    print(f"  • Pair balance (extremes):      {pair_slack} × {pair_slack_weight} = {pair_slack * pair_slack_weight:.2f}")
    print(f"  • Pair consistency:             {pair_var:.2f} × {pair_var_weight} = {pair_var * pair_var_weight:.2f}")
    print(f"  • Referee balance (extremes):   {ref_slack} × {ref_slack_weight} = {ref_slack * ref_slack_weight:.2f}")
    print(f"  • Referee consistency:          {ref_var:.2f} × {ref_var_weight} = {ref_var * ref_var_weight:.2f}")
    print(f"  • Team match fairness:          {adjusted_team_match_slack} × {team_match_weight} = {adjusted_team_match_slack * team_match_weight:.2f} (raw slack: {team_match_slack})")
    print(f"  • Rematch spacing:              {rematch_delay_score:.2f} × {rematch_delay_weight} = {rematch_delay_score * rematch_delay_weight:.2f}")
    print(f"  • Bye fairness:                 {bye_balance_slack} × {bye_balance_weight} = {bye_balance_slack * bye_balance_weight:.2f}")
    print(f"  • Bye distribution:             {bye_spread_score:.2f} × {bye_spread_weight} = {bye_spread_score * bye_spread_weight:.2f}")
    print("="*60)