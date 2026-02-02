from src.datamodel import DataModel
from src.scheduler import build_round_robin_model, solve_and_extract


def make_sample():
    dm = DataModel()
    for name in ["Alpha", "Bravo", "Charlie", "Delta"]:
        dm.add_team(name)
    for r in ["RefA", "RefB"]:
        dm.add_referee(r)
    dm.set_rounds(3)
    # demo uses 2-team matches by default
    dm.generate_team_matches(2)
    return dm


def main():
    dm = make_sample()
    model, solver, vars = build_round_robin_model(dm, dm.rounds)
    print(f"Teams: {[t.name for t in dm.teams]}")
    print(f"Referees: {[r.name for r in dm.referees]}")
    print(f"Rounds: {dm.rounds}")
    print(f"Generated {len(dm.matches)} pair matches.")

    schedule, counts = solve_and_extract(model, solver, vars, dm, dm.rounds, time_limit_seconds=5)
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
        return

    print("\nSchedule:")
    for r in sorted(schedule.keys()):
        print(f"Round {r}:")
        for entry in schedule[r]:
            # entries are (mid, team_names, referee)
            mid, team_names, ref = entry
            ref_text = ref if ref is not None else "(unassigned)"
            print(f"  Match {mid}: {team_names}    Referee: {ref_text}")

    if counts:
        print("\nPer-team x Referee counts:")
        # print a simple table
        refs = [r.name for r in dm.referees]
        teams = [t.name for t in dm.teams]
        header = "Team\t" + "\t".join(refs)
        print(header)
        for team in teams:
            row = [str(counts.get((team, ref), 0)) for ref in refs]
            print(team + "\t" + "\t".join(row))
    
    # Calculate 8-factor score breakdown
    import itertools
    import statistics
    
    print("\n" + "="*60)
    print("SCORE BREAKDOWN")
    print("="*60)
    
    # 1. Pair balance
    pair_counts = {}
    for r in schedule.values():
        for mid, team_names, ref in r:
            teams_in_match = [s.strip() for s in team_names.split(", ")]
            for t1, t2 in itertools.combinations(sorted(teams_in_match), 2):
                key = (t1, t2) if t1 < t2 else (t2, t1)
                pair_counts[key] = pair_counts.get(key, 0) + 1
    
    # Include ALL possible pairs (even 0-count) for accurate slack calculation
    teams_list = [t.name for t in dm.teams]
    all_pair_counts = []
    for t1, t2 in itertools.combinations(sorted(teams_list), 2):
        key = (t1, t2) if t1 < t2 else (t2, t1)
        all_pair_counts.append(pair_counts.get(key, 0))
    
    pair_vals = all_pair_counts if all_pair_counts else [0]
    pair_slack = max(pair_vals) - min(pair_vals)
    pair_mean = sum(pair_vals) / len(pair_vals) if pair_vals else 0
    pair_var = sum((x - pair_mean) ** 2 for x in pair_vals) / len(pair_vals) if pair_vals else 0.0
    
    # 2. Referee balance
    teams_list = [t.name for t in dm.teams]
    refs_list = [r.name for r in dm.referees]
    all_ref_counts = []
    for team in teams_list:
        for ref in refs_list:
            all_ref_counts.append(counts.get((team, ref), 0))
    
    if all_ref_counts:
        ref_slack = max(all_ref_counts) - min(all_ref_counts)
        ref_mean = sum(all_ref_counts) / len(all_ref_counts)
        ref_var = sum((x - ref_mean) ** 2 for x in all_ref_counts) / len(all_ref_counts)
    else:
        ref_slack, ref_var = 0, 0.0
    
    # 3. Team match fairness
    team_match_counts = {t.name: 0 for t in dm.teams}
    for r in schedule.values():
        for mid, team_names, ref in r:
            for team in team_names.split(", "):
                team_match_counts[team.strip()] += 1
    match_vals = list(team_match_counts.values())
    team_match_slack = max(match_vals) - min(match_vals)
    
    # 4. Rematch delay
    pair_meetings = {}
    for round_num in sorted(schedule.keys()):
        for mid, team_names, ref in schedule[round_num]:
            teams_in_match = [s.strip() for s in team_names.split(", ")]
            for t1, t2 in itertools.combinations(sorted(teams_in_match), 2):
                key = (t1, t2) if t1 < t2 else (t2, t1)
                if key not in pair_meetings:
                    pair_meetings[key] = []
                pair_meetings[key].append(round_num)
    
    rematch_gaps = []
    ideal_gap = dm.rounds / 2.0
    for pair, rounds in pair_meetings.items():
        if len(rounds) > 1:
            sorted_rounds = sorted(rounds)
            gaps = [sorted_rounds[i+1] - sorted_rounds[i] for i in range(len(sorted_rounds)-1)]
            avg_gap = sum(gaps) / len(gaps)
            gap_penalty = max(0, (ideal_gap - avg_gap) / ideal_gap * 5.0)
            rematch_gaps.append(gap_penalty)
    rematch_delay_score = statistics.mean(rematch_gaps) if rematch_gaps else 0.0
    
    # 5. Bye balance and spread
    team_play_rounds = {t.name: [] for t in dm.teams}
    for round_num in sorted(schedule.keys()):
        for mid, team_names, ref in schedule[round_num]:
            for team in team_names.split(", "):
                team_play_rounds[team.strip()].append(round_num)
    
    team_bye_counts = {t: dm.rounds - len(set(rounds)) for t, rounds in team_play_rounds.items()}
    bye_vals = list(team_bye_counts.values())
    bye_balance_slack = max(bye_vals) - min(bye_vals)
    
    bye_spread_vars = []
    for team, play_rounds in team_play_rounds.items():
        all_rounds = set(range(1, dm.rounds + 1))
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
    num_teams = len(dm.teams)
    num_refs = len(dm.referees)
    teams_per_match_count = 2  # Demo uses 2-team matches
    num_rounds_val = dm.rounds
    total_team_slots = num_rounds_val * num_refs * teams_per_match_count
    min_imbalance = total_team_slots % num_teams if num_teams > 0 else 0
    
    adjusted_team_match_slack = team_match_slack
    if team_match_slack <= 1 and min_imbalance > 0:
        adjusted_team_match_slack = 0.0
    
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


if __name__ == "__main__":
    main()
