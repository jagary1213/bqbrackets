import itertools
import statistics

from src.domain.models import ScoreBreakdown


def score_schedule(
    schedule_dict,
    counts_global,
    *,
    teams,
    refs,
    num_rounds,
    num_refs,
    teams_per_match,
    num_teams,
    reduced_match_size,
    pair_slack_weight,
    pair_var_weight,
    ref_slack_weight,
    ref_var_weight,
    team_match_weight,
    rematch_delay_weight,
    reduced_match_balance_weight,
    reduced_match_weight,
    bye_balance_weight,
    bye_spread_weight,
) -> ScoreBreakdown:
    pcs = {}
    for r in schedule_dict.values():
        for entry in r:
            if len(entry) == 3:
                _, team_names, _ = entry
                teams_in_match = [s.strip() for s in team_names.split(",") if s.strip()]
            else:
                teams_in_match = []
                for part in entry[1:-1]:
                    teams_in_match.extend([s.strip() for s in str(part).split(",") if s.strip()])
            for t1, t2 in itertools.combinations(sorted(teams_in_match), 2):
                key = (t1, t2) if t1 < t2 else (t2, t1)
                pcs[key] = pcs.get(key, 0) + 1

    all_pair_counts = []
    for t1, t2 in itertools.combinations(sorted(teams), 2):
        key = (t1, t2) if t1 < t2 else (t2, t1)
        all_pair_counts.append(pcs.get(key, 0))

    if all_pair_counts:
        pair_slack = max(all_pair_counts) - min(all_pair_counts)
        pair_var = statistics.pvariance(all_pair_counts) if len(all_pair_counts) > 1 else 0.0
    else:
        pair_slack = 0
        pair_var = 0.0

    ref_vals = []
    for team in teams:
        for ref in refs:
            ref_vals.append(counts_global.get((team, ref), 0))
    if ref_vals:
        ref_slack = max(ref_vals) - min(ref_vals)
        ref_var = statistics.pvariance(ref_vals) if len(ref_vals) > 1 else 0.0
    else:
        ref_slack = 0
        ref_var = 0.0

    team_match_counts = {t: 0 for t in teams}
    for r in schedule_dict.values():
        for entry in r:
            if len(entry) == 3:
                _, team_names, _ = entry
                teams_in_match = [s.strip() for s in team_names.split(",") if s.strip()]
            else:
                teams_in_match = []
                for part in entry[1:-1]:
                    teams_in_match.extend([s.strip() for s in str(part).split(",") if s.strip()])
            for t in teams_in_match:
                team_match_counts[t] += 1

    team_vals = list(team_match_counts.values()) if team_match_counts else [0]
    team_match_slack = max(team_vals) - min(team_vals) if team_vals else 0
    team_match_var = statistics.pvariance(team_vals) if len(team_vals) > 1 else 0.0

    total_team_slots = num_rounds * num_refs * teams_per_match
    min_imbalance = total_team_slots % num_teams if num_teams > 0 else 0

    if team_match_slack <= 1 and min_imbalance > 0:
        adjusted_team_match_slack = 0.0
    else:
        adjusted_team_match_slack = team_match_slack

    pair_meetings = {}
    for round_num, matches in schedule_dict.items():
        for entry in matches:
            if len(entry) == 3:
                _, team_names, _ = entry
                teams_in_match = [s.strip() for s in team_names.split(",") if s.strip()]
            else:
                teams_in_match = []
                for part in entry[1:-1]:
                    teams_in_match.extend([s.strip() for s in str(part).split(",") if s.strip()])
            for t1, t2 in itertools.combinations(sorted(teams_in_match), 2):
                key = (t1, t2) if t1 < t2 else (t2, t1)
                if key not in pair_meetings:
                    pair_meetings[key] = []
                pair_meetings[key].append(round_num)

    rematch_gaps = []
    ideal_gap = num_rounds / 2.0
    for rounds in pair_meetings.values():
        if len(rounds) > 1:
            sorted_rounds = sorted(rounds)
            gaps = [sorted_rounds[i + 1] - sorted_rounds[i] for i in range(len(sorted_rounds) - 1)]
            avg_gap = sum(gaps) / len(gaps)
            gap_penalty = max(0, (ideal_gap - avg_gap) / ideal_gap * 5.0)
            rematch_gaps.append(gap_penalty)
    rematch_delay_score = statistics.mean(rematch_gaps) if rematch_gaps else 0.0

    reduced_match_counts = {t: 0 for t in teams}
    if reduced_match_size is not None and reduced_match_size >= 2:
        for r in schedule_dict.values():
            for entry in r:
                if len(entry) == 3:
                    _, team_names, _ = entry
                    teams_in_match = [s.strip() for s in team_names.split(",") if s.strip()]
                else:
                    teams_in_match = []
                    for part in entry[1:-1]:
                        teams_in_match.extend([s.strip() for s in str(part).split(",") if s.strip()])
                if len(teams_in_match) == reduced_match_size:
                    for t in teams_in_match:
                        if t in reduced_match_counts:
                            reduced_match_counts[t] += 1

    reduced_vals = list(reduced_match_counts.values())
    reduced_match_balance_slack = max(reduced_vals) - min(reduced_vals) if reduced_vals else 0
    total_reduced_matches = sum(reduced_match_counts.values()) / max(1, reduced_match_size) if reduced_match_size else 0

    team_bye_counts = {t: 0 for t in teams}
    team_play_rounds = {t: [] for t in teams}
    for round_num, matches in schedule_dict.items():
        playing_this_round = set()
        for entry in matches:
            if len(entry) == 3:
                _, team_names, _ = entry
                teams_in_match = [s.strip() for s in team_names.split(",") if s.strip()]
            else:
                teams_in_match = []
                for part in entry[1:-1]:
                    teams_in_match.extend([s.strip() for s in str(part).split(",") if s.strip()])
            for t in teams_in_match:
                playing_this_round.add(t)
                if t in team_play_rounds:
                    team_play_rounds[t].append(round_num)
        for t in teams:
            if t not in playing_this_round:
                team_bye_counts[t] += 1

    bye_vals = list(team_bye_counts.values())
    bye_balance_slack = max(bye_vals) - min(bye_vals) if bye_vals else 0

    bye_spread_vars = []
    for team, play_rounds in team_play_rounds.items():
        bye_rounds = [r for r in range(1, num_rounds + 1) if r not in play_rounds]
        if len(bye_rounds) > 1:
            bye_spread_vars.append(statistics.pvariance(bye_rounds))
        elif len(bye_rounds) == 1:
            bye_spread_vars.append(0.0)
    bye_spread_score = statistics.mean(bye_spread_vars) if bye_spread_vars else 0.0

    score = (
        pair_slack * pair_slack_weight
        + pair_var * pair_var_weight
        + ref_slack * ref_slack_weight
        + ref_var * ref_var_weight
        + adjusted_team_match_slack * team_match_weight
        + rematch_delay_score * rematch_delay_weight
        + reduced_match_balance_slack * reduced_match_balance_weight
        + total_reduced_matches * reduced_match_weight
        + bye_balance_slack * bye_balance_weight
        + bye_spread_score * bye_spread_weight
    )

    return ScoreBreakdown(
        total=score,
        pair_slack=pair_slack,
        pair_var=pair_var,
        ref_slack=ref_slack,
        ref_var=ref_var,
        team_match_slack=team_match_slack,
        team_match_var=team_match_var,
        rematch_delay=rematch_delay_score,
        reduced_match_balance=reduced_match_balance_slack,
        bye_balance=bye_balance_slack,
        bye_spread=bye_spread_score,
    )
