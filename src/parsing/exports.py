import itertools

import pandas as pd


def build_export_csv(schedule, counts, teams, refs) -> str:
    rows = []
    for r in sorted(schedule.keys()):
        for entry in schedule[r]:
            _, team_names, ref = entry
            teams_in_match = [s.strip() for s in team_names.split(",") if s.strip()]
            rows.append({
                "Round": r,
                "Referee": ref if ref else "(unassigned)",
                "Teams": ", ".join(sorted(teams_in_match)),
            })

    df_schedule = pd.DataFrame(rows)

    balance_rows = []
    for team in teams:
        row = {"Team": team}
        for ref in refs:
            row[ref] = counts.get((team, ref), 0)
        balance_rows.append(row)
    df_balance = pd.DataFrame(balance_rows)

    pair_counts = {}
    for r in schedule.values():
        for entry in r:
            _, team_names, _ = entry
            teams_in_match = [s.strip() for s in team_names.split(",") if s.strip()]
            for t1, t2 in itertools.combinations(sorted(teams_in_match), 2):
                key = (t1, t2) if t1 < t2 else (t2, t1)
                pair_counts[key] = pair_counts.get(key, 0) + 1

    pair_matrix = pd.DataFrame(0, index=teams, columns=teams)
    for (t1, t2), count in pair_counts.items():
        pair_matrix.loc[t1, t2] = count
        pair_matrix.loc[t2, t1] = count

    csv_schedule = df_schedule.to_csv(index=False)
    csv_balance = df_balance.to_csv(index=False)
    csv_pairs = pair_matrix.to_csv()
    return (
        f"SCHEDULE\n{csv_schedule}"
        f"\n\nREFEREE BALANCE\n{csv_balance}"
        f"\n\nPAIR ENCOUNTERS\n{csv_pairs}"
    )
