import itertools

import numpy as np
import pandas as pd
import streamlit as st

from src.domain.models import ScoreBreakdown


def render_schedule_matrix(schedule_dict, title_prefix, *, refs, num_rounds):
    st.subheader(f"📋 {title_prefix} Match Schedule (Referee × Round)")

    matrix_data = {}
    for ref in sorted(refs):
        matrix_data[ref] = {}
        for round_num in range(1, num_rounds + 1):
            matrix_data[ref][f"Round {round_num}"] = []

    for round_num in sorted(schedule_dict.keys()):
        for entry in schedule_dict[round_num]:
            _, team_names, ref = entry
            if ref and ref in matrix_data:
                teams_in_match = [s.strip() for s in team_names.split(",") if s.strip()]
                match_str = ", ".join(sorted(teams_in_match))
                matrix_data[ref][f"Round {round_num}"].append(match_str)

    matrix_display = {}
    for round_num in range(1, num_rounds + 1):
        round_label = f"Round {round_num}"
        matrix_display[round_label] = {}
        for ref in sorted(refs):
            matches = matrix_data[ref][round_label]
            if matches:
                matrix_display[round_label][ref] = " | ".join(matches)
            else:
                matrix_display[round_label][ref] = "—"

    df_matrix_local = pd.DataFrame(matrix_display).T

    def apply_zebra(styler):
        for i, idx in enumerate(styler.index):
            if i % 2 == 1:
                styler.set_properties(subset=pd.IndexSlice[idx, :], **{"background-color": "rgba(0, 0, 0, 0.05)"})
        return styler

    st.dataframe(
        apply_zebra(df_matrix_local.style),
        use_container_width=True,
        height=200 + (num_rounds * 35),
    )


def render_heatmaps(schedule_dict, counts_global, title_prefix, *, teams, refs, num_rounds, reduced_match_size):
    st.subheader(f"⚖️ {title_prefix} Referee Balance (per team)")
    balance_rows = []
    for team in teams:
        row = {"Team": team}
        for ref in refs:
            row[ref] = counts_global.get((team, ref), 0)
        balance_rows.append(row)

    df_balance_local = pd.DataFrame(balance_rows)
    total_assignments = sum(counts_global.values())
    expected_ref_assignments = total_assignments / (len(teams) * len(refs)) if len(teams) * len(refs) > 0 else 0
    ref_cols = [col for col in df_balance_local.columns if col != "Team"]
    if ref_cols:
        max_deviation = max(
            abs(df_balance_local[ref_cols].min().min() - expected_ref_assignments),
            abs(df_balance_local[ref_cols].max().max() - expected_ref_assignments),
        )
        vmin = expected_ref_assignments - max_deviation
        vmax = expected_ref_assignments + max_deviation
    else:
        vmin = 0
        vmax = 0

    st.dataframe(
        df_balance_local.style.background_gradient(cmap="RdYlGn_r", subset=ref_cols, axis=None, vmin=vmin, vmax=vmax)
        .format(precision=0, subset=ref_cols),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(f"🤝 {title_prefix} Pair Encounters")
    pair_counts = {}
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
                pair_counts[key] = pair_counts.get(key, 0) + 1

    pair_matrix = pd.DataFrame(0, index=teams, columns=teams)
    for (t1, t2), count in pair_counts.items():
        pair_matrix.loc[t1, t2] = count
        pair_matrix.loc[t2, t1] = count

    total_pair_encounters = sum(pair_counts.values())
    num_possible_pairs = len(teams) * (len(teams) - 1) // 2
    expected_encounters = total_pair_encounters / num_possible_pairs if num_possible_pairs > 0 else 0

    mask = np.triu(np.ones(pair_matrix.shape), k=0).astype(bool)
    pair_matrix_lower = pair_matrix.astype(float).mask(mask, np.nan)

    def style_triangle(styler):
        if not pair_matrix_lower.empty:
            max_deviation = max(
                abs(pair_matrix_lower.min().min() - expected_encounters),
                abs(pair_matrix_lower.max().max() - expected_encounters),
            )
        else:
            max_deviation = 0
        vmin = expected_encounters - max_deviation
        vmax = expected_encounters + max_deviation
        styler.background_gradient(cmap="RdYlGn_r", axis=None, vmin=vmin, vmax=vmax)
        styler.format(precision=0, na_rep="")
        return styler

    st.dataframe(
        style_triangle(pair_matrix_lower.style),
        use_container_width=True,
    )

    st.subheader(f"🔥 {title_prefix} Team Match Heatmap")
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
                if t in team_match_counts:
                    team_match_counts[t] += 1

    df_team_heatmap = pd.DataFrame([team_match_counts], index=["Matches"])
    expected_matches = sum(team_match_counts.values()) / len(teams) if len(teams) > 0 else 0
    max_deviation = max(
        abs(min(team_match_counts.values()) - expected_matches),
        abs(max(team_match_counts.values()) - expected_matches),
    ) if team_match_counts else 0
    vmin = expected_matches - max_deviation
    vmax = expected_matches + max_deviation
    st.dataframe(
        df_team_heatmap.style.background_gradient(cmap="RdYlGn_r", axis=None, vmin=vmin, vmax=vmax).format(precision=0),
        use_container_width=True,
    )

    if reduced_match_size is not None and reduced_match_size >= 2:
        st.subheader(f"🧩 {title_prefix} Reduced Match Participation")
        reduced_match_counts = {team: 0 for team in teams}
        for round_num in sorted(schedule_dict.keys()):
            for entry in schedule_dict[round_num]:
                _, team_names, _ = entry
                teams_in_match = [s.strip() for s in team_names.split(",") if s.strip()]
                if len(teams_in_match) == reduced_match_size:
                    for t in teams_in_match:
                        if t in reduced_match_counts:
                            reduced_match_counts[t] += 1

        df_reduced = pd.DataFrame([reduced_match_counts], index=["Reduced Matches"])
        expected_reduced = sum(reduced_match_counts.values()) / len(teams) if len(teams) > 0 else 0
        max_dev = max(
            abs(min(reduced_match_counts.values()) - expected_reduced),
            abs(max(reduced_match_counts.values()) - expected_reduced),
        ) if reduced_match_counts else 0
        vmin = expected_reduced - max_dev
        vmax = expected_reduced + max_dev

        st.dataframe(
            df_reduced.style.background_gradient(cmap="RdYlGn_r", axis=None, vmin=vmin, vmax=vmax).format(precision=0),
            use_container_width=True,
        )

    st.subheader(f"😴 {title_prefix} Team Byes (Rest Rounds)")
    bye_matrix = {}
    team_bye_counts = {}
    for team in teams:
        bye_matrix[team] = {}
        playing_rounds = set()
        for round_num, matches in schedule_dict.items():
            for entry in matches:
                if len(entry) == 3:
                    _, team_names, _ = entry
                    teams_in_match = [s.strip() for s in team_names.split(",") if s.strip()]
                else:
                    teams_in_match = []
                    for part in entry[1:-1]:
                        teams_in_match.extend([s.strip() for s in str(part).split(",") if s.strip()])
                if team in teams_in_match:
                    playing_rounds.add(round_num)
        bye_count = 0
        for round_num in range(1, num_rounds + 1):
            is_bye = 1 if round_num not in playing_rounds else 0
            bye_matrix[team][f"Round {round_num}"] = is_bye
            bye_count += is_bye
        team_bye_counts[team] = bye_count

    df_byes_local = pd.DataFrame(bye_matrix).T
    st.dataframe(
        df_byes_local.style.background_gradient(cmap="RdYlGn_r", axis=None, vmin=0, vmax=1).format(precision=0),
        use_container_width=True,
    )

    st.subheader(f"📊 {title_prefix} Total Byes per Team")
    df_bye_totals_local = pd.DataFrame([team_bye_counts], index=["Bye Count"])
    expected_byes = sum(team_bye_counts.values()) / len(teams) if len(teams) > 0 else 0
    max_deviation = max(
        abs(min(team_bye_counts.values()) - expected_byes),
        abs(max(team_bye_counts.values()) - expected_byes),
    ) if team_bye_counts else 0
    vmin = expected_byes - max_deviation
    vmax = expected_byes + max_deviation
    st.dataframe(
        df_bye_totals_local.style.background_gradient(cmap="RdYlGn_r", axis=None, vmin=vmin, vmax=vmax).format(precision=0),
        use_container_width=True,
    )


def render_score_breakdown(
    score_breakdown: ScoreBreakdown,
    schedule_dict,
    title_prefix,
    *,
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
    reduced_match_size,
):
    st.subheader(f"📊 {title_prefix} Score Breakdown")
    st.write(f"**Final Score**: {score_breakdown.total:.2f}")
    st.write(f"  • Pair balance (extremes): {score_breakdown.pair_slack:.2f} × {pair_slack_weight} = {score_breakdown.pair_slack * pair_slack_weight:.2f}")
    st.write(f"  • Pair consistency: {score_breakdown.pair_var:.2f} × {pair_var_weight} = {score_breakdown.pair_var * pair_var_weight:.2f}")
    st.write(f"  • Referee balance (extremes): {score_breakdown.ref_slack:.2f} × {ref_slack_weight} = {score_breakdown.ref_slack * ref_slack_weight:.2f}")
    st.write(f"  • Referee consistency: {score_breakdown.ref_var:.2f} × {ref_var_weight} = {score_breakdown.ref_var * ref_var_weight:.2f}")
    st.write(f"  • Team match fairness: {score_breakdown.team_match_slack:.2f} × {team_match_weight} = {score_breakdown.team_match_slack * team_match_weight:.2f}")
    st.write(f"  • Rematch spacing: {score_breakdown.rematch_delay:.2f} × {rematch_delay_weight} = {score_breakdown.rematch_delay * rematch_delay_weight:.2f}")
    st.write(f"  • Reduced match balance: {score_breakdown.reduced_match_balance:.2f} × {reduced_match_balance_weight} = {score_breakdown.reduced_match_balance * reduced_match_balance_weight:.2f}")
    if reduced_match_size is not None and reduced_match_size >= 2:
        total_reduced_matches = 0
        for round_num in sorted(schedule_dict.keys()):
            for entry in schedule_dict[round_num]:
                _, team_names, _ = entry
                teams_in_match = [s.strip() for s in team_names.split(",") if s.strip()]
                if len(teams_in_match) == reduced_match_size:
                    total_reduced_matches += 1
        st.write(f"  • Reduced match penalty: {total_reduced_matches:.2f} × {reduced_match_weight} = {total_reduced_matches * reduced_match_weight:.2f}")
    st.write(f"  • Bye fairness: {score_breakdown.bye_balance:.2f} × {bye_balance_weight} = {score_breakdown.bye_balance * bye_balance_weight:.2f}")
    st.write(f"  • Bye distribution: {score_breakdown.bye_spread:.2f} × {bye_spread_weight} = {score_breakdown.bye_spread * bye_spread_weight:.2f}")


def render_results(
    schedule_dict,
    counts_global,
    title_prefix,
    *,
    score_breakdown: ScoreBreakdown,
    show_score_line: bool = False,
    teams,
    refs,
    num_rounds,
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
):
    st.subheader(f"🏷️ {title_prefix}")
    if show_score_line:
        st.success(f"✅ Best-so-far (Score: {score_breakdown.total:.2f})")
    render_schedule_matrix(schedule_dict, title_prefix, refs=refs, num_rounds=num_rounds)
    render_heatmaps(
        schedule_dict,
        counts_global,
        title_prefix,
        teams=teams,
        refs=refs,
        num_rounds=num_rounds,
        reduced_match_size=reduced_match_size,
    )
    render_score_breakdown(
        score_breakdown,
        schedule_dict,
        title_prefix,
        pair_slack_weight=pair_slack_weight,
        pair_var_weight=pair_var_weight,
        ref_slack_weight=ref_slack_weight,
        ref_var_weight=ref_var_weight,
        team_match_weight=team_match_weight,
        rematch_delay_weight=rematch_delay_weight,
        reduced_match_balance_weight=reduced_match_balance_weight,
        reduced_match_weight=reduced_match_weight,
        bye_balance_weight=bye_balance_weight,
        bye_spread_weight=bye_spread_weight,
        reduced_match_size=reduced_match_size,
    )
