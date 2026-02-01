import streamlit as st
import pandas as pd
import itertools
import statistics
import random
from src.datamodel import DataModel
from src.scheduler import build_round_robin_model, solve_and_extract

st.set_page_config(page_title="Tournament Scheduler", layout="wide")
st.title("🏆 Tournament Scheduler")
st.markdown("Generate fair tournament schedules with referee balance.")

# Sidebar inputs
with st.sidebar:
    st.header("Tournament Setup")
    num_teams = st.number_input(
        "Number of Teams",
        min_value=2,
        max_value=100,
        value=12
    )
    teams_per_match = st.number_input(
        "Teams per match",
        min_value=2,
        max_value=10,
        value=3
    )
    num_refs = st.number_input(
        "Number of Referees",
        min_value=1,
        max_value=50,
        value=4
    )
    num_rounds = st.number_input(
        "Number of Rounds",
        min_value=1,
        max_value=20,
        value=8
    )
    
    # Calculate and display suggestions
    suggested_refs = max(1, num_teams // teams_per_match)
    
    # Calculate matches per team with current settings
    team_slots_per_round = num_refs * teams_per_match
    matches_per_team = (num_rounds * team_slots_per_round) / num_teams if num_teams > 0 else 0
    
    # Calculate byes per round
    teams_with_bye_per_round = max(0, num_teams - team_slots_per_round)
    
    # Calculate rounds needed for target matches per team
    target_matches = 8
    suggested_rounds = (target_matches * num_teams) / (num_refs * teams_per_match) if (num_refs * teams_per_match) > 0 else num_rounds
    suggested_rounds = round(suggested_rounds)
    
    st.sidebar.info(
        f"💡 **Suggestions:**\n\n"
        f"**Referees:**\n"
        f"{suggested_refs} optimal (allows {suggested_refs} concurrent)\n\n"
        f"**Rounds for balanced play:**\n"
        f"{suggested_rounds} rounds\n\n"
        f"**Your current setup:**\n"
        f"• {matches_per_team:.1f} matches per team\n"
        f"• {teams_with_bye_per_round} teams with bye per round"
    )
    
    st.sidebar.header("Advanced")
    time_limit = st.slider(
        "Solver Time Limit (seconds)",
        min_value=1,
        max_value=120,
        value=10
    )
    max_attempts = st.sidebar.slider(
        "Max attempts (1 = no retry)",
        min_value=1,
        max_value=50,
        value=10,
        help="Number of scheduling attempts. Set to 1 for single attempt. Higher values run multiple times with shuffled match orders to find the best schedule."
    )
    score_threshold = st.sidebar.slider(
        "Acceptable score (lower is better)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        help="The scheduler will stop retrying once it achieves a score below this threshold. Lower value means better overall balance. Typical excellent: 20-40. Good: 40-60."
    )
    
    st.sidebar.markdown("**Scoring weights** (higher = more important)")
    pair_slack_weight = st.sidebar.slider(
        "Pair encounter balance (prevent extremes)",
        min_value=0.0,
        max_value=50.0,
        value=20.0,
        help="Prevents any pair of teams from playing drastically more/less than others. Focuses on eliminating the worst imbalances. Example: if pairs typically play 2 times, this prevents one pair from playing 5 times while another plays 1 time."
    )
    pair_var_weight = st.sidebar.slider(
        "Pair consistency (smooth distribution)",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        help="Makes all pair encounter frequencies similar to the average. Works together with balance to ensure ALL pairs play roughly the same number of times, not just prevent extremes."
    )
    ref_slack_weight = st.sidebar.slider(
        "Referee balance (prevent extremes)",
        min_value=0.0,
        max_value=50.0,
        value=10.0,
        help="Prevents any team from seeing one referee drastically more than others. Focuses on eliminating the worst referee assignment imbalances."
    )
    ref_var_weight = st.sidebar.slider(
        "Referee consistency (smooth distribution)",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        help="Ensures each team sees each referee a similar number of times. Works together with balance to ensure smooth referee distribution across all teams."
    )
    team_match_weight = st.sidebar.slider(
        "Team match fairness (equal participation)",
        min_value=0.0,
        max_value=500.0,
        value=300.0,
        help="CRITICAL: Ensures all teams play roughly the same number of matches. This is the most important factor for tournament fairness. Higher = stricter enforcement that every team gets similar opportunities to compete."
    )
    rematch_delay_weight = st.sidebar.slider(
        "Rematch spacing (spread rematches apart)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        help="When pairs meet multiple times, this spreads rematches across the tournament. Low weight: minimal spacing (e.g., rounds 1 & 2). High weight: maximize gaps (e.g., rounds 1 & 8). Weighted fairly low as it's less critical than other factors."
    )
    bye_balance_weight = st.sidebar.slider(
        "Bye fairness (equal rest opportunities)",
        min_value=0.0,
        max_value=50.0,
        value=10.0,
        help="Ensures all teams get roughly the same number of bye rounds (rounds where they don't play). Important when not all teams can play each round. Example: 14 teams with 12 playing per round means 2 byes/round."
    )
    bye_spread_weight = st.sidebar.slider(
        "Bye distribution (spread byes apart)",
        min_value=0.0,
        max_value=20.0,
        value=3.0,
        help="Prevents teams from having clustered bye rounds. Better: byes in rounds 1, 4, 7. Worse: byes in rounds 1, 2, 3. Ensures rest opportunities are spread across the tournament."
    )

# Parse inputs
try:
    # Generate team and referee names
    teams = [f"Team {i+1}" for i in range(num_teams)]
    refs = [f"Ref {i+1}" for i in range(num_refs)]

    if not teams or len(teams) < 2:
        st.error("At least 2 teams required.")
        st.stop()
    if not refs or len(refs) < 1:
        st.error("At least 1 referee required.")
        st.stop()

    # Build DataModel
    dm = DataModel()
    for name in teams:
        dm.add_team(name)
    for name in refs:
        dm.add_referee(name)
    dm.set_rounds(num_rounds)
    if teams_per_match > len(teams):
        st.error("teams_per_match cannot be greater than the number of teams")
        st.stop()
    dm.generate_team_matches(teams_per_match)

    # Display setup summary
    col1, col2, col3 = st.columns(3)
    col1.metric("Teams", len(teams))
    col2.metric("Referees", len(refs))
    col3.metric("Rounds", num_rounds)
    
    # Calculate matches per round to fully utilize referees
    matches_per_round = len(refs)  # Use all referees every round
    st.info(f"ℹ️ Will schedule **{matches_per_round} matches per round** (one per referee)")

    st.divider()
    
    def score_schedule(schedule_dict, counts_global):
        # pair slack = max - min of pair counts
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
        if pcs:
            vals = list(pcs.values())
            pair_slack = max(vals) - min(vals)
            pair_var = statistics.pvariance(vals) if len(vals) > 1 else 0.0
        else:
            pair_slack = 0
            pair_var = 0.0

        # referee balance: compute slack and variance of counts per (team, ref)
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

        # team match balance: count how many matches each team plays
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
        
        # Intelligent penalty scaling: only penalize imbalance beyond mathematical minimum
        # Calculate min_imbalance based on total team slots available
        total_team_slots = num_rounds * num_refs * teams_per_match
        min_imbalance = total_team_slots % num_teams if num_teams > 0 else 0
        
        # Adjust team_match_slack based on feasibility:
        # If slack <= 1 and min_imbalance > 0: 1-diff is unavoidable, don't penalize
        # If slack <= 1 and min_imbalance == 0: 1-diff is avoidable, light penalty
        # If slack >= 2: always penalize heavily
        if team_match_slack <= 1 and min_imbalance > 0:
            adjusted_team_match_slack = 0.0
        else:
            adjusted_team_match_slack = team_match_slack

        # Rematch delay: for pairs that meet multiple times, calculate average round gap
        # Higher gap = better spacing between rematches
        pair_meetings = {}  # (t1,t2) -> [list of round numbers]
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
        
        # Calculate rematch delay penalty: penalize small gaps between rematches
        # Use linear scale: ideal gap = num_rounds/2, normalize to 0-5 scale
        rematch_gaps = []
        ideal_gap = num_rounds / 2.0  # Ideal spacing is half the tournament
        for pair, rounds in pair_meetings.items():
            if len(rounds) > 1:
                sorted_rounds = sorted(rounds)
                gaps = [sorted_rounds[i+1] - sorted_rounds[i] for i in range(len(sorted_rounds)-1)]
                avg_gap = sum(gaps) / len(gaps)
                # Penalty: how far below ideal? Normalized to 0-5
                gap_penalty = max(0, (ideal_gap - avg_gap) / ideal_gap * 5.0)
                rematch_gaps.append(gap_penalty)
        rematch_delay_score = statistics.mean(rematch_gaps) if rematch_gaps else 0.0

        # Bye balance: count bye rounds per team (rounds where team doesn't play)
        team_bye_counts = {t: 0 for t in teams}
        team_play_rounds = {t: [] for t in teams}  # track which rounds each team plays
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
            # Teams not playing have a bye
            for t in teams:
                if t not in playing_this_round:
                    team_bye_counts[t] += 1
        
        bye_vals = list(team_bye_counts.values())
        bye_balance_slack = max(bye_vals) - min(bye_vals) if bye_vals else 0

        # Bye spread: for each team, calculate variance of bye round positions
        # Lower variance = byes more spread out
        bye_spread_vars = []
        for team, play_rounds in team_play_rounds.items():
            # Find bye rounds
            bye_rounds = [r for r in range(1, num_rounds + 1) if r not in play_rounds]
            if len(bye_rounds) > 1:
                bye_spread_vars.append(statistics.pvariance(bye_rounds))
            elif len(bye_rounds) == 1:
                bye_spread_vars.append(0.0)
        bye_spread_score = statistics.mean(bye_spread_vars) if bye_spread_vars else 0.0

        # Combine into scalar score: weighted sum (weights exposed in UI)
        score = (pair_slack * pair_slack_weight + 
                 pair_var * pair_var_weight + 
                 ref_slack * ref_slack_weight +
                 ref_var * ref_var_weight +
                 adjusted_team_match_slack * team_match_weight +
                 rematch_delay_score * rematch_delay_weight +
                 bye_balance_slack * bye_balance_weight +
                 bye_spread_score * bye_spread_weight)
        return score, pair_slack, pair_var, ref_slack, ref_var, team_match_slack, team_match_var, rematch_delay_score, bye_balance_slack, bye_spread_score

    # Generate schedule button
    if st.button("Generate Schedule", type="primary", use_container_width=True):
        with st.spinner("Solving..."):
            best_schedule = None
            best_counts = None
            best_score = float("inf")
            attempts = 0
            counts_global = {}

            total_attempts = max_attempts

            # UI placeholders: progress bar and attempts log
            progress_bar = st.progress(0)
            attempts_container = st.empty()
            attempt_records = []

            while attempts < total_attempts:
                attempts += 1
                # shuffle match order to vary solver behavior
                random.shuffle(dm.matches)
                model, solver, vars = build_round_robin_model(
                    dm, num_rounds, 
                    matches_per_round=matches_per_round,
                    pair_slack_weight=pair_slack_weight,
                    pair_var_weight=pair_var_weight,
                    ref_slack_weight=ref_slack_weight,
                    ref_var_weight=ref_var_weight,
                    team_match_weight=team_match_weight
                )
                schedule, counts = solve_and_extract(model, solver, vars, dm, num_rounds, time_limit_seconds=time_limit)

                if schedule is None:
                    last_diag = counts.get("_diag") if isinstance(counts, dict) else None
                    counts_global = {}
                    attempt_records.append({
                        "attempt": attempts,
                        "status": "failed",
                        "score": None,
                        "pair_slack": None,
                        "pair_var": None,
                        "ref_slack": None,
                        "ref_var": None,
                        "team_match_slack": None,
                        "rematch_delay": None,
                        "bye_balance": None,
                        "bye_spread": None,
                    })
                else:
                    counts_global = counts
                    sc, sc_pair_slack, sc_pair_var, sc_ref_slack, sc_ref_var, sc_team_match_slack, sc_team_match_var, sc_rematch_delay, sc_bye_balance, sc_bye_spread = score_schedule(schedule, counts_global)
                    attempt_records.append({
                        "attempt": attempts,
                        "status": "ok",
                        "score": sc,
                        "pair_slack": sc_pair_slack,
                        "pair_var": sc_pair_var,
                        "ref_slack": sc_ref_slack,
                        "ref_var": sc_ref_var,
                        "team_match_slack": sc_team_match_slack,
                        "rematch_delay": sc_rematch_delay,
                        "bye_balance": sc_bye_balance,
                        "bye_spread": sc_bye_spread,
                    })

                    if sc < best_score:
                        best_score = sc
                        best_schedule = schedule
                        best_counts = counts
                        best_components = (sc_pair_slack, sc_pair_var, sc_ref_slack, sc_ref_var, sc_team_match_slack, sc_rematch_delay, sc_bye_balance, sc_bye_spread)

                # update progress UI
                pct = int(attempts / total_attempts * 100)
                progress_bar.progress(pct)
                # build attempts DataFrame for display
                df_attempts = pd.DataFrame(attempt_records)
                # mark best attempt
                if not df_attempts.empty and df_attempts['score'].notna().any():
                    min_score = df_attempts['score'].dropna().min()
                    df_attempts['best'] = df_attempts['score'] == min_score
                else:
                    df_attempts['best'] = False
                
                # Rename columns for better display
                df_attempts_display = df_attempts.rename(columns={
                    'attempt': 'Attempt',
                    'status': 'Status',
                    'score': 'Score',
                    'pair_slack': 'Pair Balance',
                    'pair_var': 'Pair Consistency',
                    'ref_slack': 'Ref Balance',
                    'ref_var': 'Ref Consistency',
                    'team_match_slack': 'Team Match',
                    'rematch_delay': 'Rematch Gap',
                    'bye_balance': 'Bye Balance',
                    'bye_spread': 'Bye Spread',
                    'best': 'Best'
                })
                attempts_container.table(df_attempts_display)

                if best_score <= score_threshold:
                    break

            # done attempts
            schedule = best_schedule
            counts = best_counts
            if 'best_components' not in locals():
                best_components = (0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)

        if schedule is None:
            st.error(
                f"❌ No feasible solution found with current settings ({num_teams} teams, {teams_per_match} per match, {num_refs} refs, {num_rounds} rounds).\n\n"
                f"💡 **Suggestions:**\n"
                f"• Try {suggested_refs} referees (optimal for your teams/match size)\n"
                f"• Try {suggested_rounds} rounds (for balanced play)"
            )
        else:
            st.success("✅ Schedule generated successfully!")

            # Build table of all matches
            rows = []
            for r in sorted(schedule.keys()):
                for entry in schedule[r]:
                    mid, team_names, ref = entry
                    rows.append({
                        "Round": r,
                        "Match ID": mid,
                        "Teams": team_names,
                        "Referee": ref if ref else "(unassigned)"
                    })

            df_schedule = pd.DataFrame(rows)
            df_schedule = df_schedule[["Round", "Referee", "Teams"]]
            df_schedule = df_schedule.sort_values(["Round", "Referee"], kind="stable").reset_index(drop=True)
            st.subheader("📋 Match Schedule")

            def _zebra(row):
                return ["background-color: rgba(28, 131, 225, 0.1)" if row.name % 2 else "background-color: rgba(28, 131, 225, 0.05)" for _ in row]

            st.dataframe(
                df_schedule.style.apply(_zebra, axis=1),
                use_container_width=True,
                hide_index=True
            )

            # Referee balance table
            st.subheader("⚖️ Referee Balance (per team)")
            balance_rows = []
            for team in teams:
                row = {"Team": team}
                for ref in refs:
                    row[ref] = counts.get((team, ref), 0)
                balance_rows.append(row)

            df_balance = pd.DataFrame(balance_rows)
            st.dataframe(
                df_balance.style.background_gradient(cmap="Blues"),
                use_container_width=True,
                hide_index=True
            )
            
            # Pair-encounter matrix
            st.subheader("🤝 Pair Encounters")
            pair_counts = {}
            for r in schedule.values():
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
            st.dataframe(
                pair_matrix.style.background_gradient(cmap="Blues"),
                use_container_width=True
            )

            # Team match counts
            st.subheader("🧮 Team Match Counts")
            team_match_counts = {t: 0 for t in teams}
            for r in schedule.values():
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

            # Team match heatmap
            st.subheader("🔥 Team Match Heatmap")
            df_team_heatmap = pd.DataFrame([team_match_counts], index=["Matches"])
            st.dataframe(
                df_team_heatmap.style.background_gradient(cmap="Blues", axis=1),
                use_container_width=True
            )
            
            # Team byes heatmap
            st.subheader("😴 Team Byes (Rest Rounds)")
            
            # Build bye matrix: rows=teams, cols=rounds, value=1 if bye, 0 if playing
            bye_matrix = {}
            team_bye_counts = {}
            for team in teams:
                bye_matrix[team] = {}
                playing_rounds = set()
                # Find which rounds this team plays
                for round_num, matches in schedule.items():
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
                # For each round, 1 = bye, 0 = playing
                bye_count = 0
                for round_num in range(1, num_rounds + 1):
                    is_bye = 1 if round_num not in playing_rounds else 0
                    bye_matrix[team][f"Round {round_num}"] = is_bye
                    bye_count += is_bye
                team_bye_counts[team] = bye_count
            
            df_byes = pd.DataFrame(bye_matrix).T
            st.dataframe(
                df_byes.style.background_gradient(cmap="Blues", axis=1, vmin=0, vmax=1),
                use_container_width=True
            )
            
            # Team bye totals heatmap
            st.subheader("📊 Total Byes per Team")
            df_bye_totals = pd.DataFrame([team_bye_counts], index=["Bye Count"])
            st.dataframe(
                df_bye_totals.style.background_gradient(cmap="Blues", axis=1),
                use_container_width=True
            )
            
            # Score breakdown
            st.subheader("📊 Score Breakdown")
            score_pair_slack, score_pair_var, score_ref_slack, score_ref_var, score_team_match_slack, score_rematch_delay, score_bye_balance, score_bye_spread = best_components
            st.write(f"**Final Score**: {best_score:.2f}")
            st.write(f"  • Pair balance (extremes): {score_pair_slack:.2f} × {pair_slack_weight} = {score_pair_slack * pair_slack_weight:.2f}")
            st.write(f"  • Pair consistency: {score_pair_var:.2f} × {pair_var_weight} = {score_pair_var * pair_var_weight:.2f}")
            st.write(f"  • Referee balance (extremes): {score_ref_slack:.2f} × {ref_slack_weight} = {score_ref_slack * ref_slack_weight:.2f}")
            st.write(f"  • Referee consistency: {score_ref_var:.2f} × {ref_var_weight} = {score_ref_var * ref_var_weight:.2f}")
            st.write(f"  • Team match fairness: {score_team_match_slack:.2f} × {team_match_weight} = {score_team_match_slack * team_match_weight:.2f}")
            st.write(f"  • Rematch spacing: {score_rematch_delay:.2f} × {rematch_delay_weight} = {score_rematch_delay * rematch_delay_weight:.2f}")
            st.write(f"  • Bye fairness: {score_bye_balance:.2f} × {bye_balance_weight} = {score_bye_balance * bye_balance_weight:.2f}")
            st.write(f"  • Bye distribution: {score_bye_spread:.2f} × {bye_spread_weight} = {score_bye_spread * bye_spread_weight:.2f}")

            # Download CSV
            csv_schedule = df_schedule.to_csv(index=False)
            csv_balance = df_balance.to_csv(index=False)
            csv_pairs = pair_matrix.to_csv()
            combined_csv = (
                f"SCHEDULE\n{csv_schedule}"
                f"\n\nREFEREE BALANCE\n{csv_balance}"
                f"\n\nPAIR ENCOUNTERS\n{csv_pairs}"
            )

            st.download_button(
                label="📥 Download Schedule (CSV)",
                data=combined_csv,
                file_name="tournament_schedule.csv",
                mime="text/csv"
            )

except Exception as e:
    st.error(f"Error: {str(e)}")
