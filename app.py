import streamlit as st
import pandas as pd
import itertools
import random
from src.domain.models import ScoreBreakdown
from src.ui.sidebar import collect_sidebar_state
from src.datamodel import DataModel
from src.parsing.exports import build_export_csv
from src.parsing.hints import parse_hint_matrix_csv, parse_hint_matrix_to_schedule
from src.rendering.schedule_views import render_results
from src.scheduler import build_round_robin_model, solve_and_extract
from src.scoring.score import score_schedule

st.set_page_config(page_title="Tournament Scheduler", layout="wide")

# Custom CSS for green checkbox
st.markdown("""
    <style>
    /* Style for checked checkbox */
    .stCheckbox > label > div[data-testid="stMarkdownContainer"] > p {
        color: inherit;
    }
    /* When checkbox is checked, make the entire row green */
    input[type="checkbox"]:checked + div {
        color: #00cc00 !important;
    }
    input[type="checkbox"]:checked ~ div {
        color: #00cc00 !important;
    }
    </style>
""", unsafe_allow_html=True)



st.title("🏆 Tournament Scheduler")
st.markdown("Generate fair tournament schedules with referee balance.")

sidebar_state, hint_csv_file = collect_sidebar_state()

num_teams = sidebar_state.num_teams
min_teams_per_match = sidebar_state.min_teams_per_match
max_teams_per_match = sidebar_state.max_teams_per_match
num_refs = sidebar_state.num_refs
num_rounds = sidebar_state.num_rounds
reduced_matches_per_round = sidebar_state.reduced_matches_per_round
enforce_pair_limits = sidebar_state.enforce_pair_limits
min_pair_encounters = sidebar_state.min_pair_encounters
max_pair_encounters = sidebar_state.max_pair_encounters
enforce_ref_limits = sidebar_state.enforce_ref_limits
min_ref_encounters = sidebar_state.min_ref_encounters
max_ref_encounters = sidebar_state.max_ref_encounters
optimal_solve = sidebar_state.optimal_solve
time_limit = sidebar_state.time_limit
max_attempts = sidebar_state.max_attempts
score_threshold = sidebar_state.score_threshold
max_match_variance = sidebar_state.max_match_variance
pair_slack_weight = sidebar_state.pair_slack_weight
pair_var_weight = sidebar_state.pair_var_weight
ref_slack_weight = sidebar_state.ref_slack_weight
ref_var_weight = sidebar_state.ref_var_weight
team_match_weight = sidebar_state.team_match_weight
rematch_delay_weight = sidebar_state.rematch_delay_weight
bye_balance_weight = sidebar_state.bye_balance_weight
bye_spread_weight = sidebar_state.bye_spread_weight
reduced_match_balance_weight = sidebar_state.reduced_match_balance_weight
reduced_match_weight = sidebar_state.reduced_match_weight

# Parse inputs
try:
    teams_per_match = max_teams_per_match
    reduced_match_size = min_teams_per_match if min_teams_per_match < max_teams_per_match else None

    # Generate team and referee names
    teams = [f"T{i+1}" for i in range(num_teams)]
    refs = [f"Ref{i+1}" for i in range(num_refs)]

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
        st.error("Teams per match (max) cannot be greater than the number of teams")
        st.stop()

    if reduced_matches_per_round == 0:
        reduced_match_size = None
    if reduced_match_size is not None and reduced_match_size < 2:
        reduced_match_size = None

    if reduced_match_size is None:
        dm.generate_team_matches(teams_per_match)
    else:
        dm.generate_mixed_team_matches(teams_per_match, reduced_match_size)

    score_context = {
        "teams": teams,
        "refs": refs,
        "num_rounds": num_rounds,
        "num_refs": num_refs,
        "teams_per_match": teams_per_match,
        "num_teams": num_teams,
        "reduced_match_size": reduced_match_size,
        "pair_slack_weight": pair_slack_weight,
        "pair_var_weight": pair_var_weight,
        "ref_slack_weight": ref_slack_weight,
        "ref_var_weight": ref_var_weight,
        "team_match_weight": team_match_weight,
        "rematch_delay_weight": rematch_delay_weight,
        "reduced_match_balance_weight": reduced_match_balance_weight,
        "reduced_match_weight": reduced_match_weight,
        "bye_balance_weight": bye_balance_weight,
        "bye_spread_weight": bye_spread_weight,
    }
    render_context = {
        "teams": teams,
        "refs": refs,
        "num_rounds": num_rounds,
        "reduced_match_size": reduced_match_size,
        "pair_slack_weight": pair_slack_weight,
        "pair_var_weight": pair_var_weight,
        "ref_slack_weight": ref_slack_weight,
        "ref_var_weight": ref_var_weight,
        "team_match_weight": team_match_weight,
        "rematch_delay_weight": rematch_delay_weight,
        "reduced_match_balance_weight": reduced_match_balance_weight,
        "reduced_match_weight": reduced_match_weight,
        "bye_balance_weight": bye_balance_weight,
        "bye_spread_weight": bye_spread_weight,
    }

    # Display setup summary
    col1, col2, col3 = st.columns(3)
    col1.metric("Teams", len(teams))
    col2.metric("Referees", len(refs))
    col3.metric("Rounds", num_rounds)
    
    # Calculate matches per round to fully utilize referees
    matches_per_round = len(refs)  # Use all referees every round
    st.info(f"ℹ️ Will schedule **{matches_per_round} matches per round** (one per referee)")

    st.divider()

    action_panel = st.container()
    results_panel = st.container()

    if "generating" not in st.session_state:
        st.session_state["generating"] = False
    if "schedule_generated" not in st.session_state:
        st.session_state["schedule_generated"] = False

    hint_results_container = results_panel.empty()
    live_results_container = results_panel.empty()

    # Parse optional CSV hints (matrix format)
    hint_assignments = []
    hint_warnings = []
    hint_info_message = None
    if 'hint_csv_file' in locals() and hint_csv_file is not None:
        team_name_to_id = {t.name: t.id for t in dm.teams}
        ref_name_to_id = {r.name: r.id for r in dm.referees}
        match_lookup = {frozenset(t.id for t in m.teams): m.id for m in dm.matches}
        hint_result = parse_hint_matrix_csv(
            hint_csv_file,
            team_name_to_id,
            ref_name_to_id,
            match_lookup,
            num_rounds
        )
        hint_assignments = hint_result.hints
        if hint_assignments:
            hint_info_message = f"💡 Loaded {len(hint_assignments)} hint(s) from CSV."
        if hint_result.warnings:
            hint_warnings = hint_result.warnings
    
    with action_panel:
        if 'hint_csv_file' in locals() and hint_csv_file is not None:
            if hint_info_message:
                st.info(hint_info_message)
            if hint_warnings:
                with st.expander("⚠️ Hint CSV issues", expanded=False):
                    for msg in hint_warnings:
                        st.write(f"• {msg}")
            if st.button(
                "Score Hint CSV",
                type="secondary",
                use_container_width=True,
                disabled=st.session_state.get("generating")
            ):
                match_lookup = {frozenset(t.id for t in m.teams): m.id for m in dm.matches}
                hint_schedule_result = parse_hint_matrix_to_schedule(
                    hint_csv_file,
                    teams,
                    refs,
                    match_lookup,
                    num_rounds
                )
                if hint_schedule_result.warnings:
                    with st.expander("⚠️ Hint CSV score issues", expanded=False):
                        for msg in hint_schedule_result.warnings:
                            st.write(f"• {msg}")
                score_breakdown = score_schedule(
                    hint_schedule_result.schedule,
                    hint_schedule_result.counts,
                    **score_context,
                )
                with hint_results_container.container():
                    render_results(
                        hint_schedule_result.schedule,
                        hint_schedule_result.counts,
                        "Hint CSV",
                        score_breakdown=score_breakdown,
                        **render_context,
                    )

    # Generate schedule button
    with action_panel:
        generate_clicked = st.button("Generate Schedule", type="primary", use_container_width=True)

    if generate_clicked:
        with action_panel:
            st.session_state["generating"] = True
            st.session_state["schedule_generated"] = False
            try:
                with st.spinner("Solving..."):
                    hint_results_container.empty()
                    live_results_container.empty()
                    best_schedule = None
                    best_counts = None
                    best_score = float("inf")
                    attempts = 0
                    counts_global = {}

                    total_attempts = max_attempts

                    # UI placeholders: progress bar and attempts log
                    progress_bar = st.progress(0)
                    attempts_container = st.empty()
                    failure_container = st.empty()
                    attempt_records = []
                    failure_diags = []

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
                            team_match_weight=team_match_weight,
                            min_pair_encounters=min_pair_encounters,
                            max_pair_encounters=max_pair_encounters,
                            max_match_variance=max_match_variance,
                            reduced_match_size=reduced_match_size,
                            max_reduced_matches_per_round=reduced_matches_per_round,
                            reduced_match_weight=reduced_match_weight,
                            min_ref_encounters=min_ref_encounters,
                            max_ref_encounters=max_ref_encounters,
                            reduced_match_balance_weight=reduced_match_balance_weight,
                            hint_assignments=hint_assignments
                        )
                        diag_context = {
                            "min_pair_encounters": min_pair_encounters,
                            "max_pair_encounters": max_pair_encounters,
                            "min_ref_encounters": min_ref_encounters,
                            "max_ref_encounters": max_ref_encounters,
                            "max_match_variance": max_match_variance,
                            "reduced_match_size": reduced_match_size,
                            "max_reduced_matches_per_round": reduced_matches_per_round,
                            "matches_per_round_target": matches_per_round,
                            "teams_per_match": teams_per_match,
                            "num_refs": num_refs
                        }
                        schedule, counts = solve_and_extract(
                            model, solver, vars, dm, num_rounds,
                            time_limit_seconds=time_limit,
                            diag_context=diag_context
                        )

                        if schedule is None:
                            last_diag = counts.get("_diag") if isinstance(counts, dict) else None
                            counts_global = {}
                            if last_diag:
                                failure_diags.append({"attempt": attempts, "diag": last_diag})
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
                                "reduced_balance": None,
                                "bye_balance": None,
                                "bye_spread": None,
                            })
                        else:
                            counts_global = counts
                            score_breakdown = score_schedule(
                                schedule,
                                counts_global,
                                **score_context,
                            )
                            attempt_records.append({
                                "attempt": attempts,
                                "status": "ok",
                                "score": score_breakdown.total,
                                "pair_slack": score_breakdown.pair_slack,
                                "pair_var": score_breakdown.pair_var,
                                "ref_slack": score_breakdown.ref_slack,
                                "ref_var": score_breakdown.ref_var,
                                "team_match_slack": score_breakdown.team_match_slack,
                                "rematch_delay": score_breakdown.rematch_delay,
                                "reduced_balance": score_breakdown.reduced_match_balance,
                                "bye_balance": score_breakdown.bye_balance,
                                "bye_spread": score_breakdown.bye_spread,
                            })

                            if score_breakdown.total < best_score:
                                best_score = score_breakdown.total
                                best_schedule = schedule
                                best_counts = counts
                                best_components = score_breakdown
                                with live_results_container.container():
                                    render_results(
                                        best_schedule,
                                        best_counts,
                                        "Best-so-far",
                                        score_breakdown=score_breakdown,
                                        show_score_line=True,
                                        **render_context,
                                    )

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
                        
                        # Display best score prominently (mobile-friendly)
                        if not df_attempts.empty and df_attempts['score'].notna().any():
                            current_best = df_attempts['score'].dropna().min()
                            run_label = "Optimal run" if optimal_solve else f"Attempt {attempts}/{total_attempts}"
                            attempts_container.metric(
                                label=run_label,
                                value=f"Best Score: {current_best:.1f}",
                                delta=None
                            )
                        
                        # Show detailed table in expander (mobile-friendly)
                        with attempts_container.expander("📊 View All Attempts", expanded=False):
                            df_attempts_display = df_attempts.rename(columns={
                                'attempt': '#',
                                'status': 'Status',
                                'score': 'Score',
                                'pair_slack': 'Pair',
                                'pair_var': 'P.Var',
                                'ref_slack': 'Ref',
                                'ref_var': 'R.Var',
                                'team_match_slack': 'Team',
                                'rematch_delay': 'Rmtch',
                                'reduced_balance': 'Red',
                                'bye_balance': 'Bye',
                                'bye_spread': 'B.Spr',
                                'best': '⭐'
                            })
                            # Round numeric columns to 2 decimal places for mobile readability
                            numeric_cols = ['Score', 'Pair', 'P.Var', 'Ref', 'R.Var', 'Team', 'Rmtch', 'Red', 'Bye', 'B.Spr']
                            for col in numeric_cols:
                                if col in df_attempts_display.columns:
                                    df_attempts_display[col] = df_attempts_display[col].apply(
                                        lambda x: round(x, 2) if pd.notna(x) else x
                                    )
                            st.dataframe(df_attempts_display, use_container_width=True, hide_index=True)

                        # Show latest failure diagnostics near progress bar
                        if failure_diags:
                            latest_failure = failure_diags[-1]
                            failure_container.warning(
                                f"❌ Attempt {latest_failure['attempt']} failed.\n\n{latest_failure['diag']}"
                            )
                        else:
                            failure_container.empty()

                        if best_score <= score_threshold:
                            break

                # done attempts
                schedule = best_schedule
                counts = best_counts
                if 'best_components' not in locals():
                    best_components = ScoreBreakdown(
                        total=0.0,
                        pair_slack=0.0,
                        pair_var=0.0,
                        ref_slack=0.0,
                        ref_var=0.0,
                        team_match_slack=0.0,
                        team_match_var=0.0,
                        rematch_delay=0.0,
                        reduced_match_balance=0.0,
                        bye_balance=0.0,
                        bye_spread=0.0,
                    )

                # Ensure suggestion variables exist for failure message
                if 'suggested_refs' not in locals():
                    suggested_refs = max(1, num_teams // teams_per_match)
                if 'suggested_rounds' not in locals():
                    target_matches = 8
                    suggested_rounds = (target_matches * num_teams) / (num_refs * teams_per_match) if (num_refs * teams_per_match) > 0 else num_rounds
                    suggested_rounds = round(suggested_rounds)

            finally:
                st.session_state["generating"] = False

        if schedule is None:
            st.error(
                f"❌ No feasible solution found with current settings ({num_teams} teams, {teams_per_match} per match, {num_refs} refs, {num_rounds} rounds).\n\n"
                f"💡 **Suggestions:**\n"
                f"• Try {suggested_refs} referees (optimal for your teams/match size)\n"
                f"• Try {suggested_rounds} rounds (for balanced play)"
            )
        else:
            st.session_state["schedule_generated"] = True
            live_results_container.empty()
            with live_results_container.container():
                st.success(f"✅ Schedule generated successfully! (Score: {best_score:.2f})")
                render_results(
                    schedule,
                    counts,
                    "Final",
                    score_breakdown=best_components,
                    **render_context,
                )
                # Download CSV
                combined_csv = build_export_csv(schedule, counts, teams, refs)

                st.download_button(
                    label="📥 Download Schedule (CSV)",
                    data=combined_csv,
                    file_name="tournament_schedule.csv",
                    mime="text/csv"
                )

except Exception as e:
    st.error(f"Error: {str(e)}")
