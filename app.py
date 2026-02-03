import streamlit as st
import pandas as pd
import itertools
import statistics
import random
from src.datamodel import DataModel
from src.scheduler import build_round_robin_model, solve_and_extract

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

def get_num_teams():
    """Get number of teams input"""
    return st.number_input(
        "👥 Number of Teams",
        min_value=2,
        max_value=100,
        value=12,
        help="Total number of teams participating in the tournament"
    )

def get_teams_per_match():
    """Get teams per match input"""
    return st.number_input(
        "🤝 Teams per match",
        min_value=2,
        max_value=10,
        value=3,
        help="How many teams compete simultaneously in each match"
    )

def get_num_referees():
    """Get number of referees input"""
    return st.number_input(
        "🔔 Number of Referees",
        min_value=1,
        max_value=50,
        value=4,
        help="Number of available referees (determines how many concurrent matches possible)"
    )

def get_pair_encounter_inputs():
    """Get pair encounter constraint inputs"""
    enforce_pair_limits = st.sidebar.checkbox(
        "⚖️ Pair encounters",
        value=False,
        help="Guarantees each pair meets between min and max times. Set min=max for exact encounters. Solver will fail if impossible."
    )
    
    min_pair_encounters = None
    max_pair_encounters = None
    if enforce_pair_limits:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            min_pair_encounters = st.number_input("Min encounters", min_value=0, max_value=20, value=1)
        with col2:
            max_pair_encounters = st.number_input("Max encounters", min_value=0, max_value=20, value=2)
        
        if min_pair_encounters > max_pair_encounters:
            st.sidebar.error("Min must be ≤ Max")
    
    return enforce_pair_limits, min_pair_encounters, max_pair_encounters

def get_num_rounds():
    """Get number of rounds input"""
    return st.number_input(
        "🔄 Number of Rounds",
        min_value=1,
        max_value=20,
        value=8,
        help="Total number of rounds in the tournament"
    )

def show_pair_validation(enforce_pair_limits, min_pair_encounters, max_pair_encounters, 
                        num_teams, teams_per_match, num_refs, num_rounds):
    """Show validation for pair encounter constraints"""
    if enforce_pair_limits and min_pair_encounters is not None and max_pair_encounters is not None:
        # Show validation when min=max (exact constraint)
        if min_pair_encounters == max_pair_encounters and min_pair_encounters > 0:
            total_pairs = num_teams * (num_teams - 1) // 2
            pairs_per_match = teams_per_match * (teams_per_match - 1) // 2
            encounters_per_round = num_refs * pairs_per_match
            
            required_rounds = (total_pairs * min_pair_encounters) / encounters_per_round if encounters_per_round > 0 else 0
            suggested_rounds_exact = round(required_rounds)
            
            # Check if exact is mathematically possible (required_rounds must be close to a whole number)
            is_feasible = abs(required_rounds - suggested_rounds_exact) < 0.01
            
            if is_feasible and suggested_rounds_exact == num_rounds:
                st.sidebar.success(f"✅ Perfect! {num_rounds} rounds will give exactly {min_pair_encounters} encounters per pair.")
            elif is_feasible and suggested_rounds_exact != num_rounds:
                st.sidebar.info(f"💡 For exactly {min_pair_encounters} encounters per pair, use {suggested_rounds_exact} rounds (you have {num_rounds}).")
            else:
                # Not feasible - required_rounds is not a whole number
                st.sidebar.warning(
                    f"⚠️ Exact constraint is **mathematically impossible** with your configuration.\n\n"
                    f"You need {required_rounds:.2f} rounds for exactly {min_pair_encounters} encounters per pair.\n\n"
                    f"**Options:**\n"
                    f"• Use min={min_pair_encounters-1 if min_pair_encounters > 1 else 1}, max={min_pair_encounters+1} for flexible range\n"
                    f"• Adjust teams, referees, or teams-per-match to make exact feasible"
                )

def show_suggestions(num_teams, teams_per_match, num_refs, num_rounds):
    """Calculate and display suggestions"""
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

def get_advanced_settings():
    """Get advanced settings from expander"""
    with st.sidebar.expander("⚙️ Advanced Settings", expanded=False):
        st.markdown("**Solver Configuration**")
        time_limit = st.slider(
            "Solver Time Limit (seconds)",
            min_value=1,
            max_value=120,
            value=15
        )
        max_attempts = st.slider(
            "Max attempts (1 = no retry)",
            min_value=1,
            max_value=50,
            value=10,
            help="Number of scheduling attempts. Set to 1 for single attempt. Higher values run multiple times with shuffled match orders to find the best schedule."
        )
        score_threshold = st.slider(
            "Acceptable score (lower is better)",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            help="The scheduler will stop retrying once it achieves a score below this threshold. Lower value means better overall balance. Typical excellent: 20-40. Good: 40-60."
        )
        
        st.markdown("**Hard Constraints**")
        max_match_variance = st.slider(
            "Max match count variance per team",
            min_value=0,
            max_value=3,
            value=0,
            help="Maximum difference in match counts between teams. 0 = all teams must play exactly equal matches (some teams may get more byes). 1+ = teams can differ by this many matches (distributes byes more evenly)."
        )
        
        st.markdown("**Scoring weights** (higher = more important)")
        pair_slack_weight = st.slider(
            "Pair encounter balance (prevent extremes)",
            min_value=0.0,
            max_value=50.0,
            value=20.0,
            help="Prevents any pair of teams from playing drastically more/less than others. Focuses on eliminating the worst imbalances. Example: if pairs typically play 2 times, this prevents one pair from playing 5 times while another plays 1 time."
        )
        pair_var_weight = st.slider(
            "Pair consistency (smooth distribution)",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            help="Makes all pair encounter frequencies similar to the average. Works together with balance to ensure ALL pairs play roughly the same number of times, not just prevent extremes."
        )
        ref_slack_weight = st.slider(
            "Referee balance (prevent extremes)",
            min_value=0.0,
            max_value=50.0,
            value=10.0,
            help="Prevents any team from seeing one referee drastically more than others. Focuses on eliminating the worst referee assignment imbalances."
        )
        ref_var_weight = st.slider(
            "Referee consistency (smooth distribution)",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            help="Ensures each team sees each referee a similar number of times. Works together with balance to ensure smooth referee distribution across all teams."
        )
        team_match_weight = st.slider(
            "Team match fairness (equal participation)",
            min_value=0.0,
            max_value=500.0,
            value=300.0,
            help="CRITICAL: Ensures all teams play roughly the same number of matches. This is the most important factor for tournament fairness. Higher = stricter enforcement that every team gets similar opportunities to compete."
        )
        rematch_delay_weight = st.slider(
            "Rematch spacing (spread rematches apart)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            help="When pairs meet multiple times, this spreads rematches across the tournament. Low weight: minimal spacing (e.g., rounds 1 & 2). High weight: maximize gaps (e.g., rounds 1 & 8). Weighted fairly low as it's less critical than other factors."
        )
        bye_balance_weight = st.slider(
            "Bye fairness (equal rest opportunities)",
            min_value=0.0,
            max_value=50.0,
            value=10.0,
            help="Ensures all teams get roughly the same number of bye rounds (rounds where they don't play). Important when not all teams can play each round. Example: 14 teams with 12 playing per round means 2 byes/round."
        )
        bye_spread_weight = st.slider(
            "Bye distribution (spread byes apart)",
            min_value=0.0,
            max_value=20.0,
            value=3.0,
            help="Prevents teams from having clustered bye rounds. Better: byes in rounds 1, 4, 7. Worse: byes in rounds 1, 2, 3. Ensures rest opportunities are spread across the tournament."
        )
        
        return (time_limit, max_attempts, score_threshold, max_match_variance, pair_slack_weight, pair_var_weight,
                ref_slack_weight, ref_var_weight, team_match_weight, rematch_delay_weight,
                bye_balance_weight, bye_spread_weight)

st.set_page_config(page_title="Tournament Scheduler", layout="wide")
st.title("🏆 Tournament Scheduler")
st.markdown("Generate fair tournament schedules with referee balance.")

# Sidebar inputs - call functions in order
with st.sidebar:
    st.header("Tournament Setup")
    
    # 1. Number of Teams
    num_teams = get_num_teams()
    
    # 2. Teams per match
    teams_per_match = get_teams_per_match()
    
    # 3. Number of Referees
    num_refs = get_num_referees()
    
    # 4. Pair Encounter Inputs
    enforce_pair_limits, min_pair_encounters, max_pair_encounters = get_pair_encounter_inputs()
    
    # 5. Number of Rounds
    num_rounds = get_num_rounds()
    
    # 6. Validation box (after number of rounds)
    show_pair_validation(enforce_pair_limits, min_pair_encounters, max_pair_encounters,
                        num_teams, teams_per_match, num_refs, num_rounds)
    
    # 7. Suggestions box
    show_suggestions(num_teams, teams_per_match, num_refs, num_rounds)
    
    # 8. Advanced Settings
    (time_limit, max_attempts, score_threshold, max_match_variance, pair_slack_weight, pair_var_weight,
     ref_slack_weight, ref_var_weight, team_match_weight, rematch_delay_weight,
     bye_balance_weight, bye_spread_weight) = get_advanced_settings()
    
    # 9. Help & Documentation
    with st.sidebar.expander("📚 Help & Documentation", expanded=False):
        st.markdown("""
        ### About This Tool
        
        This tournament scheduler uses **Google OR-Tools**, a powerful constraint programming solver, to generate optimal tournament schedules. 
        The solver works by defining variables (match assignments, referee assignments) and applying both **hard constraints** (rules that must be satisfied) 
        and **soft constraints** (preferences to optimize) to find the best possible schedule.
        
        The tool makes multiple attempts with different random starting configurations (controlled by "Max attempts") to explore the solution space. 
        It evaluates each schedule using a scoring function and presents the best one found. The process continues until either an excellent 
        score is achieved (below the "Acceptable score" threshold) or the maximum number of attempts is reached.
        
        ---
        
        ### Hard Constraints
        
        These are **mandatory rules** that the solver must satisfy. The schedule will fail if these cannot be met:
        
        - **One match per team per round**: Teams cannot play in multiple matches simultaneously
        - **One referee per match**: Each match must have exactly one referee assigned
        - **One match per referee per round**: Referees can only officiate one match at a time
        - **Max matches per round**: Limited by the number of available referees
        - **Pair encounter limits** (optional): When enabled, enforces min/max times each pair of teams can meet
        - **Match count variance** (optional): Limits how much match counts can differ between teams (0 = perfectly equal)
        
        ---
        
        ### Soft Constraints (Scoring Weights)
        
        These are **preferences** that the solver tries to optimize within the bounds of hard constraints. The weights are **relative to each other** 
        and determine trade-offs when multiple objectives conflict:
        
        - **Higher weight** = More important, solver prioritizes this factor more heavily in the final score
        - **Lower weight** = Less important, solver may sacrifice this to improve higher-weighted factors
        - **Weight of 0** = Ignore this factor completely
        
        For example, if "Team match fairness" has weight 300 and "Rematch spacing" has weight 1, the solver considers equal match distribution 
        300× more important than spreading out rematches. When these objectives conflict, the solver will prioritize equal matches over rematch spacing.
        
        **Available soft constraints:**
        
        - **Pair encounter balance**: Prevents extreme imbalances in how often pairs meet (e.g., one pair plays 5 times while another plays once)
        - **Pair consistency**: Makes all pair encounter counts similar to the average
        - **Referee balance**: Prevents teams from seeing one referee dramatically more than others
        - **Referee consistency**: Ensures each team sees each referee a similar number of times
        - **Team match fairness**: Ensures all teams play roughly equal matches (most critical for fairness)
        - **Rematch spacing**: When pairs meet multiple times, spreads encounters across rounds
        - **Bye fairness**: Distributes bye rounds (rest opportunities) evenly across teams
        - **Bye distribution**: Prevents teams from having clustered bye rounds (spreads them out)
        
        The solver combines all weighted factors into a single score, then searches for the schedule with the lowest score (best balance of all preferences).
        """)

# Parse inputs
try:
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
        
        # Include ALL possible pairs (even those with 0 encounters) for accurate slack calculation
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
                    team_match_weight=team_match_weight,
                    min_pair_encounters=min_pair_encounters,
                    max_pair_encounters=max_pair_encounters,
                    max_match_variance=max_match_variance
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
                
                # Display best score prominently (mobile-friendly)
                if not df_attempts.empty and df_attempts['score'].notna().any():
                    current_best = df_attempts['score'].dropna().min()
                    attempts_container.metric(
                        label=f"Attempt {attempts}/{total_attempts}",
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
                        'bye_balance': 'Bye',
                        'bye_spread': 'B.Spr',
                        'best': '⭐'
                    })
                    # Round numeric columns to 2 decimal places for mobile readability
                    numeric_cols = ['Score', 'Pair', 'P.Var', 'Ref', 'R.Var', 'Team', 'Rmtch', 'Bye', 'B.Spr']
                    for col in numeric_cols:
                        if col in df_attempts_display.columns:
                            df_attempts_display[col] = df_attempts_display[col].apply(
                                lambda x: round(x, 2) if pd.notna(x) else x
                            )
                    st.dataframe(df_attempts_display, use_container_width=True, hide_index=True)

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

            # Build referee-round matrix
            st.subheader("📋 Match Schedule (Referee × Round)")
            
            # Create matrix: rows=refs, cols=rounds
            matrix_data = {}
            for ref in sorted(refs):
                matrix_data[ref] = {}
                for round_num in range(1, num_rounds + 1):
                    matrix_data[ref][f"Round {round_num}"] = []
            
            # Populate matrix with teams
            for round_num in sorted(schedule.keys()):
                for entry in schedule[round_num]:
                    mid, team_names, ref = entry
                    if ref and ref in matrix_data:
                        teams_in_match = [s.strip() for s in team_names.split(",") if s.strip()]
                        match_str = ", ".join(sorted(teams_in_match))
                        matrix_data[ref][f"Round {round_num}"].append(match_str)
            
            # Convert to DataFrame for display (rows=rounds, cols=refs)
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
            
            df_matrix = pd.DataFrame(matrix_display).T
            
            # Apply zebra striping for readability using actual index values
            def apply_zebra(styler):
                for i, idx in enumerate(styler.index):
                    if i % 2 == 1:
                        styler.set_properties(subset=pd.IndexSlice[idx, :], **{'background-color': 'rgba(0, 0, 0, 0.05)'})
                return styler
            
            st.dataframe(
                apply_zebra(df_matrix.style),
                use_container_width=True,
                height=200 + (num_rounds * 35)
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
            
            # Calculate expected referee assignments per team
            total_assignments = sum(counts.values())
            expected_ref_assignments = total_assignments / (len(teams) * len(refs)) if len(teams) * len(refs) > 0 else 0
            
            # Apply diverging colormap centered on expected value
            ref_cols = [col for col in df_balance.columns if col != "Team"]
            max_deviation = max(abs(df_balance[ref_cols].min().min() - expected_ref_assignments),
                               abs(df_balance[ref_cols].max().max() - expected_ref_assignments))
            vmin = expected_ref_assignments - max_deviation
            vmax = expected_ref_assignments + max_deviation
            
            st.dataframe(
                df_balance.style.background_gradient(cmap="RdYlGn_r", subset=ref_cols, axis=None, vmin=vmin, vmax=vmax).format(precision=0, subset=ref_cols),
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
            
            # Create lower-triangle pair matrix
            import numpy as np
            pair_matrix = pd.DataFrame(0, index=teams, columns=teams)
            for (t1, t2), count in pair_counts.items():
                pair_matrix.loc[t1, t2] = count
                pair_matrix.loc[t2, t1] = count
            
            # Calculate expected pair encounters
            total_pair_encounters = sum(pair_counts.values())
            num_possible_pairs = len(teams) * (len(teams) - 1) // 2
            expected_encounters = total_pair_encounters / num_possible_pairs if num_possible_pairs > 0 else 0
            
            # Mask upper triangle and diagonal to show only lower triangle
            mask = np.triu(np.ones(pair_matrix.shape), k=0).astype(bool)
            pair_matrix_lower = pair_matrix.astype(float).mask(mask, np.nan)
            
            # Style with diverging colormap centered on expected value
            def style_triangle(styler):
                # Calculate max deviation from expected for symmetric color scale
                max_deviation = max(abs(pair_matrix_lower.min().min() - expected_encounters), 
                                   abs(pair_matrix_lower.max().max() - expected_encounters))
                vmin = expected_encounters - max_deviation
                vmax = expected_encounters + max_deviation
                
                styler.background_gradient(cmap="RdYlGn_r", axis=None, vmin=vmin, vmax=vmax)
                styler.format(precision=0, na_rep="")
                return styler
            
            st.dataframe(
                style_triangle(pair_matrix_lower.style),
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
            
            # Calculate expected matches per team
            expected_matches = sum(team_match_counts.values()) / len(teams) if len(teams) > 0 else 0
            max_deviation = max(abs(min(team_match_counts.values()) - expected_matches),
                               abs(max(team_match_counts.values()) - expected_matches))
            vmin = expected_matches - max_deviation
            vmax = expected_matches + max_deviation
            
            st.dataframe(
                df_team_heatmap.style.background_gradient(cmap="RdYlGn_r", axis=None, vmin=vmin, vmax=vmax).format(precision=0),
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
            
            # Calculate expected bye rate (proportion of teams with byes each round)
            # For binary matrix (0 or 1), center on 0.5 for diverging colormap
            expected_bye = 0.5
            
            st.dataframe(
                df_byes.style.background_gradient(cmap="RdYlGn_r", axis=None, vmin=0, vmax=1).format(precision=0),
                use_container_width=True
            )
            
            # Team bye totals heatmap
            st.subheader("📊 Total Byes per Team")
            df_bye_totals = pd.DataFrame([team_bye_counts], index=["Bye Count"])
            
            # Calculate expected byes per team
            expected_byes = sum(team_bye_counts.values()) / len(teams) if len(teams) > 0 else 0
            max_deviation = max(abs(min(team_bye_counts.values()) - expected_byes),
                               abs(max(team_bye_counts.values()) - expected_byes))
            vmin = expected_byes - max_deviation
            vmax = expected_byes + max_deviation
            
            st.dataframe(
                df_bye_totals.style.background_gradient(cmap="RdYlGn_r", axis=None, vmin=vmin, vmax=vmax).format(precision=0),
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
            # Build df_schedule for CSV export
            rows = []
            for r in sorted(schedule.keys()):
                for entry in schedule[r]:
                    mid, team_names, ref = entry
                    teams_in_match = [s.strip() for s in team_names.split(",") if s.strip()]
                    rows.append({
                        "Round": r,
                        "Referee": ref if ref else "(unassigned)",
                        "Teams": ", ".join(sorted(teams_in_match))
                    })
            
            df_schedule = pd.DataFrame(rows)


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
