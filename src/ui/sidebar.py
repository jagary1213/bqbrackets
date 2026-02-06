from __future__ import annotations

from typing import Optional, Tuple
import streamlit as st

from src.domain.models import SidebarState


def get_num_teams() -> int:
    return st.number_input(
        "👥 Number of Teams",
        min_value=2,
        max_value=100,
        value=12,
        help="Total number of teams participating in the tournament"
    )


def get_teams_per_match() -> Tuple[int, int]:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        min_teams = st.number_input(
            "🤝 Teams per match (min)",
            min_value=2,
            max_value=10,
            value=3,
            help="Minimum teams in a match"
        )
    with col2:
        max_teams = st.number_input(
            "🤝 Teams per match (max)",
            min_value=2,
            max_value=10,
            value=3,
            help="Maximum teams in a match"
        )

    if min_teams > max_teams:
        st.sidebar.error("Min teams per match must be ≤ Max")

    return min_teams, max_teams


def compute_max_reduced_matches_per_round(
    min_teams_per_match: int,
    max_teams_per_match: int,
    num_refs: int,
    num_teams: int
) -> int:
    if min_teams_per_match >= max_teams_per_match:
        return 0

    reduced_match_size = min_teams_per_match
    max_by_refs = num_refs
    max_by_teams = num_teams // reduced_match_size if reduced_match_size > 0 else 0
    return max(0, min(max_by_refs, max_by_teams))


def get_num_referees() -> int:
    return st.number_input(
        "🔔 Number of Referees",
        min_value=1,
        max_value=50,
        value=4,
        help="Number of available referees (determines how many concurrent matches possible)"
    )


def get_pair_encounter_inputs() -> Tuple[bool, Optional[int], Optional[int]]:
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


def get_referee_encounter_inputs() -> Tuple[bool, Optional[int], Optional[int]]:
    enforce_ref_limits = st.sidebar.checkbox(
        "🧑‍⚖️ Referee encounters",
        value=False,
        help="Guarantees each team sees each referee between min and max times. Set min=max for exact encounters. Solver will fail if impossible."
    )

    min_ref_encounters = None
    max_ref_encounters = None
    if enforce_ref_limits:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            min_ref_encounters = st.number_input("Min encounters", min_value=0, max_value=50, value=0)
        with col2:
            max_ref_encounters = st.number_input("Max encounters", min_value=0, max_value=50, value=2)

        if min_ref_encounters > max_ref_encounters:
            st.sidebar.error("Min must be ≤ Max")

    return enforce_ref_limits, min_ref_encounters, max_ref_encounters


def get_num_rounds() -> int:
    return st.number_input(
        "🔄 Number of Rounds",
        min_value=1,
        max_value=20,
        value=8,
        help="Total number of rounds in the tournament"
    )


def show_pair_validation(
    enforce_pair_limits: bool,
    min_pair_encounters: Optional[int],
    max_pair_encounters: Optional[int],
    num_teams: int,
    teams_per_match: int,
    num_refs: int,
    num_rounds: int
) -> None:
    if enforce_pair_limits and min_pair_encounters is not None and max_pair_encounters is not None:
        if min_pair_encounters == max_pair_encounters and min_pair_encounters > 0:
            total_pairs = num_teams * (num_teams - 1) // 2
            pairs_per_match = teams_per_match * (teams_per_match - 1) // 2
            encounters_per_round = num_refs * pairs_per_match

            required_rounds = (total_pairs * min_pair_encounters) / encounters_per_round if encounters_per_round > 0 else 0
            suggested_rounds_exact = round(required_rounds)

            is_feasible = abs(required_rounds - suggested_rounds_exact) < 0.01

            if is_feasible and suggested_rounds_exact == num_rounds:
                st.sidebar.success(f"✅ Perfect! {num_rounds} rounds will give exactly {min_pair_encounters} encounters per pair.")
            elif is_feasible and suggested_rounds_exact != num_rounds:
                st.sidebar.info(f"💡 For exactly {min_pair_encounters} encounters per pair, use {suggested_rounds_exact} rounds (you have {num_rounds}).")
            else:
                st.sidebar.warning(
                    f"⚠️ Exact constraint is **mathematically impossible** with your configuration.\n\n"
                    f"You need {required_rounds:.2f} rounds for exactly {min_pair_encounters} encounters per pair.\n\n"
                    f"**Options:**\n"
                    f"• Use min={min_pair_encounters-1 if min_pair_encounters > 1 else 1}, max={min_pair_encounters+1} for flexible range\n"
                    f"• Adjust teams, referees, or teams-per-match to make exact feasible"
                )


def show_ref_validation(
    enforce_ref_limits: bool,
    min_ref_encounters: Optional[int],
    max_ref_encounters: Optional[int],
    num_teams: int,
    teams_per_match: int,
    num_refs: int,
    num_rounds: int
) -> None:
    if enforce_ref_limits and min_ref_encounters is not None and max_ref_encounters is not None:
        if min_ref_encounters == max_ref_encounters and min_ref_encounters > 0:
            total_team_ref_encounters = num_teams * num_refs * min_ref_encounters
            team_slots_per_round = num_refs * teams_per_match
            required_rounds = (total_team_ref_encounters / team_slots_per_round) if team_slots_per_round > 0 else 0
            suggested_rounds_exact = round(required_rounds)

            is_feasible = abs(required_rounds - suggested_rounds_exact) < 0.01

            if is_feasible and suggested_rounds_exact == num_rounds:
                st.sidebar.success(
                    f"✅ Perfect! {num_rounds} rounds will give exactly {min_ref_encounters} referee encounters per team/ref."
                )
            elif is_feasible and suggested_rounds_exact != num_rounds:
                st.sidebar.info(
                    f"💡 For exactly {min_ref_encounters} encounters per team/ref, use {suggested_rounds_exact} rounds (you have {num_rounds})."
                )
            else:
                st.sidebar.warning(
                    f"⚠️ Exact referee encounters are **mathematically impossible** with your configuration.\n\n"
                    f"You need {required_rounds:.2f} rounds for exactly {min_ref_encounters} encounters per team/ref (assuming full matches each round).\n\n"
                    f"**Options:**\n"
                    f"• Use min={min_ref_encounters-1 if min_ref_encounters > 1 else 0}, max={min_ref_encounters+1} for flexible range\n"
                    f"• Adjust teams, referees, teams-per-match, or rounds"
                )


def show_suggestions(
    num_teams: int,
    max_teams_per_match: int,
    min_teams_per_match: int,
    num_refs: int,
    num_rounds: int,
    max_reduced_matches_per_round: int
) -> None:
    suggested_refs = max(1, num_teams // max_teams_per_match)
    team_slots_per_round = num_refs * max_teams_per_match
    matches_per_team = (num_rounds * team_slots_per_round) / num_teams if num_teams > 0 else 0

    reduced_match_size = min_teams_per_match if min_teams_per_match < max_teams_per_match else None
    max_reduced = max(0, min(max_reduced_matches_per_round or 0, num_refs))
    reduced_delta = max_teams_per_match - min_teams_per_match if reduced_match_size else 0
    team_slots_with_max_reduced = team_slots_per_round - (max_reduced * reduced_delta) if reduced_match_size and reduced_match_size >= 2 else team_slots_per_round

    teams_with_bye_per_round = max(0, num_teams - team_slots_per_round)
    teams_with_bye_if_max_reduced = max(0, num_teams - team_slots_with_max_reduced)

    target_matches = 8
    suggested_rounds = (target_matches * num_teams) / (num_refs * max_teams_per_match) if (num_refs * max_teams_per_match) > 0 else num_rounds
    suggested_rounds = max(1, round(suggested_rounds))

    if num_teams > 0 and max_teams_per_match > 0 and num_refs > 0:
        full_slots = num_refs * max_teams_per_match
        reduced_size = min_teams_per_match if min_teams_per_match < max_teams_per_match else None
        max_reduced_per_round = max(0, min(max_reduced_matches_per_round or 0, num_refs))

        def is_feasible_rounds(r: int) -> bool:
            if r <= 0:
                return False
            if not reduced_size or reduced_size < 2 or max_reduced_per_round == 0:
                return (full_slots * r) % num_teams == 0
            total_full = full_slots * r
            max_reduced_total = max_reduced_per_round * r
            for reduced_count in range(0, max_reduced_total + 1):
                if (total_full - reduced_count) % num_teams == 0:
                    return True
            return False

        if not is_feasible_rounds(suggested_rounds):
            for offset in range(1, 21):
                down = suggested_rounds - offset
                up = suggested_rounds + offset
                if down >= 1 and is_feasible_rounds(down):
                    suggested_rounds = down
                    break
                if up <= 20 and is_feasible_rounds(up):
                    suggested_rounds = up
                    break

    st.sidebar.info(
        f"💡 **Suggestions:**\n\n"
        f"**Referees:**\n"
        f"{suggested_refs} optimal (allows {suggested_refs} concurrent)\n\n"
        f"**Rounds for balanced play:**\n"
        f"{suggested_rounds} rounds\n\n"
        f"**Your current setup:**\n"
        f"• {matches_per_team:.1f} matches per team\n"
        f"• Teams per match: {min_teams_per_match}-{max_teams_per_match}\n"
        f"• Team slots per round (full): {team_slots_per_round}\n"
        f"• Team slots per round (with max reduced): {team_slots_with_max_reduced}\n"
        f"• Teams playing per round (full): {min(num_teams, team_slots_per_round)}\n"
        f"• Teams playing per round (max reduced): {min(num_teams, team_slots_with_max_reduced)}\n"
        f"• Teams with bye per round: {teams_with_bye_per_round}\n"
        f"• Teams with bye (if max reduced): {teams_with_bye_if_max_reduced}"
    )


def get_advanced_settings() -> Tuple[bool, Optional[int], int, float, int, float, float, float, float, float, float, float, float, float, float]:
    with st.sidebar.expander("⚙️ Advanced Settings", expanded=False):
        st.markdown("**Solver Configuration**")
        optimal_solve = st.checkbox(
            "Optimal solve (single run, no time limit)",
            value=False,
            help="Runs a single optimization without time limit. Disables retries and time limit settings."
        )
        time_limit = st.slider(
            "Solver Time Limit (seconds)",
            min_value=1,
            max_value=120,
            value=15,
            disabled=optimal_solve
        )
        max_attempts = st.slider(
            "Max attempts (1 = no retry)",
            min_value=1,
            max_value=50,
            value=10,
            disabled=optimal_solve,
            help="Number of scheduling attempts. Set to 1 for single attempt. Higher values run multiple times with shuffled match orders to find the best schedule."
        )
        if optimal_solve:
            time_limit = None
            max_attempts = 1
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

        reduced_match_balance_weight = st.slider(
            "Reduced match balance (equal distribution)",
            min_value=0.0,
            max_value=200.0,
            value=100.0,
            help="Encourages reduced-size matches to be spread evenly across teams. Higher values make distribution more uniform."
        )

        reduced_match_weight = st.slider(
            "Reduced match penalty (prefer full matches)",
            min_value=0.0,
            max_value=200.0,
            value=50.0,
            help="Penalty per reduced-size match. Higher values make full matches more strongly preferred."
        )

        return (
            optimal_solve,
            time_limit,
            max_attempts,
            score_threshold,
            max_match_variance,
            pair_slack_weight,
            pair_var_weight,
            ref_slack_weight,
            ref_var_weight,
            team_match_weight,
            rematch_delay_weight,
            bye_balance_weight,
            bye_spread_weight,
            reduced_match_balance_weight,
            reduced_match_weight,
        )


def collect_sidebar_state():
    with st.sidebar:
        st.header("Tournament Setup")

        num_teams = get_num_teams()
        min_teams_per_match, max_teams_per_match = get_teams_per_match()
        num_refs = get_num_referees()
        reduced_matches_per_round = compute_max_reduced_matches_per_round(
            min_teams_per_match,
            max_teams_per_match,
            num_refs,
            num_teams
        )
        enforce_pair_limits, min_pair_encounters, max_pair_encounters = get_pair_encounter_inputs()
        enforce_ref_limits, min_ref_encounters, max_ref_encounters = get_referee_encounter_inputs()
        num_rounds = get_num_rounds()

        show_pair_validation(
            enforce_pair_limits,
            min_pair_encounters,
            max_pair_encounters,
            num_teams,
            max_teams_per_match,
            num_refs,
            num_rounds
        )
        show_ref_validation(
            enforce_ref_limits,
            min_ref_encounters,
            max_ref_encounters,
            num_teams,
            max_teams_per_match,
            num_refs,
            num_rounds
        )

        show_suggestions(
            num_teams,
            max_teams_per_match,
            min_teams_per_match,
            num_refs,
            num_rounds,
            reduced_matches_per_round
        )

        (
            optimal_solve,
            time_limit,
            max_attempts,
            score_threshold,
            max_match_variance,
            pair_slack_weight,
            pair_var_weight,
            ref_slack_weight,
            ref_var_weight,
            team_match_weight,
            rematch_delay_weight,
            bye_balance_weight,
            bye_spread_weight,
            reduced_match_balance_weight,
            reduced_match_weight,
        ) = get_advanced_settings()

        hint_csv_file = st.file_uploader(
            "📥 Seed schedule from CSV (matrix)",
            type=["csv"],
            help=(
                "Upload the same matrix format shown in the UI: rows = Round, columns = Ref, "
                "cells contain team lists (e.g., 'T1, T2, T3' or 'T1, T2, T3 | T4, T5, T6')."
            )
        )

        with st.sidebar.expander("📚 Help & Documentation", expanded=False):
            st.markdown(
                """
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
        - **Max matches per round**: Limited by the number of available referees (refs can be idle)
        - **Pair encounter limits** (optional): When enabled, enforces min/max times each pair of teams can meet
        - **Referee encounter limits** (optional): When enabled, enforces min/max times each team sees each referee (exact min=max may require specific round counts)
        - **Match count variance** (optional): Limits how much match counts can differ between teams (0 = perfectly equal)
        - **Teams per match (min/max)**: If min < max, the solver may use reduced-size matches (min) alongside full matches (max)
        - **Max reduced matches per round** (optional): Upper limit on reduced-size matches per round
        
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
        - **Reduced match balance**: Keeps reduced-size matches evenly distributed across teams (prevents a few teams from getting most reduced matches)
        - **Reduced match penalty**: Prefers full matches over reduced-size matches when both are possible
        
        The solver combines all weighted factors into a single score, then searches for the schedule with the lowest score (best balance of all preferences).
        """
            )

    state = SidebarState(
        num_teams=num_teams,
        min_teams_per_match=min_teams_per_match,
        max_teams_per_match=max_teams_per_match,
        num_refs=num_refs,
        num_rounds=num_rounds,
        max_match_variance=max_match_variance,
        reduced_matches_per_round=reduced_matches_per_round,
        enforce_pair_limits=enforce_pair_limits,
        min_pair_encounters=min_pair_encounters,
        max_pair_encounters=max_pair_encounters,
        enforce_ref_limits=enforce_ref_limits,
        min_ref_encounters=min_ref_encounters,
        max_ref_encounters=max_ref_encounters,
        optimal_solve=optimal_solve,
        time_limit=time_limit,
        max_attempts=max_attempts,
        score_threshold=score_threshold,
        pair_slack_weight=pair_slack_weight,
        pair_var_weight=pair_var_weight,
        ref_slack_weight=ref_slack_weight,
        ref_var_weight=ref_var_weight,
        team_match_weight=team_match_weight,
        rematch_delay_weight=rematch_delay_weight,
        bye_balance_weight=bye_balance_weight,
        bye_spread_weight=bye_spread_weight,
        reduced_match_balance_weight=reduced_match_balance_weight,
        reduced_match_weight=reduced_match_weight,
    )

    return state, hint_csv_file
