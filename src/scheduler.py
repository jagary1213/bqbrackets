"""Scheduler scaffolding using OR-Tools CP-SAT.

This module imports the CP-SAT solver and provides starter functions
to build the round-robin model. Phase 2 will add variables and constraints.
"""
from ortools.sat.python import cp_model
from typing import Dict, Tuple, List, Optional


def create_model():
    """Create and return a CP-SAT model and solver instance."""
    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    return model, solver


def build_round_robin_model(data_model, num_rounds: int, matches_per_round: int = None,
                            pair_slack_weight: float = 20.0, pair_var_weight: float = 5.0,
                            ref_slack_weight: float = 10.0, ref_var_weight: float = 5.0,
                            team_match_weight: float = 300.0,
                            min_pair_encounters: int = None, max_pair_encounters: int = None,
                            max_match_variance: int = 0,
                            reduced_match_size: int = None,
                            max_reduced_matches_per_round: int = 0,
                            reduced_match_weight: float = 50.0,
                            reduced_match_balance_weight: float = 50.0,
                            min_ref_encounters: int = None,
                            max_ref_encounters: int = None,
                            hint_assignments: Optional[List[Dict]] = None) -> Tuple[cp_model.CpModel, cp_model.CpSolver, Dict]:
    """Build a round-robin scheduling model.

    Variables:
      x[(match_id, r)] = 1 if the unordered pair `match_id` is scheduled in round `r`.

    Constraints:
      - Every unordered pair is played at least once and at most twice across all rounds.
      - A team cannot play more than one match in the same round.

    Returns model, solver and the variable dict `x`.
    """
    model, solver = create_model()

    matches = list(data_model.matches)
    rounds = list(range(1, int(num_rounds) + 1))

    # Boolean variables: x[(match_id, r)] in {0,1}
    x = {}
    for m in matches:
        for r in rounds:
            x[(m.id, r)] = model.NewBoolVar(f"match_{m.id}_r{r}")

    # Round-robin hints (warm-start): only when reduced matches are disabled and no external hints
    if reduced_match_size is None and matches_per_round is not None and matches_per_round > 0 and not hint_assignments:
        teams_per_match = None
        if matches:
            teams_per_match = len(matches[0].teams)
        if teams_per_match:
            # Build lookup from team-id set -> match id (full-size matches only)
            match_lookup = {}
            for m in matches:
                if len(m.teams) == teams_per_match:
                    team_ids = frozenset(t.id for t in m.teams)
                    match_lookup[team_ids] = m.id

            team_list = [t.id for t in data_model.teams]
            teams_to_play_per_round = matches_per_round * teams_per_match
            if teams_to_play_per_round <= len(team_list):
                for round_num in rounds:
                    selected = team_list[:teams_to_play_per_round]
                    # Partition into matches_per_round groups of size teams_per_match
                    for match_idx in range(matches_per_round):
                        start = match_idx * teams_per_match
                        end = start + teams_per_match
                        group = selected[start:end]
                        if len(group) == teams_per_match:
                            key = frozenset(group)
                            match_id = match_lookup.get(key)
                            if match_id is not None:
                                model.AddHint(x[(match_id, round_num)], 1)

                    # Rotate team list for next round
                    team_list = team_list[1:] + [team_list[0]]

    # Replace 1-2 per combo with pair-encounter balancing
    # Matches are optional; objective will schedule them to balance pair encounters
    # Count how many times each pair of teams face each other across all matches
    pair_encounters = {}
    for t1 in data_model.teams:
        for t2 in data_model.teams:
            if t1.id < t2.id:  # unordered pair
                pair_encounters[(t1.id, t2.id)] = model.NewIntVar(
                    0, len(rounds) * len(matches), f"pair_{t1.id}_{t2.id}"
                )
                # count: how many times does this pair appear in scheduled matches?
                model.Add(pair_encounters[(t1.id, t2.id)] == sum(
                    x[(m.id, r)]
                    for m in matches
                    for r in rounds
                    if any(ta.id == t1.id for ta in m.teams) and any(tb.id == t2.id for tb in m.teams)
                ))
                
                # Hard constraints for pair encounters (min/max bounds)
                if min_pair_encounters is not None:
                    model.Add(pair_encounters[(t1.id, t2.id)] >= min_pair_encounters)
                if max_pair_encounters is not None:
                    model.Add(pair_encounters[(t1.id, t2.id)] <= max_pair_encounters)

    # Team cannot play two matches in same round (byes allowed)
    for team in data_model.teams:
        for r in rounds:
            involved = []
            for m in matches:
                # match involves team if any team in m.teams has same id
                if any(t.id == team.id for t in m.teams):
                    involved.append(x[(m.id, r)])
            if involved:
                model.Add(sum(involved) <= 1)
    
    # Matches-per-round handling:
    # - If reduced matches are disabled, enforce exact matches_per_round (hard constraint)
    # - If reduced matches are enabled, allow shortfall with a penalty (soft preference)
    shortfalls = []
    if matches_per_round is not None and matches_per_round > 0:
        for r in rounds:
            matches_in_round = sum(x[(m.id, r)] for m in matches)
            if reduced_match_size is None:
                model.Add(matches_in_round == matches_per_round)
            else:
                # Require at least one match per round to avoid empty schedules
                model.Add(matches_in_round >= 1)
                shortfall = model.NewIntVar(0, matches_per_round, f"shortfall_r{r}")
                model.Add(shortfall == matches_per_round - matches_in_round)
                shortfalls.append(shortfall)

    # Optional reduced matches per round (one fewer team than full match)
    reduced_match_ids = []
    if reduced_match_size is not None and reduced_match_size >= 2:
        reduced_match_ids = [m.id for m in matches if len(m.teams) == reduced_match_size]
        if max_reduced_matches_per_round is not None and max_reduced_matches_per_round > 0:
            for r in rounds:
                model.Add(sum(x[(m_id, r)] for m_id in reduced_match_ids) <= max_reduced_matches_per_round)

    # Referee assignment variables and balancing objective (Phase 3)
    num_refs = max(1, len(data_model.referees))

    # y[(match_id, r, ref_id)] == 1 if match m scheduled in round r is assigned to referee ref_id
    y = {}
    for m in matches:
        for r in rounds:
            for ref in data_model.referees:
                y[(m.id, r, ref.id)] = model.NewBoolVar(f"y_m{m.id}_r{r}_ref{ref.id}")

    # Apply external hint assignments (warm-start)
    if hint_assignments:
        for hint in hint_assignments:
            match_id = hint.get("match_id")
            round_num = hint.get("round")
            ref_id = hint.get("ref_id")
            if (match_id, round_num) in x:
                model.AddHint(x[(match_id, round_num)], 1)
            if ref_id is not None and (match_id, round_num, ref_id) in y:
                model.AddHint(y[(match_id, round_num, ref_id)], 1)

    # If a match is assigned to a referee in a round then the match must be scheduled in that round
    for m in matches:
        for r in rounds:
            model.Add(sum(y[(m.id, r, ref.id)] for ref in data_model.referees) == x[(m.id, r)])

    # Limit concurrent matches per round to available referees
    for r in rounds:
        model.Add(sum(x[(m.id, r)] for m in matches) <= num_refs)
    
    # CRITICAL: Each referee can only officiate ONE match per round
    for ref in data_model.referees:
        for r in rounds:
            model.Add(sum(y[(m.id, r, ref.id)] for m in matches) <= 1)

    # Count how many times each team saw each referee
    counts = {}
    for team in data_model.teams:
        for ref in data_model.referees:
            counts[(team.id, ref.id)] = model.NewIntVar(0, num_rounds, f"count_t{team.id}_ref{ref.id}")
            # define the sum: count of matches where this team participates and referee ref is assigned
            model.Add(counts[(team.id, ref.id)] == sum(
                y[(m.id, r, ref.id)]
                for m in matches
                for r in rounds
                if any(t.id == team.id for t in m.teams)
            ))

            # Hard constraints for referee encounters (min/max bounds)
            if min_ref_encounters is not None:
                model.Add(counts[(team.id, ref.id)] >= min_ref_encounters)
            if max_ref_encounters is not None:
                model.Add(counts[(team.id, ref.id)] <= max_ref_encounters)

    # For each team, define max and min count across referees and minimize their difference (soft balance)
    team_slacks = []
    for team in data_model.teams:
        max_var = model.NewIntVar(0, num_rounds, f"max_t{team.id}")
        min_var = model.NewIntVar(0, num_rounds, f"min_t{team.id}")
        for ref in data_model.referees:
            model.Add(max_var >= counts[(team.id, ref.id)])
            model.Add(min_var <= counts[(team.id, ref.id)])
        slack = model.NewIntVar(0, num_rounds, f"slack_t{team.id}")
        model.Add(slack == max_var - min_var)
        team_slacks.append(slack)
    
    # Calculate referee variance: deviation of all (team, ref) counts from mean
    # We'll use sum of absolute deviations as a proxy for variance (easier in CP-SAT)
    all_ref_counts = [counts[(t.id, r.id)] for t in data_model.teams for r in data_model.referees]
    if all_ref_counts:
        max_ref_count = model.NewIntVar(0, num_rounds, "max_ref_count")
        min_ref_count = model.NewIntVar(0, num_rounds, "min_ref_count")
        for count_var in all_ref_counts:
            model.Add(max_ref_count >= count_var)
            model.Add(min_ref_count <= count_var)
        ref_variance_proxy = model.NewIntVar(0, num_rounds, "ref_variance_proxy")
        model.Add(ref_variance_proxy == max_ref_count - min_ref_count)
    else:
        ref_variance_proxy = 0

    # Soft constraint: balance pair encounters across all rounds
    all_pair_counts = [pair_encounters[(t1.id, t2.id)]
                       for t1 in data_model.teams
                       for t2 in data_model.teams
                       if t1.id < t2.id]
    if all_pair_counts:
        max_pair = model.NewIntVar(0, len(rounds) * len(matches), "max_pair_count")
        min_pair = model.NewIntVar(0, len(rounds) * len(matches), "min_pair_count")
        for count_var in all_pair_counts:
            model.Add(max_pair >= count_var)
            model.Add(min_pair <= count_var)
        pair_slack = model.NewIntVar(0, len(rounds) * len(matches), "pair_slack")
        model.Add(pair_slack == max_pair - min_pair)
    else:
        pair_slack = 0
    
    # Team match balance: count matches per team and minimize slack
    team_match_counts = {}
    for team in data_model.teams:
        team_match_counts[team.id] = model.NewIntVar(0, len(rounds), f"team_match_count_{team.id}")
        model.Add(team_match_counts[team.id] == sum(
            x[(m.id, r)]
            for m in matches
            for r in rounds
            if any(t.id == team.id for t in m.teams)
        ))
    
    # Min-max slack for team match counts
    max_team_matches = model.NewIntVar(0, len(rounds), "max_team_matches")
    min_team_matches = model.NewIntVar(0, len(rounds), "min_team_matches")
    for count_var in team_match_counts.values():
        model.Add(max_team_matches >= count_var)
        model.Add(min_team_matches <= count_var)
    team_match_slack = model.NewIntVar(0, len(rounds), "team_match_slack")
    model.Add(team_match_slack == max_team_matches - min_team_matches)
    
    # Hard constraint: enforce max match variance between teams
    model.Add(team_match_slack <= max_match_variance)

    # Reduced match balance: minimize variance of reduced-size matches per team
    reduced_match_balance_slack = 0
    if reduced_match_ids:
        reduced_match_team_ids = {
            m.id: [t.id for t in m.teams]
            for m in matches
            if m.id in reduced_match_ids
        }
        reduced_match_counts = {}
        for team in data_model.teams:
            reduced_match_counts[team.id] = model.NewIntVar(0, len(rounds), f"reduced_match_count_{team.id}")
            model.Add(reduced_match_counts[team.id] == sum(
                x[(m_id, r)]
                for m_id, team_ids in reduced_match_team_ids.items()
                for r in rounds
                if team.id in team_ids
            ))

        max_reduced = model.NewIntVar(0, len(rounds), "max_reduced_match")
        min_reduced = model.NewIntVar(0, len(rounds), "min_reduced_match")
        for count_var in reduced_match_counts.values():
            model.Add(max_reduced >= count_var)
            model.Add(min_reduced <= count_var)
        reduced_match_balance_slack = model.NewIntVar(0, len(rounds), "reduced_match_balance_slack")
        model.Add(reduced_match_balance_slack == max_reduced - min_reduced)
    
    # Combine objectives using provided weights: maximize scheduling, then balance using weights
    # Strongly prioritize: get matches_per_round matches per round
    total_matches_scheduled = sum(x[(m.id, r)] for m in matches for r in rounds)
    
    # Build objective with all balance terms
    ref_var_term = ref_variance_proxy if not isinstance(ref_variance_proxy, int) else 0
    shortfall_term = sum(shortfalls) if shortfalls else 0
    reduced_match_term = sum(
        x[(m_id, r)] for m_id in reduced_match_ids for r in rounds
    ) if reduced_match_ids else 0
    
    if isinstance(pair_slack, int) and pair_slack == 0:
        # No pair constraint; use ref and team weights
        model.Minimize(sum(team_slacks) * ref_slack_weight + ref_var_term * ref_var_weight + team_match_slack * team_match_weight + reduced_match_balance_slack * reduced_match_balance_weight + shortfall_term * 200 + reduced_match_term * reduced_match_weight - total_matches_scheduled * 1000)
    else:
        # Use all weights to balance the objective
        model.Minimize(pair_slack * pair_slack_weight + pair_var_weight * 10 + sum(team_slacks) * ref_slack_weight + ref_var_term * ref_var_weight + team_match_slack * team_match_weight + reduced_match_balance_slack * reduced_match_balance_weight + shortfall_term * 200 + reduced_match_term * reduced_match_weight - total_matches_scheduled * 1000)

    # Return model, solver, and both variable dicts
    vars = {"x": x, "y": y, "counts": counts}
    return model, solver, vars


def solve_and_extract(model: cp_model.CpModel, solver: cp_model.CpSolver, xvars: Dict, data_model, num_rounds: int, time_limit_seconds: int = 10, diag_context: Dict = None):
    """Solve the model and extract a simple schedule mapping rounds -> match occurrences.

    Returns (schedule, counts_dict) or (None, {}) if infeasible.
    """
    if time_limit_seconds is not None:
        solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = 8

    active_solver = solver
    result = active_solver.Solve(model)
    if result not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Fallback: try a fast feasible pass if we hit the time limit
        if result == cp_model.UNKNOWN and time_limit_seconds is not None:
            fallback_solver = cp_model.CpSolver()
            fallback_solver.parameters.max_time_in_seconds = max(1, min(5, time_limit_seconds))
            fallback_solver.parameters.stop_after_first_solution = True
            fallback_solver.parameters.num_search_workers = 8
            fb_result = fallback_solver.Solve(model)
            if fb_result in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                active_solver = fallback_solver
                result = fb_result

    if result not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Build diagnostics to help explain infeasibility
        diagnostics = {}
        n_teams = len(data_model.teams)
        matches_list = list(data_model.matches)
        total_matches = len(matches_list)
        num_refs = max(1, len(data_model.referees))
        k = None
        if matches_list:
            k = len(matches_list[0].teams)

        diagnostics["n_teams"] = n_teams
        diagnostics["teams"] = [t.name for t in data_model.teams]
        diagnostics["teams_per_match"] = k
        diagnostics["num_refs"] = num_refs
        diagnostics["rounds_provided"] = int(num_rounds)
        if diag_context:
            diagnostics["hard_constraints"] = diag_context

        # Stronger feasibility checks based on hard constraints
        reasons = []
        max_team_slots_per_round = num_refs * k if k else 0
        max_total_team_slots = int(num_rounds) * max_team_slots_per_round
        max_matches_per_team = int(num_rounds)  # one match per team per round

        if diag_context:
            min_pair = diag_context.get("min_pair_encounters")
            max_pair = diag_context.get("max_pair_encounters")
            min_ref = diag_context.get("min_ref_encounters")
            max_ref = diag_context.get("max_ref_encounters")
            max_var = diag_context.get("max_match_variance")

            min_matches_from_ref = min_ref * num_refs if min_ref is not None else 0
            max_matches_from_ref = max_ref * num_refs if max_ref is not None else max_matches_per_team

            min_matches_from_pair = 0
            max_matches_from_pair = max_matches_per_team
            if k and min_pair is not None:
                import math
                min_matches_from_pair = math.ceil(((n_teams - 1) * min_pair) / max(1, (k - 1)))
            if k and max_pair is not None:
                import math
                max_matches_from_pair = math.floor(((n_teams - 1) * max_pair) / max(1, (k - 1)))

            # Basic per-team match bounds
            lower_bound = max(min_matches_from_ref, min_matches_from_pair, 0)
            upper_bound = min(max_matches_from_ref, max_matches_from_pair, max_matches_per_team)

            if lower_bound > upper_bound:
                reasons.append(
                    f"Per-team match bounds conflict: min {lower_bound} > max {upper_bound}. "
                    f"(from ref/pair limits and rounds)"
                )

            # Ref encounter feasibility
            if min_ref is not None and min_ref > 0:
                if min_matches_from_ref > max_matches_per_team:
                    reasons.append(
                        f"Ref encounters require at least {min_matches_from_ref} matches per team, but only {max_matches_per_team} rounds are available."
                    )
                if (min_matches_from_ref * n_teams) > max_total_team_slots:
                    reasons.append(
                        f"Ref encounters require at least {min_matches_from_ref * n_teams} team-slots, but at most {max_total_team_slots} are available."
                    )

            if max_ref is not None and max_ref * num_refs == 0 and (min_ref or 0) > 0:
                reasons.append("Ref encounters set to 0 while min_ref_encounters > 0.")

            # Pair encounter feasibility (upper bound using full matches)
            if min_pair is not None and min_pair > 0 and k:
                num_pairs = n_teams * (n_teams - 1) // 2
                max_pair_encounters_total = int(num_rounds) * num_refs * (k * (k - 1) // 2)
                required_pair_encounters = num_pairs * min_pair
                if required_pair_encounters > max_pair_encounters_total:
                    reasons.append(
                        f"Pair encounters require {required_pair_encounters} total pairings, but at most {max_pair_encounters_total} are possible in {num_rounds} rounds."
                    )

            # Exact match counts when variance is zero
            if max_var == 0 and lower_bound <= upper_bound:
                # Must pick a single matches-per-team value m
                feasible_m = [m for m in range(lower_bound, upper_bound + 1) if (m * n_teams) % max(1, k) == 0]
                if not feasible_m:
                    reasons.append(
                        f"With zero match variance, no integer matches-per-team value satisfies divisibility by teams-per-match (k={k})."
                    )
                else:
                    # Capacity check for exact m
                    m = feasible_m[0]
                    total_matches_needed = (m * n_teams) / max(1, k)
                    max_matches_available = num_refs * int(num_rounds)
                    if total_matches_needed > max_matches_available:
                        reasons.append(
                            f"Exact match count implies {total_matches_needed:.0f} matches total, but only {max_matches_available} can fit with {num_refs} refs over {num_rounds} rounds."
                        )

        if result == cp_model.UNKNOWN:
            reasons = [
                "No solution found within the current time limit. Try increasing the time limit, enabling Optimal solve, or relaxing constraints."
            ]
        elif not reasons:
            reasons.append(
                "No simple capacity violation detected. Infeasibility likely comes from the combination of hard constraints (pair/ref limits, match variance, reduced match limits)."
            )

        diagnostics["reasons"] = reasons

        return None, {"_diag": diagnostics}

    schedule = {r: [] for r in range(1, int(num_rounds) + 1)}
    # If y variables exist, extract referee assignments
    yvars = xvars.get("y") if isinstance(xvars, dict) else None
    if yvars:
        # Build schedule entries with referee
        for m in data_model.matches:
                for r in range(1, int(num_rounds) + 1):
                    for ref in data_model.referees:
                        if active_solver.Value(yvars[(m.id, r, ref.id)]) == 1:
                            team_names = ", ".join([t.name for t in m.teams])
                            schedule[r].append((m.id, team_names, ref.name))
    else:
        for m in data_model.matches:
            for r in range(1, int(num_rounds) + 1):
                if active_solver.Value(xvars[(m.id, r)]) == 1:
                    team_names = ", ".join([t.name for t in m.teams])
                    schedule[r].append((m.id, team_names, None))

    # Build counts mapping (team_name, ref_name) -> count
    counts_out = {}
    counts_vars = xvars.get("counts") if isinstance(xvars, dict) else None
    if counts_vars:
        for team in data_model.teams:
            for ref in data_model.referees:
                counts_out[(team.name, ref.name)] = active_solver.Value(counts_vars[(team.id, ref.id)])

    return schedule, counts_out

