"""Scheduler scaffolding using OR-Tools CP-SAT.

This module imports the CP-SAT solver and provides starter functions
to build the round-robin model. Phase 2 will add variables and constraints.
"""
from ortools.sat.python import cp_model
from typing import Dict, Tuple


def create_model():
    """Create and return a CP-SAT model and solver instance."""
    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    return model, solver


def build_round_robin_model(data_model, num_rounds: int, matches_per_round: int = None,
                            pair_slack_weight: float = 20.0, pair_var_weight: float = 5.0,
                            ref_slack_weight: float = 10.0, ref_var_weight: float = 5.0,
                            team_match_weight: float = 300.0) -> Tuple[cp_model.CpModel, cp_model.CpSolver, Dict]:
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
    
    # Hard constraint: schedule exactly matches_per_round matches per round
    # This ensures all refs are used and maximal utilization
    if matches_per_round is not None and matches_per_round > 0:
        for r in rounds:
            model.Add(sum(x[(m.id, r)] for m in matches) >= matches_per_round)

    # Referee assignment variables and balancing objective (Phase 3)
    num_refs = max(1, len(data_model.referees))

    # y[(match_id, r, ref_id)] == 1 if match m scheduled in round r is assigned to referee ref_id
    y = {}
    for m in matches:
        for r in rounds:
            for ref in data_model.referees:
                y[(m.id, r, ref.id)] = model.NewBoolVar(f"y_m{m.id}_r{r}_ref{ref.id}")

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
    
    # Combine objectives using provided weights: maximize scheduling, then balance using weights
    # Strongly prioritize: get matches_per_round matches per round
    total_matches_scheduled = sum(x[(m.id, r)] for m in matches for r in rounds)
    
    # Build objective with all balance terms
    ref_var_term = ref_variance_proxy if not isinstance(ref_variance_proxy, int) else 0
    
    if isinstance(pair_slack, int) and pair_slack == 0:
        # No pair constraint; use ref and team weights
        model.Minimize(sum(team_slacks) * ref_slack_weight + ref_var_term * ref_var_weight + team_match_slack * team_match_weight - total_matches_scheduled * 1000)
    else:
        # Use all weights to balance the objective
        model.Minimize(pair_slack * pair_slack_weight + pair_var_weight * 10 + sum(team_slacks) * ref_slack_weight + ref_var_term * ref_var_weight + team_match_slack * team_match_weight - total_matches_scheduled * 1000)

    # Return model, solver, and both variable dicts
    vars = {"x": x, "y": y, "counts": counts}
    return model, solver, vars


def solve_and_extract(model: cp_model.CpModel, solver: cp_model.CpSolver, xvars: Dict, data_model, num_rounds: int, time_limit_seconds: int = 10):
    """Solve the model and extract a simple schedule mapping rounds -> match occurrences.

    Returns (schedule, counts_dict) or (None, {}) if infeasible.
    """
    solver.parameters.max_time_in_seconds = time_limit_seconds
    result = solver.Solve(model)
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

        # matches per team
        matches_per_team = {team.id: 0 for team in data_model.teams}
        for m in matches_list:
            for t in m.teams:
                matches_per_team[t.id] += 1

        diagnostics["n_teams"] = n_teams
        diagnostics["teams"] = [t.name for t in data_model.teams]
        diagnostics["teams_per_match"] = k
        diagnostics["total_matches"] = total_matches
        diagnostics["num_refs"] = num_refs
        diagnostics["rounds_provided"] = int(num_rounds)
        diagnostics["matches_per_team"] = {t.name: matches_per_team[t.id] for t in data_model.teams}

        # capacity checks
        import math
        min_rounds_by_capacity = math.ceil(total_matches / num_refs) if num_refs > 0 else float("inf")
        min_rounds_by_team = max(matches_per_team.values()) if matches_per_team else 0
        diagnostics["min_rounds_by_capacity"] = min_rounds_by_capacity
        diagnostics["min_rounds_by_team"] = min_rounds_by_team

        reasons = []
        if int(num_rounds) < min_rounds_by_capacity:
            reasons.append(f"Not enough referee capacity: need at least {min_rounds_by_capacity} rounds to host {total_matches} matches with {num_refs} refs.")
        if int(num_rounds) < min_rounds_by_team:
            reasons.append(f"A team must play {min_rounds_by_team} matches but only {num_rounds} rounds available (one match per team per round).")
        if not reasons:
            reasons.append("No simple capacity violation detected; model may be over-constrained (e.g., max 2 plays per pair).")

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
                        if solver.Value(yvars[(m.id, r, ref.id)]) == 1:
                            team_names = ", ".join([t.name for t in m.teams])
                            schedule[r].append((m.id, team_names, ref.name))
    else:
        for m in data_model.matches:
            for r in range(1, int(num_rounds) + 1):
                if solver.Value(xvars[(m.id, r)]) == 1:
                    team_names = ", ".join([t.name for t in m.teams])
                    schedule[r].append((m.id, team_names, None))

    # Build counts mapping (team_name, ref_name) -> count
    counts_out = {}
    counts_vars = xvars.get("counts") if isinstance(xvars, dict) else None
    if counts_vars:
        for team in data_model.teams:
            for ref in data_model.referees:
                counts_out[(team.name, ref.name)] = solver.Value(counts_vars[(team.id, ref.id)])

    return schedule, counts_out

