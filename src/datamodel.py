from dataclasses import dataclass, field
from typing import List, Optional
import itertools


@dataclass
class Team:
    id: int
    name: str


@dataclass
class Referee:
    id: int
    name: str


@dataclass
class Match:
    id: int
    teams: List[Team]
    round: Optional[int] = None
    referee: Optional[Referee] = None


@dataclass
class DataModel:
    teams: List[Team] = field(default_factory=list)
    referees: List[Referee] = field(default_factory=list)
    rounds: int = 0
    matches: List[Match] = field(default_factory=list)

    def add_team(self, name: str) -> Team:
        tid = len(self.teams) + 1
        t = Team(id=tid, name=name)
        self.teams.append(t)
        return t

    def add_referee(self, name: str) -> Referee:
        rid = len(self.referees) + 1
        r = Referee(id=rid, name=name)
        self.referees.append(r)
        return r

    def set_rounds(self, num_rounds: int) -> None:
        self.rounds = int(num_rounds)

    def generate_team_matches(self, teams_per_match: int) -> None:
        """Generate all combinations of teams of size `teams_per_match` as matches.

        This is a simple combinatorial generator (used by the UI for quick setups).
        It clears existing matches and assigns incremental ids.
        """
        if teams_per_match < 2:
            raise ValueError("teams_per_match must be >= 2")

        self.matches.clear()
        mid = 1
        for combo in itertools.combinations(self.teams, teams_per_match):
            self.matches.append(Match(id=mid, teams=list(combo)))
            mid += 1

    def generate_mixed_team_matches(self, full_match_size: int, reduced_match_size: Optional[int] = None) -> None:
        """Generate combinations for full matches and optional reduced matches.

        full_match_size: standard teams-per-match size.
        reduced_match_size: optional smaller size (typically full_match_size - 1).
        """
        if full_match_size < 2:
            raise ValueError("full_match_size must be >= 2")
        if reduced_match_size is not None and reduced_match_size < 2:
            raise ValueError("reduced_match_size must be >= 2")
        if reduced_match_size is not None and reduced_match_size >= full_match_size:
            raise ValueError("reduced_match_size must be smaller than full_match_size")

        self.matches.clear()
        mid = 1

        for combo in itertools.combinations(self.teams, full_match_size):
            self.matches.append(Match(id=mid, teams=list(combo)))
            mid += 1

        if reduced_match_size is not None:
            for combo in itertools.combinations(self.teams, reduced_match_size):
                self.matches.append(Match(id=mid, teams=list(combo)))
                mid += 1

    def generate_round_robin_matches(self, teams_per_match: int, matches_per_round: int) -> None:
        """Generate balanced round-robin matches using a rotation schedule.

        Each round, `matches_per_round` non-overlapping matches of `teams_per_match` are created.
        Teams rotate positions each round to ensure balanced pair encounters.
        Some teams get a bye each round if they don't fit evenly.

        Args:
            teams_per_match: number of teams per match (typically 3)
            matches_per_round: number of concurrent matches per round (typically 4)
        """
        if teams_per_match < 2:
            raise ValueError("teams_per_match must be >= 2")
        if matches_per_round < 1:
            raise ValueError("matches_per_round must be >= 1")

        self.matches.clear()
        mid = 1
        n_teams = len(self.teams)
        teams_to_play_per_round = matches_per_round * teams_per_match

        if teams_to_play_per_round > n_teams:
            raise ValueError(
                f"matches_per_round ({matches_per_round}) × teams_per_match ({teams_per_match}) "
                f"= {teams_to_play_per_round} exceeds team count ({n_teams})"
            )

        # Circular rotation: rotate the team list each round to vary pairings
        team_list = list(self.teams)
        for round_num in range(self.rounds):
            # Select teams_to_play_per_round teams for this round
            selected_teams = team_list[:teams_to_play_per_round]

            # Partition into matches_per_round groups
            for match_idx in range(matches_per_round):
                start_idx = match_idx * teams_per_match
                end_idx = start_idx + teams_per_match
                match_teams = selected_teams[start_idx:end_idx]
                self.matches.append(Match(id=mid, teams=match_teams))
                mid += 1

            # Rotate team list for next round (simple circular shift)
            team_list = team_list[1:] + [team_list[0]]
