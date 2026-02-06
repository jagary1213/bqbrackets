from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class SidebarState:
    num_teams: int
    min_teams_per_match: int
    max_teams_per_match: int
    num_refs: int
    num_rounds: int
    max_match_variance: int
    reduced_matches_per_round: int
    enforce_pair_limits: bool
    min_pair_encounters: Optional[int]
    max_pair_encounters: Optional[int]
    enforce_ref_limits: bool
    min_ref_encounters: Optional[int]
    max_ref_encounters: Optional[int]
    optimal_solve: bool
    time_limit: Optional[int]
    max_attempts: int
    score_threshold: float
    pair_slack_weight: float
    pair_var_weight: float
    ref_slack_weight: float
    ref_var_weight: float
    team_match_weight: float
    rematch_delay_weight: float
    bye_balance_weight: float
    bye_spread_weight: float
    reduced_match_balance_weight: float
    reduced_match_weight: float


@dataclass
class ScoreBreakdown:
    total: float
    pair_slack: float
    pair_var: float
    ref_slack: float
    ref_var: float
    team_match_slack: float
    team_match_var: float
    rematch_delay: float
    reduced_match_balance: float
    bye_balance: float
    bye_spread: float


@dataclass
class HintParseResult:
    hints: List[Dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    schedule: Optional[Dict] = None
    counts: Optional[Dict[Tuple[str, str], int]] = None
