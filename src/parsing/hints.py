from typing import Optional
import re

import pandas as pd

from src.domain.models import HintParseResult


def _parse_round_number(value) -> Optional[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        return int(value)
    match = re.search(r"(\d+)", str(value))
    if match:
        return int(match.group(1))
    return None


def parse_hint_matrix_to_schedule(hint_csv_file, teams, refs, match_lookup, num_rounds) -> HintParseResult:
    warnings = []
    schedule = {r: [] for r in range(1, num_rounds + 1)}
    counts = {}
    seen = set()

    try:
        hint_csv_file.seek(0)
    except Exception:
        pass

    try:
        df = pd.read_csv(hint_csv_file)
    except Exception as exc:
        return HintParseResult(schedule=schedule, counts=counts, warnings=[f"Could not read CSV: {exc}"])

    if df.empty:
        return HintParseResult(schedule=schedule, counts=counts, warnings=["CSV is empty."])

    round_col = None
    for col in df.columns:
        col_str = str(col).strip().lower()
        if col_str in ("round", "rounds"):
            round_col = col
            break
    if round_col is None:
        for col in df.columns:
            if "round" in str(col).strip().lower():
                round_col = col
                break
    if round_col is None:
        return HintParseResult(schedule=schedule, counts=counts, warnings=["CSV is missing a 'Round' column."])

    for _, row in df.iterrows():
        round_num = _parse_round_number(row.get(round_col))
        if round_num is None:
            warnings.append("Skipping row with missing round value.")
            continue
        if round_num < 1 or round_num > num_rounds:
            warnings.append(f"Skipping Round {round_num}: outside 1..{num_rounds}.")
            continue

        for ref_name_raw, cell in row.items():
            if ref_name_raw == round_col:
                continue
            ref_name = str(ref_name_raw).strip()
            if ref_name not in refs:
                continue
            if pd.isna(cell):
                continue
            cell_str = str(cell).strip()
            if cell_str in ("", "—", "-", "–", "None", "nan"):
                continue

            for segment in cell_str.split("|"):
                seg = segment.strip()
                if not seg or seg in ("—", "-", "–"):
                    continue
                team_names = [s.strip() for s in seg.split(",") if s.strip()]
                if not team_names:
                    continue
                unknown = [t for t in team_names if t not in teams]
                if unknown:
                    warnings.append(
                        f"Round {round_num}, {ref_name}: unknown team(s) {', '.join(unknown)}."
                    )
                    continue
                team_ids = [int(t[1:]) if t.startswith("T") and t[1:].isdigit() else None for t in team_names]
                match_id = None
                if all(tid is not None for tid in team_ids):
                    key = frozenset(team_ids)
                    match_id = match_lookup.get(key)
                entry_key = (round_num, ref_name, ",".join(sorted(team_names)))
                if entry_key in seen:
                    continue
                seen.add(entry_key)
                schedule[round_num].append((match_id, ", ".join(team_names), ref_name))
                for team in team_names:
                    counts[(team, ref_name)] = counts.get((team, ref_name), 0) + 1

    return HintParseResult(schedule=schedule, counts=counts, warnings=warnings)


def parse_hint_matrix_csv(hint_csv_file, team_name_to_id, ref_name_to_id, match_lookup, num_rounds) -> HintParseResult:
    teams = list(team_name_to_id.keys())
    refs = list(ref_name_to_id.keys())
    base_result = parse_hint_matrix_to_schedule(hint_csv_file, teams, refs, match_lookup, num_rounds)

    hints = []
    warnings = list(base_result.warnings)
    if base_result.schedule:
        for round_num, matches in base_result.schedule.items():
            for match_id, team_names, ref_name in matches:
                if match_id is None:
                    warnings.append(
                        f"Round {round_num}, {ref_name}: could not map teams {team_names} to a known match."
                    )
                    continue
                ref_id = ref_name_to_id.get(ref_name)
                hints.append({
                    "match_id": match_id,
                    "round": round_num,
                    "ref_id": ref_id,
                })

    return HintParseResult(
        hints=hints,
        warnings=warnings,
        schedule=base_result.schedule,
        counts=base_result.counts,
    )
