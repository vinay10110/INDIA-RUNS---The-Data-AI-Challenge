#!/usr/bin/env python3
"""
STAGE 1: Data Quality (Hard Filters Only)

Purpose:
- Remove only genuinely corrupted / unusable candidate records.
- Do NOT eliminate candidates for low engagement, inactivity, or incomplete-but-usable profiles.
- Keep Stage 1 focused on hard data quality and structural sanity.

Hard filters applied:
1. Invalid candidate ID format
2. Missing required core sections:
   - profile
   - career_history
   - redrob_signals
3. Impossible years_of_experience
4. Corrupted salary range
5. Invalid recruiter response rate
6. Invalid role dates:
   - invalid date format
   - start_date > end_date
   - dates before 1980
   - future dates
"""

import json
import argparse
import re
from pathlib import Path
from typing import Dict, Any, Tuple
from datetime import datetime, date


class Stage1DataQuality:
    """Remove only genuinely corrupted candidate data."""

    MIN_VALID_YEAR = 1980

    def __init__(self):
        self.today = date.today()
        self.stats = {
            'total_input': 0,
            'eliminated_invalid_id': 0,
            'eliminated_missing_core_fields': 0,
            'eliminated_data_corruption': 0,
            'total_output': 0,
        }

    def validate_candidate_id(self, candidate_id: str) -> bool:
        """Check if candidate_id matches format CAND_XXXXXXX (7 digits)."""
        if not isinstance(candidate_id, str):
            return False
        pattern = r'^CAND_[0-9]{7}$'
        return bool(re.match(pattern, candidate_id.strip()))

    def parse_date(self, value: str):
        """
        Parse YYYY-MM-DD date safely.
        Returns date object if valid, else None.
        """
        if not value or not isinstance(value, str):
            return None
        try:
            return datetime.strptime(value.strip(), "%Y-%m-%d").date()
        except ValueError:
            return None

    def validate_role_dates(self, career_history) -> Tuple[bool, str]:
        """
        Validate role-level dates for structural sanity only.
        Checks:
        - valid date format
        - start_date <= end_date
        - no dates before MIN_VALID_YEAR
        - no future dates
        """
        for i, role in enumerate(career_history):
            start_raw = role.get("start_date")
            end_raw = role.get("end_date")

            # Parse if provided
            start_dt = self.parse_date(start_raw) if start_raw else None
            end_dt = self.parse_date(end_raw) if end_raw else None

            # Invalid format if raw exists but failed parsing
            if start_raw and start_dt is None:
                return False, f"Role {i}: invalid start_date format ({start_raw})"
            if end_raw and end_dt is None:
                return False, f"Role {i}: invalid end_date format ({end_raw})"

            # Pre-1980 sanity check
            if start_dt and start_dt.year < self.MIN_VALID_YEAR:
                return False, f"Role {i}: start_date before {self.MIN_VALID_YEAR}"
            if end_dt and end_dt.year < self.MIN_VALID_YEAR:
                return False, f"Role {i}: end_date before {self.MIN_VALID_YEAR}"

            # Future-date sanity check
            if start_dt and start_dt > self.today:
                return False, f"Role {i}: future start_date"
            if end_dt and end_dt > self.today:
                return False, f"Role {i}: future end_date"

            # Start after end
            if start_dt and end_dt and start_dt > end_dt:
                return False, f"Role {i}: start_date after end_date"

        return True, "Valid role dates"

    def filter_candidate(self, candidate: Dict[str, Any]) -> Tuple[bool, str]:
        """Apply hard data-quality filters only."""

        # --------------------------------------------------
        # Check 1: Valid candidate ID
        # --------------------------------------------------
        cid = candidate.get('candidate_id', '')
        if not self.validate_candidate_id(cid):
            self.stats['eliminated_invalid_id'] += 1
            return False, "Invalid candidate ID format"

        # --------------------------------------------------
        # Check 2: Required core sections must exist
        # --------------------------------------------------
        profile = candidate.get('profile')
        career_history = candidate.get('career_history')
        redrob_signals = candidate.get('redrob_signals')

        if (
            not isinstance(profile, dict) or len(profile) == 0 or
            not isinstance(career_history, list) or len(career_history) == 0 or
            not isinstance(redrob_signals, dict) or len(redrob_signals) == 0
        ):
            self.stats['eliminated_missing_core_fields'] += 1
            return False, "Missing required core fields (profile/career_history/redrob_signals)"

        # --------------------------------------------------
        # Check 3: Impossible years_of_experience
        # --------------------------------------------------
        yoe = profile.get('years_of_experience')
        if yoe is not None:
            if not isinstance(yoe, (int, float)) or yoe < 0 or yoe > 50:
                self.stats['eliminated_data_corruption'] += 1
                return False, f"Impossible years_of_experience: {yoe}"

        # --------------------------------------------------
        # Check 4: Corrupted salary range
        # --------------------------------------------------
        salary_range = redrob_signals.get('expected_salary_range_inr_lpa')
        if isinstance(salary_range, dict):
            sal_min = salary_range.get('min')
            sal_max = salary_range.get('max')

            # If both are present, validate ordering
            if sal_min is not None and sal_max is not None:
                if not isinstance(sal_min, (int, float)) or not isinstance(sal_max, (int, float)):
                    self.stats['eliminated_data_corruption'] += 1
                    return False, f"Invalid salary types: min={sal_min}, max={sal_max}"

                if sal_min < 0 or sal_max < 0:
                    self.stats['eliminated_data_corruption'] += 1
                    return False, f"Negative salary values: min={sal_min}, max={sal_max}"

                if sal_min > sal_max:
                    self.stats['eliminated_data_corruption'] += 1
                    return False, f"Corrupted salary range: min {sal_min} > max {sal_max}"

        # --------------------------------------------------
        # Check 5: Invalid recruiter response rate
        # --------------------------------------------------
        response_rate = redrob_signals.get('recruiter_response_rate')
        if response_rate is not None:
            if not isinstance(response_rate, (int, float)) or response_rate < 0 or response_rate > 1:
                self.stats['eliminated_data_corruption'] += 1
                return False, f"Invalid recruiter_response_rate: {response_rate}"

        # --------------------------------------------------
        # Check 6: Role date sanity
        # --------------------------------------------------
        dates_ok, reason = self.validate_role_dates(career_history)
        if not dates_ok:
            self.stats['eliminated_data_corruption'] += 1
            return False, reason

        # PASS: Keep for downstream stages
        return True, "Pass data quality"

    def process_jsonl(self, input_path: str, output_path: str) -> Dict[str, int]:
        """Load, filter, and save valid candidates to output JSONL."""
        input_file = Path(input_path)
        output_file = Path(output_path)

        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        self.stats['total_input'] = 0
        self.stats['total_output'] = 0

        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:

            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    candidate = json.loads(line)
                except json.JSONDecodeError:
                    # Skip malformed JSON rows entirely
                    continue

                self.stats['total_input'] += 1
                passes, _ = self.filter_candidate(candidate)

                if passes:
                    outfile.write(json.dumps(candidate, ensure_ascii=False) + '\n')
                    self.stats['total_output'] += 1

        return self.stats

    def print_report(self):
        """Print summary statistics."""
        print("\n" + "=" * 70)
        print("STAGE 1: DATA QUALITY (HARD FILTERS ONLY)")
        print("=" * 70)
        print(f"Total input:                          {self.stats['total_input']:>10,}")
        print(f"  - Invalid ID:                       {self.stats['eliminated_invalid_id']:>10,}")
        print(f"  - Missing core fields:              {self.stats['eliminated_missing_core_fields']:>10,}")
        print(f"  - Data corruption:                  {self.stats['eliminated_data_corruption']:>10,}")
        print(f"Total output:                         {self.stats['total_output']:>10,}")
        eliminated = self.stats['total_input'] - self.stats['total_output']
        rate = (eliminated / max(self.stats['total_input'], 1)) * 100
        print(f"Elimination rate:                     {rate:>9.1f}%")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Stage 1: Hard data quality filters only.'
    )
    parser.add_argument('--input', required=True, help='Input JSONL file')
    parser.add_argument('--output', required=True, help='Output JSONL file')
    args = parser.parse_args()

    stage = Stage1DataQuality()
    try:
        stage.process_jsonl(args.input, args.output)
        stage.print_report()
        print(f"\nOutput written to: {args.output}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    raise SystemExit(main())