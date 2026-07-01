#!/usr/bin/env python3
"""
REFINED Stage 5: Behavioral Signals (Soft Penalties)

Behavioral signals should primarily DOWN-WEIGHT candidates, not eliminate them.
This stage computes behavioral penalties and attaches interpretable features
for downstream scoring (Stage 6+).

Behavioral signals handled:
- Inactivity
- Profile completeness
- Recruiter response rate
- Interview completion rate

Only truly extreme inactivity + disengagement cases are eliminated.
Everything else passes through with behavioral penalties attached.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional


class Stage5BehavioralPenalties:
    """Apply behavioral signal penalties and attach interpretable features."""

    def __init__(self, reference_date=None):
        self.reference_date = reference_date or datetime.now()
        self.stats = {
            'total_input': 0,
            'eliminated_extreme_inactive': 0,
            'total_output': 0,
        }

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Safely coerce numeric values."""
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _safe_bool(self, value: Any, default: bool = False) -> bool:
        """Safely coerce booleans."""
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "y"}
        return bool(value)

    def get_days_inactive(self, candidate: Dict[str, Any]) -> Optional[int]:
        """Return days inactive if available, else None."""
        signals = candidate.get('redrob_signals', {})
        last_active_str = signals.get('last_active_date')

        if not last_active_str:
            return None

        try:
            last_active = datetime.strptime(last_active_str, '%Y-%m-%d')
            return max((self.reference_date - last_active).days, 0)
        except (ValueError, TypeError):
            return None

    def calculate_inactivity_penalty(self, candidate: Dict[str, Any]) -> Tuple[float, bool, Optional[int]]:
        """
        Calculate inactivity penalty.

        Returns:
            (penalty, should_eliminate, days_inactive)

        Eliminate only if extremely inactive AND disengaged.
        Otherwise return a soft penalty.
        """
        signals = candidate.get('redrob_signals', {})
        days_inactive = self.get_days_inactive(candidate)

        # Missing or invalid activity date = mild uncertainty penalty
        if days_inactive is None:
            return -10.0, False, None

        open_to_work = self._safe_bool(signals.get('open_to_work_flag', False))
        profile_views = self._safe_float(signals.get('profile_views_received_30d', 0), 0.0)

        # Extreme inactivity + no visible engagement
        if days_inactive > 180 and not open_to_work and profile_views == 0:
            return 0.0, True, days_inactive

        # Otherwise scale penalty by inactivity
        if days_inactive < 30:
            return 0.0, False, days_inactive
        elif days_inactive < 90:
            return -5.0, False, days_inactive
        elif days_inactive < 180:
            return -15.0, False, days_inactive
        else:
            return -25.0, False, days_inactive

    def calculate_completeness_penalty(self, candidate: Dict[str, Any]) -> Tuple[float, float]:
        """
        Penalty for low profile completeness.

        Intended rule:
        -5 points per 10% below 60%

        Examples:
        60 -> 0
        50 -> -5
        40 -> -10
        30 -> -15
        20 -> -20
        """
        signals = candidate.get('redrob_signals', {})
        completeness = self._safe_float(signals.get('profile_completeness_score', 100), 100.0)

        # Clamp to a sensible range
        completeness = max(0.0, min(completeness, 100.0))

        if completeness >= 60:
            return 0.0, completeness

        penalty = -0.5 * (60 - completeness)  # -5 per 10 points below 60
        penalty = max(penalty, -30.0)         # cap penalty so it doesn't dominate
        return penalty, completeness

    def calculate_engagement_penalty(self, candidate: Dict[str, Any]) -> Tuple[float, float, float]:
        """
        Penalty for low engagement signals.

        IMPORTANT:
        accumulate both response-rate and interview-rate penalties.
        """
        signals = candidate.get('redrob_signals', {})

        response_rate = self._safe_float(signals.get('recruiter_response_rate', 0.5), 0.5)
        interview_rate = self._safe_float(signals.get('interview_completion_rate', 0.5), 0.5)

        # Clamp to [0, 1] if upstream data is messy
        response_rate = max(0.0, min(response_rate, 1.0))
        interview_rate = max(0.0, min(interview_rate, 1.0))

        penalty = 0.0

        # Recruiter response penalty
        if response_rate < 0.2:
            penalty -= 10.0
        elif response_rate < 0.4:
            penalty -= 5.0

        # Interview completion penalty
        if interview_rate < 0.3:
            penalty -= 8.0
        elif interview_rate < 0.5:
            penalty -= 3.0

        return penalty, response_rate, interview_rate

    def calculate_behavioral_penalty(self, candidate: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """
        Calculate total behavioral penalty and return interpretable features.

        Returns:
            (behavioral_features, should_eliminate)
        """
        inactivity_penalty, extreme_inactive, days_inactive = self.calculate_inactivity_penalty(candidate)

        # If extreme inactivity case, eliminate directly
        if extreme_inactive:
            features = {
                'inactivity_penalty': 0.0,
                'completeness_penalty': 0.0,
                'engagement_penalty': 0.0,
                'total_penalty': 0.0,
                'days_inactive': days_inactive,
                'profile_completeness_score': None,
                'response_rate': None,
                'interview_completion_rate': None,
                'elimination_reason': 'Extreme inactivity + no engagement'
            }
            return features, True

        completeness_penalty, completeness = self.calculate_completeness_penalty(candidate)
        engagement_penalty, response_rate, interview_rate = self.calculate_engagement_penalty(candidate)

        total_penalty = inactivity_penalty + completeness_penalty + engagement_penalty

        features = {
            'inactivity_penalty': inactivity_penalty,
            'completeness_penalty': completeness_penalty,
            'engagement_penalty': engagement_penalty,
            'total_penalty': total_penalty,
            'days_inactive': days_inactive,
            'profile_completeness_score': completeness,
            'response_rate': response_rate,
            'interview_completion_rate': interview_rate,
            'elimination_reason': None
        }

        # Optional guardrail: only eliminate if behavioral signals are catastrophically bad.
        # This should be rare because Stage 5 is mostly a soft-penalty stage.
        should_eliminate = total_penalty < -45

        if should_eliminate:
            features['elimination_reason'] = 'Catastrophic combined behavioral penalty'

        return features, should_eliminate

    def filter_candidate(self, candidate: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Apply behavioral checks.

        Returns:
            (pass, reason, behavioral_features)
        """
        behavioral_features, should_eliminate = self.calculate_behavioral_penalty(candidate)

        if should_eliminate:
            reason = behavioral_features.get('elimination_reason') or 'Behavioral elimination'
            return False, reason, behavioral_features

        return True, f"Behavioral penalty: {behavioral_features['total_penalty']:.1f}", behavioral_features

    def process_jsonl(self, input_path: str, output_path: str) -> Dict[str, int]:
        """Load, apply penalties, enrich candidate records, and save."""
        input_file = Path(input_path)
        output_file = Path(output_path)

        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        self.stats['total_input'] = 0
        self.stats['total_output'] = 0

        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:

            for line in infile:
                line = line.strip()
                if not line:
                    continue

                try:
                    candidate = json.loads(line)
                except json.JSONDecodeError:
                    continue

                self.stats['total_input'] += 1
                passes, _, behavioral_features = self.filter_candidate(candidate)

                if passes:
                    # Backward-compatible single score
                    candidate['_behavioral_penalty'] = behavioral_features['total_penalty']

                    # Rich interpretable feature block for downstream scoring / debugging
                    candidate['_behavioral_features'] = behavioral_features

                    outfile.write(json.dumps(candidate) + '\n')
                    self.stats['total_output'] += 1
                else:
                    self.stats['eliminated_extreme_inactive'] += 1

        return self.stats

    def print_report(self):
        """Print statistics."""
        print("\n" + "=" * 70)
        print("STAGE 5: BEHAVIORAL SIGNAL PENALTIES")
        print("=" * 70)
        print(f"Total input:                          {self.stats['total_input']:>10,}")
        print(f"  - Extreme cases eliminated:         {self.stats['eliminated_extreme_inactive']:>10,}")
        print(f"Total output (with penalties):        {self.stats['total_output']:>10,}")
        eliminated = self.stats['total_input'] - self.stats['total_output']
        rate = (eliminated / max(self.stats['total_input'], 1)) * 100
        print(f"Elimination rate:                     {rate:>9.1f}%")
        print("\nNotes:")
        print("  - Most candidates pass through with behavioral penalties.")
        print("  - _behavioral_penalty stores the final scalar penalty.")
        print("  - _behavioral_features stores the component breakdown for Stage 6.")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Stage 5: Behavioral signal penalties.'
    )
    parser.add_argument('--input', required=True, help='Input JSONL')
    parser.add_argument('--output', required=True, help='Output JSONL')
    args = parser.parse_args()

    stage = Stage5BehavioralPenalties()
    try:
        stage.process_jsonl(args.input, args.output)
        stage.print_report()
        print(f"\nOutput: {args.output}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())