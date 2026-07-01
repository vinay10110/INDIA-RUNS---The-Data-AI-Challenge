#!/usr/bin/env python3
"""
REFINED Stage 4: Suspicious Honeypot Pattern Detection

Stage 4 should align with earlier stages:
- Stage 1 handles raw data corruption / invalid dates / impossible date ranges
- Stage 2 handles relevance scoring and weak-fit filtering
- Stage 3 handles JD-based disqualifiers
- Stage 4 handles suspicious profile fabrication / honeypot-like patterns

Key principles:
1. Do NOT duplicate Stage 1 raw date validation here.
2. Not all overlaps are bad:
   - Same-company transition/promotion overlap is often normal
   - Freelance/consulting overlap is often normal
3. Skill endorsement inflation alone should NOT hard-eliminate a candidate.
   It should contribute to a suspiciousness score.
4. Eliminate only when suspicious evidence is strong enough overall.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime, date
from typing import Dict, Any, Tuple, List, Optional


class Stage4RefinedHoneypot:
    """Detect suspicious honeypot-like patterns using severity-aware logic."""

    OVERLAP_ELIMINATION_THRESHOLD = 4.0  # composite suspicion threshold

    FREELANCE_KEYWORDS = {
        'freelance', 'consultant', 'contract', 'contractor', 'advisor', 'advisory'
    }

    PROMOTION_HINTS = {
        'senior', 'staff', 'principal', 'lead', 'manager', 'director', 'head'
    }

    def __init__(self):
        self.stats = {
            'total_input': 0,
            'eliminated_honeypot_pattern': 0,
            'flagged_severe_overlap': 0,
            'flagged_skill_inflation': 0,
            'total_output': 0,
        }

    def normalize(self, text: Any) -> str:
        """Normalize text safely."""
        return str(text).lower().strip() if text else ""

    def parse_date(self, value: Any) -> Optional[date]:
        """
        Parse YYYY-MM-DD dates.
        Stage 1 should already validate date sanity; here we fail soft and skip if unparsable.
        """
        if not value:
            return None
        try:
            return datetime.strptime(str(value), "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return None

    def role_is_freelance_like(self, role: Dict[str, Any]) -> bool:
        """Check if a role looks freelance / consulting / contract based."""
        title = self.normalize(role.get('title', ''))
        company = self.normalize(role.get('company', ''))
        description = self.normalize(role.get('description', ''))

        haystack = f"{title} {company} {description}"
        return any(kw in haystack for kw in self.FREELANCE_KEYWORDS)

    def extract_level(self, title: str) -> float:
        """Rough title seniority extraction for same-company transition sanity."""
        title = self.normalize(title)
        levels = {
            'intern': 0.5,
            'junior': 1.0,
            'associate': 1.5,
            'engineer': 2.0,
            'developer': 2.0,
            'analyst': 2.0,
            'senior': 3.0,
            'manager': 3.5,
            'lead': 4.0,
            'staff': 4.2,
            'head': 4.5,
            'principal': 5.0,
            'director': 5.0,
            'vp': 5.5,
        }
        best = 2.0
        for kw, lvl in levels.items():
            if kw in title:
                best = max(best, lvl)
        return best

    def looks_like_same_company_transition(self, role1: Dict[str, Any], role2: Dict[str, Any], overlap_days: int) -> bool:
        """
        Heuristic for same-company overlaps that are probably fine:
        - short overlap (<= 120 days)
        - or title progression / internal movement
        """
        title1 = self.normalize(role1.get('title', ''))
        title2 = self.normalize(role2.get('title', ''))

        level1 = self.extract_level(title1)
        level2 = self.extract_level(title2)

        # Small same-company overlap is often harmless
        if overlap_days <= 120:
            return True

        # Title progression / internal move hint
        if level1 != level2:
            return True

        # Same company + one title contains promotion-like markers
        combined = f"{title1} {title2}"
        if any(kw in combined for kw in self.PROMOTION_HINTS):
            return True

        return False

    def compute_overlap_days(self, role1: Dict[str, Any], role2: Dict[str, Any]) -> Optional[int]:
        """Return overlap in days if roles overlap, else None."""
        start1 = self.parse_date(role1.get('start_date'))
        end1 = self.parse_date(role1.get('end_date')) or datetime.now().date()

        start2 = self.parse_date(role2.get('start_date'))
        end2 = self.parse_date(role2.get('end_date')) or datetime.now().date()

        if not start1 or not start2:
            return None

        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)

        if overlap_start <= overlap_end:
            return (overlap_end - overlap_start).days
        return None

    def analyze_timeline_overlap(self, candidate: Dict[str, Any]) -> Tuple[float, List[str], bool]:
        """
        Analyze suspicious overlap patterns.

        Returns:
            suspicion_score,
            reasons,
            severe_overlap_flag

        Scoring philosophy:
        - severe cross-company overlap = strong suspicious signal
        - same-company overlap is usually okay unless it is huge and doesn't
          resemble a transition/promotion
        - freelance/consulting overlap is often okay
        """
        career_history = candidate.get('career_history', [])
        if len(career_history) < 2:
            return 0.0, [], False

        suspicion_score = 0.0
        reasons: List[str] = []
        severe_overlap_flag = False

        for i in range(len(career_history)):
            role1 = career_history[i]
            company1 = self.normalize(role1.get('company', ''))
            title1 = self.normalize(role1.get('title', ''))

            for j in range(i + 1, len(career_history)):
                role2 = career_history[j]
                company2 = self.normalize(role2.get('company', ''))
                title2 = self.normalize(role2.get('title', ''))

                overlap_days = self.compute_overlap_days(role1, role2)
                if overlap_days is None or overlap_days <= 0:
                    continue

                is_same_company = bool(company1 and company2 and company1 == company2)
                is_freelance_1 = self.role_is_freelance_like(role1)
                is_freelance_2 = self.role_is_freelance_like(role2)

                # Freelance / consulting overlap is common; allow moderate overlap
                if is_freelance_1 or is_freelance_2:
                    if overlap_days <= 365:
                        continue
                    # Very large freelance overlap still mildly suspicious
                    suspicion_score += 0.5
                    reasons.append(
                        f"Long freelance/consulting overlap ({overlap_days}d): "
                        f"{title1 or 'role1'} / {title2 or 'role2'}"
                    )
                    continue

                # Same company overlap: allow likely transitions/promotions
                if is_same_company:
                    if self.looks_like_same_company_transition(role1, role2, overlap_days):
                        continue

                    # Huge same-company overlap without transition signal is suspicious,
                    # but weaker than different-company overlap
                    if overlap_days >= 365:
                        suspicion_score += 1.0
                        reasons.append(
                            f"Large same-company overlap without clear transition ({overlap_days}d) at {company1}"
                        )
                    continue

                # Different-company overlaps
                if overlap_days < 90:
                    # short overlap is usually handoff / notice period / part-time transition
                    continue
                elif overlap_days < 180:
                    suspicion_score += 0.75
                    reasons.append(
                        f"Moderate cross-company overlap ({overlap_days}d): {company1} vs {company2}"
                    )
                elif overlap_days < 365:
                    suspicion_score += 1.25
                    reasons.append(
                        f"Large cross-company overlap ({overlap_days}d): {company1} vs {company2}"
                    )
                else:
                    suspicion_score += 2.0
                    severe_overlap_flag = True
                    reasons.append(
                        f"Severe cross-company overlap ({overlap_days}d): {company1} vs {company2}"
                    )

        return suspicion_score, reasons, severe_overlap_flag

    def analyze_skill_inflation(self, candidate: Dict[str, Any]) -> Tuple[float, List[str], bool]:
        """
        Detect suspicious endorsement patterns.

        Important:
        - This is NOT a standalone elimination rule.
        - It contributes to suspicion score.
        """
        skills = candidate.get('skills', [])
        if not isinstance(skills, list) or not skills:
            return 0.0, [], False

        suspicion_score = 0.0
        reasons: List[str] = []
        flagged = False

        for skill in skills:
            if not isinstance(skill, dict):
                continue

            name = skill.get('name', 'unknown')
            endorsements = skill.get('endorsements', 0)
            duration_months = skill.get('duration_months', 0)

            if not isinstance(endorsements, (int, float)) or endorsements < 0:
                continue
            if not isinstance(duration_months, (int, float)) or duration_months <= 0:
                continue

            rate = endorsements / duration_months

            # Mildly suspicious
            if rate > 15 and rate <= 30:
                suspicion_score += 0.35
                flagged = True
                reasons.append(
                    f"High endorsement velocity for {name}: {endorsements} in {duration_months}mo ({rate:.1f}/mo)"
                )

            # More suspicious
            elif rate > 30:
                suspicion_score += 0.75
                flagged = True
                reasons.append(
                    f"Extreme endorsement velocity for {name}: {endorsements} in {duration_months}mo ({rate:.1f}/mo)"
                )

        return suspicion_score, reasons, flagged

    def filter_candidate(self, candidate: Dict[str, Any]) -> Tuple[bool, str]:
        """Apply Stage 4 suspicious honeypot checks."""
        overlap_score, overlap_reasons, severe_overlap = self.analyze_timeline_overlap(candidate)
        skill_score, skill_reasons, skill_flag = self.analyze_skill_inflation(candidate)

        total_score = overlap_score + skill_score
        reasons = overlap_reasons + skill_reasons

        if severe_overlap:
            self.stats['flagged_severe_overlap'] += 1
        if skill_flag:
            self.stats['flagged_skill_inflation'] += 1

        # Eliminate only if suspicion is strong enough overall
        if total_score >= self.OVERLAP_ELIMINATION_THRESHOLD:
            self.stats['eliminated_honeypot_pattern'] += 1
            if reasons:
                return False, " | ".join(reasons[:3])
            return False, f"Composite suspicious honeypot score {total_score:.2f}"

        # Enrich candidate with Stage 4 flags for downstream stages (like Stage 6)
        candidate["_stage4_flags"] = {
            "has_mild_overlap": bool(overlap_score > 0 and not severe_overlap),
            "skill_inflation_flag": bool(skill_flag),
            "date_suspicion_score": float(overlap_score)
        }

        return True, "No suspicious honeypot patterns"

    def process_jsonl(self, input_path: str, output_path: str) -> Dict[str, int]:
        """Load, filter, and save."""
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
                passes, _ = self.filter_candidate(candidate)
                if passes:
                    outfile.write(json.dumps(candidate) + '\n')
                    self.stats['total_output'] += 1

        return self.stats

    def print_report(self):
        """Print statistics."""
        print("\n" + "=" * 70)
        print("STAGE 4 (REFINED): SUSPICIOUS HONEYPOT PATTERN DETECTION")
        print("=" * 70)
        print(f"Total input:                          {self.stats['total_input']:>10,}")
        print(f"  - Eliminated suspicious profiles:   {self.stats['eliminated_honeypot_pattern']:>10,}")
        print(f"  - Flagged severe overlaps:          {self.stats['flagged_severe_overlap']:>10,}")
        print(f"  - Flagged skill inflation:          {self.stats['flagged_skill_inflation']:>10,}")
        print(f"Total output:                         {self.stats['total_output']:>10,}")
        eliminated = self.stats['total_input'] - self.stats['total_output']
        rate = (eliminated / max(self.stats['total_input'], 1)) * 100
        print(f"Elimination rate:                     {rate:>9.1f}%")
        print("\nNotes:")
        print("  - Stage 1 should already have removed raw date corruption.")
        print("  - Same-company transitions/promotions are generally allowed.")
        print("  - Freelance/consulting overlaps are allowed unless unusually large.")
        print("  - Skill inflation contributes to suspicion but does not auto-eliminate.")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Stage 4 (Refined): Suspicious honeypot pattern detection.'
    )
    parser.add_argument('--input', required=True, help='Input JSONL')
    parser.add_argument('--output', required=True, help='Output JSONL')
    args = parser.parse_args()

    stage = Stage4RefinedHoneypot()
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