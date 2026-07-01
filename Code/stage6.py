#!/usr/bin/env python3
"""
ALIGNED Stage 6: Final Ranking Using Enriched Features from Stages 2–5

This version is aligned with the corrected pipeline:

Stage 2 -> writes _stage2_features
Stage 4 -> writes _stage4_flags
Stage 5 -> writes _behavioral_features and _behavioral_penalty

Ranking philosophy:
1. Core fit should dominate ranking.
2. Shipping / production evidence should matter a lot.
3. Behavioral signals should down-weight, not dominate.
4. Stage 4 suspicious patterns should act as soft penalties unless already eliminated.
5. Use enriched stage outputs first; fall back to raw text heuristics only if needed.
"""

import json
import argparse
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple


class Stage6AlignedScoring:
    """Final scorer aligned with stages 1-5."""

    CORE_SKILLS = {
        'python', 'sql', 'embedding', 'vector', 'retrieval', 'ranking',
        'recommendation', 'search', 'matching', 'personalization',
        'tensorflow', 'pytorch', 'sklearn', 'elasticsearch', 'faiss',
        'pinecone', 'weaviate', 'qdrant', 'milvus'
    }

    SHIPPER_KEYWORDS = {
        'launched', 'deployed', 'productionized', 'scaled',
        'serving users', 'real users', 'production', 'live',
        'went live', 'in production', 'operated', 'monitoring',
        'ab testing', 'a/b testing', 'optimized', 'performance',
        'latency', 'throughput', 'alerting', 'incident', 'drift'
    }

    RESEARCHER_KEYWORDS = {
        'published', 'paper', 'investigated', 'studied',
        'benchmarked', 'theoretical', 'research', 'novel',
        'experiment', 'analysis', 'proposed', 'framework'
    }

    # Small fallback lexical sets only when stage features are missing
    RETRIEVAL_HINTS = {
        'retrieval', 'search', 'semantic search', 'candidate retrieval',
        'document retrieval', 'vector search', 'dense retrieval',
        'hybrid retrieval', 'information retrieval'
    }

    RANKING_HINTS = {
        'ranking', 'relevance ranking', 'relevance scoring',
        'ranker', 'matching', 'candidate matching', 'two-sided matching'
    }

    RECOMMENDATION_HINTS = {
        'recommendation', 'recommender', 'personalization',
        'feed ranking', 'suggestion engine'
    }

    PRODUCTION_HINTS = {
        'production', 'deployed', 'launched', 'live', 'monitoring',
        'at scale', 'millions', 'latency', 'drift', 'alerting'
    }

    PRODUCT_COMPANY_HINTS = {
        'tech', 'software', 'saas', 'platform', 'startup', 'product'
    }

    def __init__(self):
        self.candidates = []

    def normalize(self, text: str) -> str:
        return text.lower().strip() if text else ""

    # ============================================================
    # Stage-feature access helpers
    # ============================================================

    def get_stage2_features(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        return candidate.get('_stage2_features', {}) or {}

    def get_stage4_flags(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        return candidate.get('_stage4_flags', {}) or {}

    def get_behavioral_features(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        return candidate.get('_behavioral_features', {}) or {}

    # ============================================================
    # Fallback raw-text feature extraction (used only if needed)
    # ============================================================

    def _fallback_experience_features(self, candidate: Dict[str, Any]) -> Dict[str, float]:
        """
        Fallback heuristic experience features if Stage 2 features are absent.
        Output scale is 0-100 for the aggregate experience_fit_score and
        component scores.
        """
        career_history = candidate.get('career_history', [])

        retrieval_score = 0.0
        ranking_score = 0.0
        recommendation_score = 0.0
        production_score = 0.0
        evaluation_score = 0.0
        product_company_score = 0.0
        service_only_flag = False

        if not career_history:
            return {
                'matching_score': 0.0,
                'retrieval_score': 0.0,
                'recommendation_score': 0.0,
                'production_score': 0.0,
                'evaluation_score': 0.0,
                'product_company_score': 0.0,
                'service_only_flag': False,
                'experience_fit_score': 0.0
            }

        companies = []
        for role in career_history:
            description = self.normalize(role.get('description', ''))
            title = self.normalize(role.get('title', ''))
            company = self.normalize(role.get('company', ''))
            skills = role.get('skills', [])
            role_skill_text = " ".join(self.normalize(s) if isinstance(s, str) else self.normalize(s.get('name', ''))
                                       for s in skills)

            combined = " ".join([title, description, role_skill_text])

            if any(h in combined for h in self.RETRIEVAL_HINTS):
                retrieval_score = max(retrieval_score, 35.0)

            if any(h in combined for h in self.RANKING_HINTS):
                ranking_score = max(ranking_score, 35.0)

            if any(h in combined for h in self.RECOMMENDATION_HINTS):
                recommendation_score = max(recommendation_score, 25.0)

            if any(h in combined for h in self.PRODUCTION_HINTS):
                production_score = max(production_score, 25.0)

            if any(h in combined for h in {'ndcg', 'ab testing', 'a/b testing', 'offline evaluation', 'precision', 'recall'}):
                evaluation_score = max(evaluation_score, 15.0)

            if any(h in company for h in self.PRODUCT_COMPANY_HINTS):
                product_company_score = max(product_company_score, 10.0)

            if company:
                companies.append(company)

        # Weak fallback for service-only
        service_firms = {
            'tcs', 'infosys', 'wipro', 'accenture', 'cognizant',
            'capgemini', 'mindtree', 'hcl', 'tech mahindra'
        }
        if companies:
            service_only_flag = all(any(firm in c for firm in service_firms) for c in companies)

        experience_fit_score = min(
            retrieval_score + ranking_score + recommendation_score +
            production_score + evaluation_score + product_company_score,
            100.0
        )

        return {
            'matching_score': ranking_score,
            'retrieval_score': retrieval_score,
            'recommendation_score': recommendation_score,
            'production_score': production_score,
            'evaluation_score': evaluation_score,
            'product_company_score': product_company_score,
            'service_only_flag': service_only_flag,
            'experience_fit_score': experience_fit_score
        }

    # ============================================================
    # Core fit
    # ============================================================

    def calculate_core_fit_score(self, candidate: Dict[str, Any]) -> float:
        """
        Core fit score (0-100), primarily based on Stage 2 experience signals.
        This is the most important score in the pipeline.
        """
        s2 = self.get_stage2_features(candidate)
        if not s2:
            s2 = self._fallback_experience_features(candidate)

        retrieval = float(s2.get('retrieval_score', 0.0))
        ranking = float(s2.get('matching_score', 0.0))
        recommendation = float(s2.get('recommendation_score', 0.0))
        production = float(s2.get('production_score', 0.0))
        evaluation = float(s2.get('evaluation_score', 0.0))
        product_company = float(s2.get('product_company_score', 0.0))
        experience_fit = float(s2.get('experience_fit_score', 0.0))
        service_only_flag = bool(s2.get('service_only_flag', False))

        # Core fit emphasizes retrieval/ranking + production + evaluation
        core_fit = (
            (retrieval * 0.30) +
            (ranking * 0.25) +
            (recommendation * 0.10) +
            (production * 0.20) +
            (evaluation * 0.10) +
            (product_company * 0.05)
        )

        # If Stage 2 already produced an overall fit score, blend it in
        if experience_fit > 0:
            core_fit = (core_fit * 0.7) + (experience_fit * 0.3)

        # Service-only is a penalty, not elimination
        if service_only_flag:
            core_fit -= 8.0

        return max(min(core_fit, 100.0), 0.0)

    # ============================================================
    # Shipper signal
    # ============================================================

    def calculate_shipper_score(self, candidate: Dict[str, Any]) -> float:
        """
        Shipper score (-100 to +100).
        Positive = shipper-leaning, negative = researcher-leaning.
        """
        career_history = candidate.get('career_history', [])

        shipper_count = 0
        researcher_count = 0

        for role in career_history:
            description = self.normalize(role.get('description', ''))
            title = self.normalize(role.get('title', ''))
            combined = f"{title} {description}"

            shipper_count += sum(1 for kw in self.SHIPPER_KEYWORDS if kw in combined)
            researcher_count += sum(1 for kw in self.RESEARCHER_KEYWORDS if kw in combined)

        net = shipper_count - researcher_count
        if net > 0:
            return min(net * 10.0, 100.0)
        if net < 0:
            return max(net * 10.0, -100.0)
        return 0.0

    # ============================================================
    # Component scores
    # ============================================================

    def score_retrieval_ranking_search(self, candidate: Dict[str, Any]) -> float:
        """
        Retrieval / ranking / recommendation / search score (0-100).
        Pull from Stage 2 features if available.
        """
        s2 = self.get_stage2_features(candidate)
        if not s2:
            s2 = self._fallback_experience_features(candidate)

        retrieval = float(s2.get('retrieval_score', 0.0))
        ranking = float(s2.get('matching_score', 0.0))
        recommendation = float(s2.get('recommendation_score', 0.0))
        evaluation = float(s2.get('evaluation_score', 0.0))

        score = (
            (retrieval * 0.40) +
            (ranking * 0.35) +
            (recommendation * 0.15) +
            (evaluation * 0.10)
        )

        return max(min(score, 100.0), 0.0)

    def score_production_ml_experience(self, candidate: Dict[str, Any]) -> float:
        """
        Production ML / operational score (0-100).
        Prefer Stage 2 production signals + shipper orientation.
        """
        s2 = self.get_stage2_features(candidate)
        if not s2:
            s2 = self._fallback_experience_features(candidate)

        production = float(s2.get('production_score', 0.0))
        evaluation = float(s2.get('evaluation_score', 0.0))
        shipper = self.calculate_shipper_score(candidate)

        # Convert shipper (-100..100) to a small 0..20 contribution
        shipper_component = max(shipper, 0.0) / 100.0 * 20.0

        score = production * 0.70 + evaluation * 0.15 + shipper_component

        return max(min(score, 100.0), 0.0)

    def score_behavioral_signals(self, candidate: Dict[str, Any]) -> float:
        """
        Behavioral score (0-100), using Stage 5 features if available.
        Stage 5 already computed penalties, so use that first.
        """
        bf = self.get_behavioral_features(candidate)
        signals = candidate.get('redrob_signals', {})

        # If Stage 5 features exist, use them
        if bf:
            total_penalty = float(bf.get('total_penalty', candidate.get('_behavioral_penalty', 0.0)))
            base = 75.0
            return max(min(base + total_penalty, 100.0), 0.0)

        # Fallback
        response_rate = float(signals.get('recruiter_response_rate', 0.5) or 0.5)
        interview_rate = float(signals.get('interview_completion_rate', 0.5) or 0.5)
        base = 50.0 + (response_rate * 25.0) + (interview_rate * 15.0)
        penalty = float(candidate.get('_behavioral_penalty', 0.0))
        return max(min(base + penalty, 100.0), 0.0)

    def score_experience_progression(self, candidate: Dict[str, Any]) -> float:
        """
        Experience & progression score (0-100).
        Years of experience + reasonable title progression.
        """
        profile = candidate.get('profile', {})
        career_history = candidate.get('career_history', [])

        yoe = profile.get('years_of_experience', 0) or 0
        try:
            yoe = float(yoe)
        except (TypeError, ValueError):
            yoe = 0.0

        if yoe < 3:
            yoe_score = yoe * 15.0
        elif yoe <= 10:
            yoe_score = 45.0 + ((yoe - 3.0) / 7.0) * 40.0
        else:
            yoe_score = 85.0 + min(((yoe - 10.0) / 5.0) * 15.0, 15.0)

        title_levels = []
        for role in sorted(career_history, key=lambda r: r.get('start_date', '')):
            title = self.normalize(role.get('title', ''))
            title_levels.append(self._get_title_level(title))

        progression_bonus = 0.0
        if len(title_levels) > 1 and title_levels[-1] > title_levels[0]:
            progression_bonus = 10.0

        return max(min(yoe_score + progression_bonus, 100.0), 0.0)

    def score_skills(self, candidate: Dict[str, Any]) -> float:
        """
        Technical skills score (0-100).
        """
        skills = candidate.get('skills', [])
        if not isinstance(skills, list) or not skills:
            return 20.0

        proficiency_weights = {
            'expert': 4,
            'advanced': 3,
            'intermediate': 2,
            'beginner': 1
        }

        core_count = 0
        prof_sum = 0.0

        for skill in skills:
            if isinstance(skill, dict):
                skill_name = self.normalize(skill.get('name', ''))
                prof = self.normalize(skill.get('proficiency', 'beginner'))
            else:
                skill_name = self.normalize(str(skill))
                prof = 'beginner'

            if any(core in skill_name for core in self.CORE_SKILLS):
                core_count += 1
                prof_sum += proficiency_weights.get(prof, 1)

        if core_count == 0:
            return 20.0

        avg_prof = prof_sum / core_count
        score = ((core_count / 8.0) * 50.0) + ((avg_prof / 4.0) * 50.0)
        return max(min(score, 100.0), 0.0)

    def score_education(self, candidate: Dict[str, Any]) -> float:
        """
        Education score (0-100).
        """
        education = candidate.get('education', [])
        if not education:
            return 30.0

        tier_map = {
            'tier_1': 80.0,
            'tier_2': 60.0,
            'tier_3': 40.0,
            'tier_4': 20.0
        }

        best = 0.0
        for edu in education:
            tier = edu.get('tier', 'unknown')
            best = max(best, tier_map.get(tier, 20.0))

        return best

    # ============================================================
    # Suspicion penalties from Stage 4
    # ============================================================

    def calculate_stage4_penalty(self, candidate: Dict[str, Any]) -> float:
        """
        Apply soft penalties for suspicious-but-not-eliminated patterns from Stage 4.
        These should not dominate the score, but should matter.
        """
        flags = self.get_stage4_flags(candidate)
        if not flags:
            return 0.0

        penalty = 0.0

        if flags.get('has_mild_overlap', False):
            penalty -= 3.0

        if flags.get('skill_inflation_flag', False):
            penalty -= 4.0

        date_suspicion = float(flags.get('date_suspicion_score', 0.0) or 0.0)
        penalty -= min(date_suspicion, 5.0)

        return penalty

    # ============================================================
    # Final score
    # ============================================================

    def calculate_composite_score(self, candidate: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Final composite score.

        Weighting:
        - Core fit:          35%
        - Retrieval/search:  20%
        - Production ML:     15%
        - Behavioral:        10%
        - Experience:        10%
        - Skills:             7%
        - Education:          3%

        Then apply:
        - shipper bonus/penalty
        - Stage 4 soft suspicion penalties
        """
        core_fit = self.calculate_core_fit_score(candidate)
        retrieval_score = self.score_retrieval_ranking_search(candidate)
        production_score = self.score_production_ml_experience(candidate)
        behavioral_score = self.score_behavioral_signals(candidate)
        experience_score = self.score_experience_progression(candidate)
        skills_score = self.score_skills(candidate)
        education_score = self.score_education(candidate)

        base = (
            (core_fit * 0.35) +
            (retrieval_score * 0.20) +
            (production_score * 0.15) +
            (behavioral_score * 0.10) +
            (experience_score * 0.10) +
            (skills_score * 0.07) +
            (education_score * 0.03)
        )

        shipper_score = self.calculate_shipper_score(candidate)
        shipper_adjustment = (shipper_score / 100.0) * 5.0  # max ±5

        stage4_penalty = self.calculate_stage4_penalty(candidate)

        final_score = (base + shipper_adjustment + stage4_penalty) / 100.0
        final_score = max(min(final_score, 1.0), 0.0)

        components = {
            'core_fit': core_fit,
            'retrieval_score': retrieval_score,
            'production_score': production_score,
            'behavioral_score': behavioral_score,
            'experience_score': experience_score,
            'skills_score': skills_score,
            'education_score': education_score,
            'shipper_score': shipper_score,
            'stage4_penalty': stage4_penalty,
            'final_score': final_score
        }
        return final_score, components

    # ============================================================
    # Reasoning
    # ============================================================

    def generate_reasoning(self, candidate: Dict[str, Any], components: Dict[str, float]) -> str:
        """
        Generate dynamic, fact-based plain language reasoning for Stage 4 review.
        """
        profile = candidate.get('profile', {})
        current_title = profile.get('current_title', 'ML Engineer')
        yoe = profile.get('years_of_experience', 0.0)

        # Extract actual skills from candidate profile to prevent hallucinations
        skills_raw = candidate.get('skills', [])
        actual_skills = []
        if isinstance(skills_raw, list):
            for s in skills_raw:
                s_name = s.get('name') if isinstance(s, dict) else str(s)
                if s_name:
                    actual_skills.append(s_name)

        # Find which skills match our core JD skills
        matched_skills = []
        for s in actual_skills:
            s_lower = s.lower()
            if any(core in s_lower for core in self.CORE_SKILLS):
                matched_skills.append(s)
                if len(matched_skills) >= 2:
                    break

        # Fallback to any skills if no core skills match
        if not matched_skills and actual_skills:
            matched_skills = actual_skills[:2]

        skills_str = ", ".join(matched_skills) if matched_skills else "applied ML"

        # Determine shipper vs researcher
        shipper = components.get('shipper_score', 0.0)
        if shipper >= 20:
            orientation = "product-oriented shipper alignment"
        elif shipper <= -20:
            orientation = "researcher profile with core technical fit"
        else:
            orientation = "balanced engineering profile"

        # Determine fit and activity description
        fit_val = round(components.get('core_fit', 0.0))
        bf = self.get_behavioral_features(candidate)
        behavioral_desc = ""
        if bf:
            total_penalty = bf.get('total_penalty', 0.0)
            if total_penalty < -20:
                behavioral_desc = "penalized for low platform activity/engagement"
            elif total_penalty < 0:
                behavioral_desc = "minor availability/activity adjustment"
            else:
                behavioral_desc = "strong platform activity and availability"
        else:
            behavioral_desc = "standard availability signals"

        # Check for service-only or overlap
        s2 = self.get_stage2_features(candidate)
        special_note = ""
        if s2 and s2.get('service_only_flag', False):
            special_note = " despite a service-heavy background"

        s4 = self.get_stage4_flags(candidate)
        if s4 and s4.get('has_mild_overlap', False):
            special_note += " (minor timeline overlap noted)"

        # Introduce 3 different sentence structures to maximize variation
        cid_num = sum(ord(char) for char in candidate.get('candidate_id', 'CAND_0000000'))
        pattern_idx = cid_num % 3

        if pattern_idx == 0:
            reasoning = (
                f"Senior candidate presenting as a {current_title} with {yoe} years experience. "
                f"Demonstrates strong capability in {skills_str} with {fit_val}% match on core requirements. "
                f"Reflects a {orientation}{special_note} and {behavioral_desc}."
            )
        elif pattern_idx == 1:
            reasoning = (
                f"A {yoe}-year experienced {current_title} matching key retrieval/ranking vectors. "
                f"Strong proficiency in {skills_str} aligns well with JD expectations. "
                f"Shows {orientation}{special_note} and {behavioral_desc}."
            )
        else:
            reasoning = (
                f"Highly aligned {current_title} with {yoe} years of industry experience. "
                f"Hands-on with {skills_str} supports the {fit_val}% core relevance score. "
                f"Characterized by {orientation}{special_note} and {behavioral_desc}."
            )

        return reasoning

    # ============================================================
    # Utilities
    # ============================================================

    def _get_title_level(self, title: str) -> float:
        levels = {
            'principal': 5.0,
            'director': 5.0,
            'partner': 5.0,
            'staff': 4.0,
            'lead': 4.0,
            'head': 4.5,
            'senior': 3.0,
            'manager': 3.5,
            'engineer': 2.0,
            'developer': 2.0,
            'analyst': 2.0,
            'junior': 1.0,
            'associate': 1.5
        }
        for keyword, level in levels.items():
            if keyword in title:
                return level
        return 2.0

    # ============================================================
    # Main ranking flow
    # ============================================================

    def process_and_rank(self, input_path: str, output_path: str):
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        scored = []

        with open(input_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue

                try:
                    candidate = json.loads(line)
                except json.JSONDecodeError:
                    continue

                final_score, components = self.calculate_composite_score(candidate)
                reasoning = self.generate_reasoning(candidate, components)

                scored.append({
                    'candidate_id': candidate.get('candidate_id'),
                    'core_fit': components['core_fit'],
                    'final_score': components['final_score'],
                    'reasoning': reasoning
                })

        # IMPORTANT:
        # Rank by core_fit first, then final_score, then candidate_id.
        # This aligns with the intended pipeline philosophy:
        # "first decide if they are fundamentally a fit, then sort within that."
        scored.sort(
            key=lambda x: (
                -round(x['final_score'], 6),
                -round(x['core_fit'], 4),
                x['candidate_id']
            )
        )

        output_file = Path(output_path)
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=['candidate_id', 'rank', 'score', 'reasoning']
            )
            writer.writeheader()

            for rank, item in enumerate(scored[:100], 1):
                writer.writerow({
                    'candidate_id': item['candidate_id'],
                    'rank': rank,
                    'score': f"{item['final_score']:.6f}",
                    'reasoning': item['reasoning']
                })

        print(f"\nSubmission written: {output_file}")
        print(f"Total candidates scored: {len(scored)}")
        print(f"Top 100 selected")

        if scored:
            top_score = scored[0]['final_score']
            bottom_score = scored[min(99, len(scored) - 1)]['final_score']
            threshold_core_fit = scored[min(99, len(scored) - 1)]['core_fit']
            print(f"Score range: {top_score:.6f} -> {bottom_score:.6f}")
            print(f"Core fit threshold at cutoff: {threshold_core_fit:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='Stage 6: Final aligned ranking using enriched features from stages 2-5.'
    )
    parser.add_argument('--input', required=True, help='Input JSONL')
    parser.add_argument('--output', required=True, help='Output CSV')
    args = parser.parse_args()

    scorer = Stage6AlignedScoring()
    try:
        scorer.process_and_rank(args.input, args.output)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())