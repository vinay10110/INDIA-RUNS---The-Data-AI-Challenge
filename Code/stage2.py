#!/usr/bin/env python3
"""
STAGE 2: Experience Relevance Scoring (JD-Aligned, Score-Based Gate)

Goal:
- Keep candidates who show evidence of building relevant systems:
  - matching / ranking
  - search / retrieval
  - recommendation / personalization
- Prefer operational / shipped production ML experience over generic AI keyword stuffing
- Avoid brittle one-keyword pass logic
- Avoid hard-eliminating service-firm-only candidates before examining actual evidence

This stage is NOT true embedding-based semantic search.
It is a rule-based evidence scorer that:
1. looks across multiple candidate text sources
2. groups evidence into concept buckets
3. aggregates evidence across roles
4. applies penalties instead of brittle binary rules where appropriate

Output:
- Candidates who clear a minimum Stage 2 relevance threshold
- Adds Stage 2 metadata onto each passing candidate:
    _stage2_score
    _stage2_breakdown
"""

import json
import argparse
import re
from pathlib import Path
from typing import Dict, Any, Tuple, List, Set


class Stage2ExperienceRelevance:
    """
    Score-based Stage 2 gate aligned to retrieval / ranking / recommendation experience.
    """

    # -----------------------------
    # Core evidence buckets
    # -----------------------------
    MATCHING_RANKING_TERMS = {
        "matching", "matchmaking", "candidate matching", "job matching",
        "relevance", "relevance scoring", "relevance ranking",
        "ranking", "ranker", "re-ranking", "reranking", "learning to rank",
        "ltr", "ranking model", "feed ranking", "sorting", "affinity scoring",
        "scoring model", "two-sided matching", "marketplace matching",
        "candidate ranking", "search ranking", "job-feed ordering",
        "job ranking", "relevance model", "ranking pipeline"
    }

    SEARCH_RETRIEVAL_TERMS = {
        "retrieval", "information retrieval", "ir", "semantic search",
        "vector search", "dense retrieval", "hybrid retrieval",
        "candidate retrieval", "document retrieval", "search",
        "search infrastructure", "search engine", "query understanding",
        "query expansion", "indexing", "index refresh", "search relevance",
        "retrieval pipeline", "embedding search", "nearest neighbor search",
        "ann", "faiss", "milvus", "pinecone", "weaviate", "elasticsearch",
        "opensearch", "solr", "retriever"
    }

    RECOMMENDATION_TERMS = {
        "recommendation", "recommender", "recommendation engine",
        "personalization", "personalisation", "feed ranking",
        "content recommendation", "content discovery", "suggestion engine",
        "recommend", "personalized ranking", "candidate recommendations",
        "job recommendations", "recommendation system", "recsys",
        "collaborative filtering", "user-item matching", "user recommendations"
    }

    # Operational / shipped-system evidence
    PRODUCTION_TERMS = {
        "production", "productionized", "productionised", "deployed",
        "launched", "shipped", "went live", "serving", "real users",
        "at scale", "scaled", "scale", "latency", "throughput", "sla",
        "monitoring", "observability", "drift", "embedding drift",
        "quality regression", "incident", "rollback", "online serving",
        "batch pipeline", "real-time pipeline", "index refresh",
        "deployed to production", "operated", "maintained in production",
        "millions", "users", "traffic"
    }

    # Evaluation / experimentation evidence
    EVAL_TERMS = {
        "ab test", "a/b test", "ab-testing", "online experiment",
        "offline evaluation", "evaluation", "metrics", "precision", "recall",
        "mrr", "map", "ndcg", "ctr", "conversion", "lift", "quality metrics",
        "relevance metrics", "ranking metrics", "model evaluation",
        "benchmarking", "experiment", "experimentation"
    }

    # Signals that role is more likely product / platform / consumer-facing
    PRODUCT_CONTEXT_TERMS = {
        "platform", "marketplace", "consumer", "product", "user-facing",
        "saas", "app", "mobile app", "web app", "search platform",
        "recommendation platform", "ads platform", "feed", "discovery",
        "growth", "personalization platform", "matching platform"
    }

    # Signals of research-heavy / non-production emphasis
    RESEARCH_TERMS = {
        "research", "published", "paper", "papers", "publication",
        "thesis", "investigated", "experimental study", "academic",
        "novel method", "theoretical"
    }

    # Consulting / service companies — penalty only, not auto-elimination
    SERVICE_FIRMS = {
        "tcs", "infosys", "wipro", "accenture", "cognizant",
        "capgemini", "mindtree", "hcl", "tech mahindra",
        "lti", "ltimindtree", "mphasis", "ibm consulting",
        "deloitte", "pwc", "ey", "kpmg"
    }

    # Seniority hints
    SENIORITY_TERMS = {
        "senior", "staff", "lead", "principal", "architect", "manager"
    }

    # -----------------------------
    # Scoring config
    # -----------------------------
    SCORE_MATCHING = 20
    SCORE_SEARCH = 20
    SCORE_RECOMMENDATION = 20
    SCORE_PRODUCTION = 15
    SCORE_EVALUATION = 10
    SCORE_PRODUCT_CONTEXT = 8
    SCORE_MULTI_RELEVANT_ROLE = 10
    SCORE_SENIOR_RELEVANT_ROLE = 8

    PENALTY_SERVICE_ONLY = -12
    PENALTY_RESEARCH_HEAVY_NO_PROD = -10

    # Minimum candidate score to survive Stage 2
    PASS_THRESHOLD = 28

    def __init__(self):
        self.stats = {
            "total_input": 0,
            "eliminated_low_relevance": 0,
            "total_output": 0,
        }

    # ============================================================
    # Normalization / text helpers
    # ============================================================

    def normalize(self, text: Any) -> str:
        """Lowercase, normalize whitespace, safe for None/non-string."""
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize_skills(self, candidate: Dict[str, Any]) -> List[str]:
        """
        Extract skill names from candidate['skills'] if present.
        Handles both:
        - [{"name": "Python"}, ...]
        - ["Python", "ML", ...]
        """
        skills = candidate.get("skills", [])
        out = []

        if isinstance(skills, list):
            for s in skills:
                if isinstance(s, dict):
                    name = self.normalize(s.get("name", ""))
                    if name:
                        out.append(name)
                elif isinstance(s, str):
                    name = self.normalize(s)
                    if name:
                        out.append(name)

        return out

    def get_profile_text(self, candidate: Dict[str, Any]) -> str:
        """
        Build a profile-level text block from likely fields if present.
        Keeps this resilient to schema variation.
        """
        profile = candidate.get("profile", {}) or {}

        parts = []
        for key in [
            "headline",
            "summary",
            "about",
            "bio",
            "current_title",
            "current_role",
            "title"
        ]:
            val = profile.get(key)
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())

        return self.normalize(" | ".join(parts))

    def contains_any(self, text: str, phrases: Set[str]) -> bool:
        """Return True if any phrase appears in text, using word boundaries for short terms."""
        if not text:
            return False
        for p in phrases:
            if len(p) <= 3:  # Short acronyms like 'ir', 'ann', 'map'
                if re.search(rf"\b{re.escape(p)}\b", text):
                     return True
            else:
                if p in text:
                     return True
        return False

    # ============================================================
    # Role scoring
    # ============================================================

    def build_role_text(self, role: Dict[str, Any], profile_text: str, skill_text: str) -> Dict[str, str]:
        """
        Build multiple role-aware text views so we don't rely only on description.
        """
        title = self.normalize(role.get("title", ""))
        description = self.normalize(role.get("description", ""))
        company = self.normalize(role.get("company", ""))

        # Focused role text
        role_text = " | ".join(x for x in [title, description] if x)

        # Broader context text
        context_text = " | ".join(
            x for x in [title, description, company, profile_text, skill_text] if x
        )

        return {
            "title": title,
            "description": description,
            "company": company,
            "role_text": role_text,
            "context_text": context_text
        }

    def score_single_role(self, role: Dict[str, Any], profile_text: str, skill_text: str) -> Dict[str, Any]:
        """
        Score one role for relevance evidence.
        Returns a breakdown dict.
        """
        texts = self.build_role_text(role, profile_text, skill_text)
        title = texts["title"]
        description = texts["description"]
        company = texts["company"]
        role_text = texts["role_text"]
        context_text = texts["context_text"]

        breakdown = {
            "matching": False,
            "search": False,
            "recommendation": False,
            "production": False,
            "evaluation": False,
            "product_context": False,
            "senior_relevant": False,
            "research_heavy": False,
            "score": 0
        }

        score = 0

        # Strongest relevance signals should be looked for in role text first
        if self.contains_any(role_text, self.MATCHING_RANKING_TERMS) or self.contains_any(context_text, self.MATCHING_RANKING_TERMS):
            breakdown["matching"] = True
            score += self.SCORE_MATCHING

        if self.contains_any(role_text, self.SEARCH_RETRIEVAL_TERMS) or self.contains_any(context_text, self.SEARCH_RETRIEVAL_TERMS):
            breakdown["search"] = True
            score += self.SCORE_SEARCH

        if self.contains_any(role_text, self.RECOMMENDATION_TERMS) or self.contains_any(context_text, self.RECOMMENDATION_TERMS):
            breakdown["recommendation"] = True
            score += self.SCORE_RECOMMENDATION

        # Production / shipped-system evidence
        if self.contains_any(context_text, self.PRODUCTION_TERMS):
            breakdown["production"] = True
            score += self.SCORE_PRODUCTION

        # Evaluation / experimentation evidence
        if self.contains_any(context_text, self.EVAL_TERMS):
            breakdown["evaluation"] = True
            score += self.SCORE_EVALUATION

        # Product / platform / marketplace context
        if self.contains_any(context_text, self.PRODUCT_CONTEXT_TERMS):
            breakdown["product_context"] = True
            score += self.SCORE_PRODUCT_CONTEXT

        # Seniority in relevant role
        if (
            self.contains_any(title, self.SENIORITY_TERMS) and
            (breakdown["matching"] or breakdown["search"] or breakdown["recommendation"])
        ):
            breakdown["senior_relevant"] = True
            score += self.SCORE_SENIOR_RELEVANT_ROLE

        # Research-heavy signal
        if self.contains_any(context_text, self.RESEARCH_TERMS):
            breakdown["research_heavy"] = True

        breakdown["score"] = score
        breakdown["role_title"] = role.get("title", "")
        breakdown["company"] = role.get("company", "")
        return breakdown

    # ============================================================
    # Candidate-level scoring
    # ============================================================

    def is_service_firm_only(self, candidate: Dict[str, Any]) -> bool:
        """
        True if every known company in career history appears to be a service/consulting firm.
        Penalty only — not a hard elimination.
        """
        career_history = candidate.get("career_history", []) or []
        companies = []

        for role in career_history:
            company = self.normalize(role.get("company", ""))
            if company:
                companies.append(company)

        if not companies:
            return False

        def is_service_company(company: str) -> bool:
            return any(firm in company for firm in self.SERVICE_FIRMS)

        return all(is_service_company(c) for c in companies)

    def score_candidate(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build candidate-level Stage 2 relevance score.
        """
        career_history = candidate.get("career_history", []) or []
        profile_text = self.get_profile_text(candidate)
        skills = self.tokenize_skills(candidate)
        skill_text = self.normalize(" | ".join(skills))

        role_breakdowns = []
        relevant_role_count = 0

        max_role_score = 0
        has_matching = False
        has_search = False
        has_recommendation = False
        has_production = False
        has_evaluation = False
        has_research = False
        has_senior_relevant = False

        for role in career_history:
            rb = self.score_single_role(role, profile_text, skill_text)
            role_breakdowns.append(rb)

            max_role_score = max(max_role_score, rb["score"])
            has_matching = has_matching or rb["matching"]
            has_search = has_search or rb["search"]
            has_recommendation = has_recommendation or rb["recommendation"]
            has_production = has_production or rb["production"]
            has_evaluation = has_evaluation or rb["evaluation"]
            has_research = has_research or rb["research_heavy"]
            has_senior_relevant = has_senior_relevant or rb["senior_relevant"]

            if rb["matching"] or rb["search"] or rb["recommendation"]:
                relevant_role_count += 1

        # Candidate aggregate score starts from strongest role
        candidate_score = max_role_score

        # Bonus if multiple roles show relevant system-building work
        if relevant_role_count >= 2:
            candidate_score += self.SCORE_MULTI_RELEVANT_ROLE

        # Service-only penalty
        service_only = self.is_service_firm_only(candidate)
        if service_only:
            candidate_score += self.PENALTY_SERVICE_ONLY

        # Research-heavy without production gets penalized
        if has_research and not has_production:
            candidate_score += self.PENALTY_RESEARCH_HEAVY_NO_PROD

        # Small additional boost if candidate has breadth across multiple relevant categories
        category_count = sum([has_matching, has_search, has_recommendation])
        if category_count >= 2:
            candidate_score += 5
        if category_count == 3:
            candidate_score += 5

        # Guardrail: if absolutely no core category evidence, candidate should fail Stage 2
        has_any_core = has_matching or has_search or has_recommendation
        if not has_any_core:
            candidate_score = min(candidate_score, 10)

        breakdown = {
            "candidate_score": candidate_score,
            "max_role_score": max_role_score,
            "relevant_role_count": relevant_role_count,
            "has_matching": has_matching,
            "has_search": has_search,
            "has_recommendation": has_recommendation,
            "has_production": has_production,
            "has_evaluation": has_evaluation,
            "has_senior_relevant": has_senior_relevant,
            "service_only_penalty_applied": service_only,
            "research_without_prod_penalty": bool(has_research and not has_production),
            "role_breakdowns": role_breakdowns,
        }

        return breakdown

    def filter_candidate(self, candidate: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Score candidate and apply Stage 2 pass threshold.
        Returns:
            (passes, reason, breakdown)
        """
        breakdown = self.score_candidate(candidate)
        score = breakdown["candidate_score"]

        if score < self.PASS_THRESHOLD:
            self.stats["eliminated_low_relevance"] += 1
            return False, f"Low Stage 2 relevance score: {score}", breakdown

        return True, f"Stage 2 relevance score: {score}", breakdown

    # ============================================================
    # IO
    # ============================================================

    def process_jsonl(self, input_path: str, output_path: str) -> Dict[str, int]:
        """
        Load input JSONL, apply Stage 2 scoring gate, write passing candidates.
        """
        input_file = Path(input_path)
        output_file = Path(output_path)

        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        self.stats["total_input"] = 0
        self.stats["total_output"] = 0

        with open(input_file, "r", encoding="utf-8") as infile, \
             open(output_file, "w", encoding="utf-8") as outfile:

            for line in infile:
                line = line.strip()
                if not line:
                    continue

                try:
                    candidate = json.loads(line)
                except json.JSONDecodeError:
                    continue

                self.stats["total_input"] += 1
                passes, _, breakdown = self.filter_candidate(candidate)

                if passes:
                    # Persist Stage 2 evidence for downstream stages
                    candidate["_stage2_score"] = breakdown["candidate_score"]
                    candidate["_stage2_breakdown"] = breakdown

                    # Align with Stage 6 expectations
                    has_matching = breakdown.get("has_matching", False)
                    has_search = breakdown.get("has_search", False)
                    has_recommendation = breakdown.get("has_recommendation", False)
                    has_production = breakdown.get("has_production", False)
                    has_evaluation = breakdown.get("has_evaluation", False)
                    has_product_context = any(rb.get("product_context", False) for rb in breakdown.get("role_breakdowns", []))

                    candidate["_stage2_features"] = {
                        "retrieval_score": 100.0 if has_search else 0.0,
                        "matching_score": 100.0 if has_matching else 0.0,
                        "recommendation_score": 100.0 if has_recommendation else 0.0,
                        "production_score": 100.0 if has_production else 0.0,
                        "evaluation_score": 100.0 if has_evaluation else 0.0,
                        "product_company_score": 100.0 if has_product_context else 0.0,
                        "experience_fit_score": float(breakdown["candidate_score"]),
                        "service_only_flag": bool(breakdown.get("service_only_penalty_applied", False))
                    }

                    outfile.write(json.dumps(candidate, ensure_ascii=False) + "\n")
                    self.stats["total_output"] += 1

        return self.stats

    def print_report(self):
        """
        Print Stage 2 summary.
        """
        print("\n" + "=" * 80)
        print("STAGE 2: EXPERIENCE RELEVANCE SCORING (JD-ALIGNED GATE)")
        print("=" * 80)
        print(f"Total input:                          {self.stats['total_input']:>10,}")
        print(f"  - Eliminated low relevance:         {self.stats['eliminated_low_relevance']:>10,}")
        print(f"Total output:                         {self.stats['total_output']:>10,}")

        eliminated = self.stats["total_input"] - self.stats["total_output"]
        rate = (eliminated / max(self.stats["total_input"], 1)) * 100
        print(f"Elimination rate:                     {rate:>9.1f}%")
        print(f"Pass threshold:                       {self.PASS_THRESHOLD:>10}")
        print("=" * 80)
        print("Notes:")
        print(" - Service-firm-only background is penalized, not auto-eliminated.")
        print(" - Role title, description, profile text, and skills all contribute.")
        print(" - Candidates need meaningful evidence of matching/search/recommendation work.")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Experience relevance scoring gate."
    )
    parser.add_argument("--input", required=True, help="Input JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL")
    args = parser.parse_args()

    stage = Stage2ExperienceRelevance()
    try:
        stage.process_jsonl(args.input, args.output)
        stage.print_report()
        print(f"\nOutput: {args.output}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())