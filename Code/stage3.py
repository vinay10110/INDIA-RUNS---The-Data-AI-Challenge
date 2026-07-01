#!/usr/bin/env python3
"""
REFINED Stage 3: Disqualifier Detection

Purpose:
Catch explicit JD-style disqualifiers AFTER Stage 2 has already filtered for
relevant retrieval / ranking / recommendation / search experience.

This stage should NOT re-score general relevance. It should only remove
candidates with strong disqualifying patterns such as:
1. Title-chasing via rapid company switching + rapid title inflation
2. Pure research / academic background without shipped production systems
3. Dominant CV / Speech / Robotics background with no meaningful NLP/IR/search exposure
4. Recent LangChain / LLM-wrapper-only background without meaningful pre-2023 ML / IR / ranking / NLP depth

Input:
- JSONL of candidate profiles

Output:
- JSONL of candidates that pass Stage 3

Notes:
- This stage intentionally uses aggregated text across title/description/skills/profile summary.
- It is conservative: a candidate is only eliminated if evidence is fairly strong.
"""

import json
import argparse
import re
from pathlib import Path
from datetime import datetime, date
from typing import Dict, Any, Tuple, List, Optional


class Stage3Disqualifier:
    """Detect disqualifying patterns conservatively."""

    RESEARCH_KEYWORDS = {
        "research", "researcher", "research scientist", "research engineer",
        "phd", "doctoral", "postdoc", "academic", "professor", "university",
        "laboratory", "lab", "publication", "published", "paper", "conference"
    }

    PRODUCTION_KEYWORDS = {
        "production", "deployed", "deployment", "launched", "shipped",
        "in production", "real users", "customers", "millions", "scale",
        "scaled", "serving", "latency", "monitoring", "slo", "sla",
        "pipeline", "online inference", "serving stack", "ab test", "a/b test"
    }

    LLM_STACK_KEYWORDS = {
        "langchain", "llm", "large language model", "gpt", "chatgpt",
        "openai", "prompt engineering", "prompting", "rag",
        "lora", "fine-tuning", "fine tuning", "agents", "agentic"
    }

    PRE_LLM_ML_KEYWORDS = {
        "retrieval", "ranking", "search", "recommendation", "recommender",
        "personalization", "matching", "semantic search", "information retrieval",
        "nlp", "embedding", "embeddings", "classification", "regression",
        "learning to rank", "ltr", "vector search", "candidate retrieval"
    }

    CV_KEYWORDS = {
        "computer vision", "vision model", "image classification", "object detection",
        "segmentation", "face recognition", "video understanding", "opencv",
        "cnn", "image retrieval", "vision transformer", "ocr"
    }

    SPEECH_KEYWORDS = {
        "speech", "asr", "automatic speech recognition", "text-to-speech",
        "tts", "voice", "speaker recognition", "acoustic model", "speech synthesis"
    }

    ROBOTICS_KEYWORDS = {
        "robotics", "robot", "robotic", "manipulation", "motion planning",
        "slam", "lidar", "path planning", "autonomous navigation", "control systems"
    }

    IR_SEARCH_RANKING_KEYWORDS = {
        "retrieval", "ranking", "search", "semantic search", "relevance",
        "re-ranking", "reranking", "matching", "recommendation",
        "personalization", "candidate retrieval", "candidate matching",
        "information retrieval", "vector search", "dense retrieval",
        "hybrid retrieval", "learning to rank", "ltr"
    }

    TITLE_LEVELS = {
        "intern": 0.5,
        "junior": 1.0,
        "associate": 1.5,
        "engineer": 2.0,
        "developer": 2.0,
        "analyst": 2.0,
        "specialist": 2.25,
        "senior": 3.0,
        "manager": 3.5,
        "lead": 4.0,
        "staff": 4.25,
        "head": 4.5,
        "principal": 5.0,
        "director": 5.5,
        "vp": 6.0,
        "vice president": 6.0,
        "partner": 6.0,
    }

    def __init__(self, reference_date: Optional[date] = None):
        self.reference_date = reference_date or datetime.utcnow().date()
        self.stats = {
            "total_input": 0,
            "eliminated_title_chaser": 0,
            "eliminated_pure_research": 0,
            "eliminated_pure_cv_speech_robotics": 0,
            "eliminated_recent_langchain_only": 0,
            "total_output": 0,
        }

    # ---------------------------------------------------------------------
    # Basic helpers
    # ---------------------------------------------------------------------

    def normalize(self, text: Any) -> str:
        if text is None:
            return ""
        return str(text).lower().strip()

    def parse_date(self, value: Optional[str]) -> Optional[date]:
        if not value:
            return None
        try:
            return datetime.strptime(value, "%Y-%m-%d").date()
        except Exception:
            return None

    def months_between(self, start: Optional[date], end: Optional[date]) -> Optional[int]:
        if not start or not end or end < start:
            return None
        return (end.year - start.year) * 12 + (end.month - start.month)

    def count_keyword_hits(self, text: str, keywords: set) -> int:
        count = 0
        for kw in keywords:
            if len(kw) <= 3:
                if re.search(rf"\b{re.escape(kw)}\b", text):
                    count += 1
            else:
                if kw in text:
                    count += 1
        return count

    def any_keyword_hit(self, text: str, keywords: set) -> bool:
        for kw in keywords:
            if len(kw) <= 3:
                if re.search(rf"\b{re.escape(kw)}\b", text):
                    return True
            else:
                if kw in text:
                    return True
        return False

    # ---------------------------------------------------------------------
    # Candidate / role text aggregation
    # ---------------------------------------------------------------------

    def extract_candidate_skill_text(self, candidate: Dict[str, Any]) -> str:
        skills = candidate.get("skills", [])
        skill_parts = []

        if isinstance(skills, list):
            for s in skills:
                if isinstance(s, dict):
                    name = s.get("name", "")
                    skill_parts.append(self.normalize(name))
                else:
                    skill_parts.append(self.normalize(s))

        return " ".join(skill_parts)

    def extract_profile_text(self, candidate: Dict[str, Any]) -> str:
        profile = candidate.get("profile", {}) or {}
        parts = []

        for key in [
            "headline",
            "summary",
            "bio",
            "current_title",
            "current_company",
            "about",
        ]:
            if key in profile:
                parts.append(self.normalize(profile.get(key)))

        return " ".join(parts)

    def build_role_text(self, role: Dict[str, Any], candidate: Dict[str, Any]) -> str:
        """
        Combine title + description + role-specific skill-like fields.
        We intentionally do NOT append full candidate profile text here,
        otherwise every role would inherit the entire candidate-level signal.
        """
        parts = [
            self.normalize(role.get("title", "")),
            self.normalize(role.get("description", "")),
            self.normalize(role.get("company", "")),
        ]

        # Optional role-local fields if present in dataset
        for key in ["summary", "project_summary", "projects", "skills"]:
            value = role.get(key)
            if isinstance(value, list):
                parts.extend(self.normalize(v.get("name", "") if isinstance(v, dict) else v) for v in value)
            elif value:
                parts.append(self.normalize(value))

        return " ".join([p for p in parts if p])

    def build_candidate_text(self, candidate: Dict[str, Any]) -> str:
        """
        Candidate-level aggregated text across profile + skills + all roles.
        """
        chunks = [
            self.extract_profile_text(candidate),
            self.extract_candidate_skill_text(candidate),
        ]

        for role in candidate.get("career_history", []):
            chunks.append(self.build_role_text(role, candidate))

        return " ".join([c for c in chunks if c])

    # ---------------------------------------------------------------------
    # Title chaser detection
    # ---------------------------------------------------------------------

    def extract_title_level(self, title: str) -> float:
        title = self.normalize(title)
        matched_levels = [lvl for kw, lvl in self.TITLE_LEVELS.items() if kw in title]
        return max(matched_levels) if matched_levels else 2.0

    def get_role_tenure_months(self, role: Dict[str, Any]) -> Optional[int]:
        # Prefer explicit duration_months if present and sane
        duration = role.get("duration_months")
        if isinstance(duration, (int, float)) and duration >= 0:
            return int(duration)

        start = self.parse_date(role.get("start_date"))
        end = self.parse_date(role.get("end_date")) or self.reference_date
        return self.months_between(start, end)

    def detect_title_chaser(self, candidate: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Conservative title-chaser detection.

        We only eliminate if the pattern strongly looks like:
        - repeated company switching
        - consistently short tenures
        - fast title escalation
        """
        roles = candidate.get("career_history", [])
        if len(roles) < 3:
            return False, ""

        sortable_roles = []
        for role in roles:
            start = self.parse_date(role.get("start_date"))
            if start is None:
                continue
            sortable_roles.append((start, role))

        if len(sortable_roles) < 3:
            return False, ""

        sortable_roles.sort(key=lambda x: x[0])
        ordered_roles = [r for _, r in sortable_roles]

        companies = [self.normalize(r.get("company", "")) for r in ordered_roles]
        titles = [self.normalize(r.get("title", "")) for r in ordered_roles]
        levels = [self.extract_title_level(t) for t in titles]

        tenures = []
        for r in ordered_roles:
            t = self.get_role_tenure_months(r)
            if t is not None:
                tenures.append(t)

        if len(tenures) < 3:
            return False, ""

        avg_tenure = sum(tenures) / len(tenures)
        company_switches = sum(1 for i in range(1, len(companies)) if companies[i] and companies[i] != companies[i - 1])

        # Count "real upward jumps" of >= 1 level
        upward_jumps = sum(1 for i in range(1, len(levels)) if levels[i] - levels[i - 1] >= 1.0)

        # Large share of short stints
        short_stint_count = sum(1 for t in tenures if t < 18)

        # Conservative elimination rule
        if (
            company_switches >= 2
            and upward_jumps >= 2
            and avg_tenure < 18
            and short_stint_count >= 2
        ):
            return True, (
                f"Title-chaser pattern: {company_switches} company switches, "
                f"{upward_jumps} rapid level jumps, avg tenure {avg_tenure:.1f} months"
            )

        return False, ""

    # ---------------------------------------------------------------------
    # Pure research detection
    # ---------------------------------------------------------------------

    def detect_pure_research(self, candidate: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Eliminate only if the candidate appears predominantly research/academic
        and lacks meaningful production / shipped-system evidence.
        """
        roles = candidate.get("career_history", [])
        if not roles:
            return False, ""

        research_like_roles = 0
        production_like_roles = 0

        for role in roles:
            role_text = self.build_role_text(role, candidate)

            research_hits = self.count_keyword_hits(role_text, self.RESEARCH_KEYWORDS)
            production_hits = self.count_keyword_hits(role_text, self.PRODUCTION_KEYWORDS)

            if research_hits >= 1:
                research_like_roles += 1
            if production_hits >= 1:
                production_like_roles += 1

        total_roles = len(roles)
        candidate_text = self.build_candidate_text(candidate)
        total_production_hits = self.count_keyword_hits(candidate_text, self.PRODUCTION_KEYWORDS)

        research_ratio = research_like_roles / max(total_roles, 1)

        # Pure research if:
        # - most roles are research-like
        # - almost no production roles
        # - little candidate-level shipped-system evidence
        if research_ratio >= 0.8 and production_like_roles == 0 and total_production_hits < 2:
            return True, "Predominantly research / academic background without production evidence"

        return False, ""

    # ---------------------------------------------------------------------
    # Pure CV / Speech / Robotics without IR/NLP/search exposure
    # ---------------------------------------------------------------------

    def detect_pure_domain_without_ir(self, candidate: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Detect candidates whose background is overwhelmingly CV / Speech / Robotics
        but who show no meaningful retrieval / ranking / search / NLP exposure.
        """
        roles = candidate.get("career_history", [])
        if not roles:
            return False, ""

        domain_roles = 0
        ir_roles = 0

        for role in roles:
            role_text = self.build_role_text(role, candidate)

            is_cv = self.any_keyword_hit(role_text, self.CV_KEYWORDS)
            is_speech = self.any_keyword_hit(role_text, self.SPEECH_KEYWORDS)
            is_robotics = self.any_keyword_hit(role_text, self.ROBOTICS_KEYWORDS)
            has_ir = self.any_keyword_hit(role_text, self.IR_SEARCH_RANKING_KEYWORDS)

            if is_cv or is_speech or is_robotics:
                domain_roles += 1
            if has_ir:
                ir_roles += 1

        total_roles = len(roles)
        candidate_text = self.build_candidate_text(candidate)
        candidate_has_ir = self.any_keyword_hit(candidate_text, self.IR_SEARCH_RANKING_KEYWORDS)

        domain_ratio = domain_roles / max(total_roles, 1)

        if domain_ratio >= 0.8 and ir_roles == 0 and not candidate_has_ir:
            return True, "Dominant CV / Speech / Robotics background without meaningful IR/search/ranking exposure"

        return False, ""

    # ---------------------------------------------------------------------
    # LangChain / recent LLM-wrapper-only detection
    # ---------------------------------------------------------------------

    def role_is_recent_llm_role(self, role: Dict[str, Any]) -> bool:
        """
        A role is considered recent-LLM if:
        - role text contains LLM-stack evidence, and
        - role is current OR ended in/after 2023 OR started in/after 2023
        """
        role_text = self.build_role_text(role, {})
        if not self.any_keyword_hit(role_text, self.LLM_STACK_KEYWORDS):
            return False

        start = self.parse_date(role.get("start_date"))
        end = self.parse_date(role.get("end_date"))

        if end is None:
            return True  # current role with LLM stack
        if end.year >= 2023:
            return True
        if start and start.year >= 2023:
            return True
        return False

    def role_has_pre_llm_ml_evidence(self, role: Dict[str, Any]) -> bool:
        """
        Stronger pre-LLM evidence:
        - role ended before 2023 (or started before 2023 if current timeline is messy)
        - role text contains classical ML / IR / ranking / NLP / recommendation signals
        """
        role_text = self.build_role_text(role, {})
        if not self.any_keyword_hit(role_text, self.PRE_LLM_ML_KEYWORDS):
            return False

        start = self.parse_date(role.get("start_date"))
        end = self.parse_date(role.get("end_date"))

        # pre-2023 evidence
        if end and end.year < 2023:
            return True
        if start and start.year < 2023:
            return True
        return False

    def detect_recent_langchain_only(self, candidate: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Eliminate only if:
        - candidate has recent LLM / LangChain / wrapper-stack experience
        - but lacks meaningful pre-2023 ML / IR / ranking / NLP / recommendation background
        """
        roles = candidate.get("career_history", [])
        if not roles:
            return False, ""

        recent_llm_roles = 0
        pre_llm_roles = 0

        for role in roles:
            if self.role_is_recent_llm_role(role):
                recent_llm_roles += 1
            if self.role_has_pre_llm_ml_evidence(role):
                pre_llm_roles += 1

        # Need at least one recent LLM role to even consider this disqualifier
        if recent_llm_roles == 0:
            return False, ""

        candidate_text = self.build_candidate_text(candidate)

        # If candidate has broader candidate-level pre-LLM evidence, do not eliminate
        candidate_has_pre_llm = self.any_keyword_hit(candidate_text, self.PRE_LLM_ML_KEYWORDS)

        # Conservative rule:
        # - recent LLM roles exist
        # - zero pre-2023 ML / IR evidence in role history
        # - weak candidate-level pre-LLM signals overall
        if recent_llm_roles > 0 and pre_llm_roles == 0 and not candidate_has_pre_llm:
            return True, "Recent LangChain / LLM-wrapper experience without meaningful pre-2023 ML / IR background"

        return False, ""

    # ---------------------------------------------------------------------
    # Main filter
    # ---------------------------------------------------------------------

    def filter_candidate(self, candidate: Dict[str, Any]) -> Tuple[bool, str]:
        # 1) Title chaser
        is_bad, reason = self.detect_title_chaser(candidate)
        if is_bad:
            self.stats["eliminated_title_chaser"] += 1
            return False, reason

        # 2) Pure research
        is_bad, reason = self.detect_pure_research(candidate)
        if is_bad:
            self.stats["eliminated_pure_research"] += 1
            return False, reason

        # 3) Pure CV / Speech / Robotics without IR
        is_bad, reason = self.detect_pure_domain_without_ir(candidate)
        if is_bad:
            self.stats["eliminated_pure_cv_speech_robotics"] += 1
            return False, reason

        # 4) Recent LangChain-only
        is_bad, reason = self.detect_recent_langchain_only(candidate)
        if is_bad:
            self.stats["eliminated_recent_langchain_only"] += 1
            return False, reason

        return True, "No disqualifiers detected"

    # ---------------------------------------------------------------------
    # IO
    # ---------------------------------------------------------------------

    def process_jsonl(self, input_path: str, output_path: str) -> Dict[str, int]:
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
                passes, _ = self.filter_candidate(candidate)
                if passes:
                    outfile.write(json.dumps(candidate, ensure_ascii=False) + "\n")
                    self.stats["total_output"] += 1

        return self.stats

    def print_report(self):
        print("\n" + "=" * 72)
        print("STAGE 3: DISQUALIFIER DETECTION (REFINED)")
        print("=" * 72)
        print(f"Total input:                          {self.stats['total_input']:>10,}")
        print(f"  - Title chasers:                    {self.stats['eliminated_title_chaser']:>10,}")
        print(f"  - Pure research:                    {self.stats['eliminated_pure_research']:>10,}")
        print(f"  - Pure CV/Speech/Robotics:          {self.stats['eliminated_pure_cv_speech_robotics']:>10,}")
        print(f"  - Recent LangChain-only:            {self.stats['eliminated_recent_langchain_only']:>10,}")
        print(f"Total output:                         {self.stats['total_output']:>10,}")

        eliminated = self.stats["total_input"] - self.stats["total_output"]
        rate = (eliminated / max(self.stats["total_input"], 1)) * 100
        print(f"Elimination rate:                     {rate:>9.1f}%")
        print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: Refined disqualifier detection."
    )
    parser.add_argument("--input", required=True, help="Input JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL")
    args = parser.parse_args()

    stage = Stage3Disqualifier()
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