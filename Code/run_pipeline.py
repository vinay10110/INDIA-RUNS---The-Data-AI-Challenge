#!/usr/bin/env python3
"""
Master Pipeline Orchestrator for the Redrob Data & AI Challenge

Executes all 6 stages in sequence:
1. Data Quality (Hard Filters)
2. Experience Relevance Scoring (JD-Aligned Gate)
3. Disqualifier Detection
4. Suspicious Honeypot Pattern Detection
5. Behavioral Signals (Soft Penalties)
6. Final Composite Ranking

At the end, it automatically validates the submission file format.
"""

import sys
import argparse
import time
from pathlib import Path

# Setup paths relative to this script (located in Code/)
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
sys.path.append(str(CODE_DIR))

try:
    from stage1 import Stage1DataQuality
    from stage2 import Stage2ExperienceRelevance
    from stage3 import Stage3Disqualifier
    from stage4 import Stage4RefinedHoneypot
    from stage5 import Stage5BehavioralPenalties
    from stage6 import Stage6AlignedScoring
    from validate_submission import validate_submission
except ImportError as e:
    print(f"Error importing pipeline stages: {e}")
    print("Please ensure the 'Code' directory contains all stage scripts.")
    sys.exit(1)


def run_pipeline(input_path: str, output_path: str):
    print("=" * 80)
    print("STARTING REDROB CANDIDATE RANKING PIPELINE")
    print("=" * 80)
    print(f"Input Candidates: {input_path}")
    print(f"Output Submission: {output_path}")
    
    # Ensure Reports and output directories exist
    reports_dir = PROJECT_ROOT / "Reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Intermediate files in the Reports folder
    stage1_out = str(reports_dir / "stage1_output.jsonl")
    stage2_out = str(reports_dir / "stage2_output.jsonl")
    stage3_out = str(reports_dir / "stage3_output.jsonl")
    stage4_out = str(reports_dir / "stage4_output.jsonl")
    stage5_out = str(reports_dir / "stage5_output.jsonl")

    start_time = time.time()

    # -------------------------------------------------------------------------
    # STAGE 1: Data Quality
    # -------------------------------------------------------------------------
    print("\n[1/6] Running Stage 1: Data Quality (Hard Filters)...")
    s1_start = time.time()
    s1 = Stage1DataQuality()
    s1.process_jsonl(input_path, stage1_out)
    s1.print_report()
    print(f"Stage 1 completed in {time.time() - s1_start:.2f}s")

    # -------------------------------------------------------------------------
    # STAGE 2: Experience Relevance Scoring
    # -------------------------------------------------------------------------
    print("\n[2/6] Running Stage 2: Experience Relevance Scoring...")
    s2_start = time.time()
    s2 = Stage2ExperienceRelevance()
    s2.process_jsonl(stage1_out, stage2_out)
    s2.print_report()
    print(f"Stage 2 completed in {time.time() - s2_start:.2f}s")

    # -------------------------------------------------------------------------
    # STAGE 3: Disqualifier Detection
    # -------------------------------------------------------------------------
    print("\n[3/6] Running Stage 3: Disqualifier Detection...")
    s3_start = time.time()
    s3 = Stage3Disqualifier()
    s3.process_jsonl(stage2_out, stage3_out)
    s3.print_report()
    print(f"Stage 3 completed in {time.time() - s3_start:.2f}s")

    # -------------------------------------------------------------------------
    # STAGE 4: Suspicious Honeypot Pattern Detection
    # -------------------------------------------------------------------------
    print("\n[4/6] Running Stage 4: Suspicious Honeypot Pattern Detection...")
    s4_start = time.time()
    s4 = Stage4RefinedHoneypot()
    s4.process_jsonl(stage3_out, stage4_out)
    s4.print_report()
    print(f"Stage 4 completed in {time.time() - s4_start:.2f}s")

    # -------------------------------------------------------------------------
    # STAGE 5: Behavioral Signals (Soft Penalties)
    # -------------------------------------------------------------------------
    print("\n[5/6] Running Stage 5: Behavioral Signals (Soft Penalties)...")
    s5_start = time.time()
    s5 = Stage5BehavioralPenalties()
    s5.process_jsonl(stage4_out, stage5_out)
    s5.print_report()
    print(f"Stage 5 completed in {time.time() - s5_start:.2f}s")

    # -------------------------------------------------------------------------
    # STAGE 6: Final Ranking
    # -------------------------------------------------------------------------
    print("\n[6/6] Running Stage 6: Final Composite Ranking...")
    s6_start = time.time()
    s6 = Stage6AlignedScoring()
    s6.process_and_rank(stage5_out, output_path)
    print(f"Stage 6 completed in {time.time() - s6_start:.2f}s")

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"PIPELINE EXECUTION COMPLETE IN {total_time:.2f}s")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # VALIDATION
    # -------------------------------------------------------------------------
    print("\nValidating output submission CSV...")
    errors = validate_submission(output_path)
    if errors:
        print("\n[WARNING] Submission validation failed with the following issues:")
        for err in errors:
            print(f"  - {err}")
        return False
    else:
        print("\n[SUCCESS] Submission is valid and ready for upload!")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Run the end-to-end candidate ranking pipeline."
    )
    parser.add_argument(
        "--input",
        default="Data/candidates.jsonl",
        help="Path to input candidates.jsonl file (default: Data/candidates.jsonl)"
    )
    parser.add_argument(
        "--output",
        default="Reports/submission_refined.csv",
        help="Path to output submission.csv file (default: Reports/submission_refined.csv)"
    )
    args = parser.parse_args()

    # Resolve relative paths against the project root for safety
    input_resolved = str(PROJECT_ROOT / args.input) if not Path(args.input).is_absolute() else args.input
    output_resolved = str(PROJECT_ROOT / args.output) if not Path(args.output).is_absolute() else args.output

    success = run_pipeline(input_resolved, output_resolved)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
