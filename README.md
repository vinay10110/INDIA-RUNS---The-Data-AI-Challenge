# Intelligent Candidate Discovery & Ranking Pipeline

End-to-end, multi-stage hybrid ranking system for the **Redrob Hackathon v4** matching candidates against a Senior Machine Learning Engineer (Search, Retrieval, & Recommender Systems) role.

Built by **Team - ME** (Bandla Sai Vinay Chakravarthi).

---

## 🚀 Quick Start Guide

This project is built entirely on the Python Standard Library (with the exception of `streamlit` for the visual sandbox). It requires zero configuration, has no heavy third-party model installations, and runs locally on CPU in under 4 minutes.

### 1. Installation
Clone the repository and install the UI dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Local Streamlit Sandbox
To launch the interactive recruiter web interface:
```bash
python -m streamlit run Code/app.py
```
*Once running, you can click the pre-loaded 10k dataset option to execute the sandbox instantly.*

### 3. Run the Command-Line CLI Pipeline
To run the full end-to-end pipeline on the 100k candidate pool:
```bash
python Code/run_pipeline.py --input ./Data/candidates.jsonl --output ./submission.csv
```

---

## 📁 Repository Structure

```
├── Code/
│   ├── app.py                     # Streamlit web application sandbox
│   ├── run_pipeline.py            # Master orchestrator script (CLI entrypoint)
│   ├── stage1.py                  # Data Quality hard filters
│   ├── stage2.py                  # JD-aligned Relevance Scoring (Word-boundary matching)
│   ├── stage3.py                  # Disqualifier Detection (Research-only, CV-only, title-chasing)
│   ├── stage4.py                  # Suspicious Honeypot & Timeline Overlap Detection
│   ├── stage5.py                  # Behavioral Platform Engagement Penalties
│   ├── stage6.py                  # Composite Ranking, 0-1 Scaling, and Recruiter Reasoning
│   └── validate_submission.py     # Auto-validator for formatting compliance
├── Data/
│   ├── candidates.jsonl           # Complete 100k candidate dataset (Ignored in Git)
│   ├── candidates_10k.jsonl       # 10k candidate sample dataset (Tracked for Streamlit)
│   ├── candidate_schema.json      # JSON Schema for candidate objects
│   └── sample_submission.csv      # Formatting sample provided by organisers
├── Reports/                       # Output folder for logs, intermediate steps, and CSVs (Ignored in Git)
├── .gitignore                     # Configured to exclude heavy files (>100MB) and python caches
├── requirements.txt               # App dependencies (streamlit)
└── submission_metadata.yaml       # Final portal submission metadata
```

---

## 🔍 Core Pipeline Architecture

The pipeline processes candidate data sequentially through 6 specialized stages. Each stage is designed to filter noise early, refine candidate signals, and calculate a trusted final score.

```
[100k Pool] ──► Stage 1 (Format & Date Validation)
                  ──► Stage 2 (Keyword Matching & Gate >= 28)
                        ──► Stage 3 (Negative Filters & Disqualifiers)
                              ──► Stage 4 (Overlap & Skill Inflation Analysis)
                                    ──► Stage 5 (Availability & Inactivity Filter)
                                          ──► Stage 6 (Weighted Scoring, 0-1 Scale, Reasoning) ──► [Top 100 Output]
```

---

## 🛠️ Stage-by-Stage Technical Breakdown

### Stage 1: Data Quality (Hard Filters)
*   **Purpose**: Remove corrupted profiles, missing core fields, and invalid identifiers.
*   **Logic**:
    *   Ensures `candidate_id` conforms to the pattern `CAND_XXXXXXX`.
    *   Validates that essential dictionary fields (`profile`, `experience`, `skills`) exist.
    *   Performs chronological date syntax validation on career history. If dates are unparseable, severely overlapping within the same company, or list start dates in the future (beyond the dataset limit), the profile is marked corrupt and removed.
*   **Elimination Rate**: **18.9%** (filters out 18,865 corrupted records from the 100k pool).

---

### Stage 2: JD-Aligned Experience Relevance Scoring
*   **Purpose**: Filter candidates based on technical keywords matching search, ranking, recommendation, production, and evaluation systems.
*   **Logic Improvements**:
    *   **Word Boundary Protection (`\b`)**: Standard substring searches create false positives (e.g. the word `"first"` or `"confirm"` matching the search acronym `"ir"`). We implemented regular expression matching with word boundaries (`\b{keyword}\b`) for all short terms ($\le 3$ chars) such as `ir`, `nlp`, `ann`, `map`.
    *   **Penalization Over Elimination**: Candidates with background experience *only* at IT consulting/services firms (TCS, Infosys, Wipro, Accenture, Cognizant, etc.) are down-weighted by a `-8.0` score adjustment in core fit rather than being instantly disqualified, allowing rare, high-quality candidates to stay in scope.
    *   **Gating Threshold**: Candidates must score $\ge 28.0$ points across categories to proceed, acting as a relevance filter.
*   **Elimination Rate**: **64.4%** (reduces noise significantly by discarding 52,286 low-relevance candidates).

---

### Stage 3: Disqualifier Detection
*   **Purpose**: Automatically eliminate profiles showing negative matching signals explicit in the Job Description.
*   **Rules Applied**:
    *   **Title Chasers**: Candidates switching companies every 1.5 years or less to optimize for titles (`Senior` ──► `Staff` ──► `Lead`).
    *   **Pure Researchers**: Profiles consisting solely of academic publications and research-only roles with no record of deploying model files to production.
    *   **Non-NLP/IR Domains**: Candidates whose career experience focuses entirely on Computer Vision, Speech, or Robotics.
    *   **Framework Enthusiasts**: Candidates whose recent AI experience (under 12 months) consists entirely of LLM-wrapper scripts (e.g., LangChain) without pre-LLM era ML fundamentals.

---

### Stage 4: Suspicious Honeypot & Timeline Overlap Detection
*   **Purpose**: Protect the ranking from fraudulent/inflated profiles and honeypots.
*   **Rules Applied**:
    *   **Timeline Overlaps**: Flags candidates claiming multiple simultaneous full-time positions at different companies.
    *   **Skill Inflation**: Detects candidates listing "expert" proficiency in multiple skills with 0 years used, or abnormally high skill endorsement velocities (endorsements per month > 30).
*   **Nuanced Design (Soft Penalties)**:
    *   Instead of auto-eliminating candidates for a single overlap (which could be legitimate side-businesses, part-time teaching, or advisory roles), the threshold is raised to a composite score of `4.0`.
    *   Mild overlaps are kept in the pipeline but penalized in Stage 6 (deducting `-3.0` points for mild overlaps, `-4.0` for skill inflation, and up to `-5.0` proportional to overlap duration), pushing high-integrity timelines to the top.

---

### Stage 5: Behavioral Signals
*   **Purpose**: Incorporate platform engagement to evaluate availability.
*   **Logic**:
    *   A candidate with a perfect profile who hasn't logged in for 6 months is likely unavailable.
    *   Computes penalties based on login recency, profile completeness, connection acceptance rate, and recruiter response speed.
    *   Saves a `_behavioral_penalty` scalar to subtract in Stage 6, while hard-disqualifying extremely inactive profiles (e.g., response rate < 5% or inactive for > 180 days).

---

### Stage 6: Composite Scoring & Final Ranking
*   **Purpose**: Synthesize all features into a final normalized score and generate human-readable reasonings.
*   **The Scoring Weights**:
    *   **Core Technical Fit (35%)**: Emphasizes retrieval, ranking, and production signals blended with the Stage 2 Relevance Gate.
    *   **Retrieval & Search Focus (20%)**: Emphasizes embedding, FAISS, Milvus, indexing, and vector search.
    *   **Production ML Experience (15%)**: Evaluates systems deployed to real users.
    *   **Platform Engagement (10%)**: Deducts points based on behavioral signals.
    *   **Experience & Title Progression (10%)**: Emphasizes total Years of Experience (YoE) and rewards title progression (promotions).
    *   **Technical Skills (7%)**: Evaluates skill count density and average proficiency.
    *   **Education (3%)**: Based on university tiers (Tier 1: 80 pts, Tier 2: 60 pts, etc.).
*   **Shipper Adjustment**: Adds up to `+5.0` points for profiles containing "shipping" keywords (*deployed*, *production*, *scaled*) and subtracts up to `-5.0` points for academic-heavy research profiles (*published*, *paper*, *thesis*).
*   **0-1 Score Scaling**:
    *   The total score is divided by `100.0` and clamped strictly between `0.0` and `1.0`.
    *   Sorted using $6$ decimal places of precision (`-round(final_score, 6)`) to ensure tie-breakers (`candidate_id` ascending) run deterministically.
*   **Dynamic Recruiter Reasoning (Stage 4 Check Compliant)**:
    *   Rather than utilizing a repetitive template, Stage 6 extracts the candidate's actual title, YoE, and matching skills (e.g., `Qdrant`, `pgvector`, `OpenSearch`).
    *   Flags warning signs (like service-firm backgrounds or timeline overlaps) explicitly.
    *   Rotates between 3 different sentence styles to avoid duplication and satisfy the variation check.

---

## 📈 Pipeline Performance Metrics

*   **Runtime**: **191.74 seconds** (End-to-End on 100k pool using standard CPU).
*   **Memory Footprint**: **~250 MB** RAM.
*   **Honeypot Rate in Top 100**: **0%** (Exactly 0 flagged profiles reached the top 100).
*   **Compliance Validation**: Passes `validate_submission.py` format checks with `[SUCCESS]`.
