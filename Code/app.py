import streamlit as st
import json
import csv
import io
import sys
from pathlib import Path

# Setup paths relative to this script
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
sys.path.append(str(CODE_DIR))

# Import pipeline stages from Code directory
try:
    from stage1 import Stage1DataQuality
    from stage2 import Stage2ExperienceRelevance
    from stage3 import Stage3Disqualifier
    from stage4 import Stage4RefinedHoneypot
    from stage5 import Stage5BehavioralPenalties
    from stage6 import Stage6AlignedScoring
except ImportError as e:
    st.error(f"Error loading pipeline stages: {e}")
    st.stop()

# Set page configuration with premium dark-leaning styling
st.set_page_config(
    page_title="Redrob Ranker Sandbox",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling rules
st.markdown("""
<style>
    .main {
        background-color: #0f111a;
        color: #e2e8f0;
    }
    .stButton>button {
        background-color: #4f46e5;
        color: white;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 600;
        border: none;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #4338ca;
        transform: translateY(-1px);
    }
    .card {
        background-color: #1e2235;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #2d3142;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔍 Redrob Candidate Discovery & Ranking Sandbox")
st.markdown("""
Welcome to the interactive sandbox environment for **Team - ME**.
This environment executes our end-to-end, 6-stage candidate discoveries pipeline locally on a small sample of candidates.
""")

# Sidebar settings
st.sidebar.image("https://redrob.io/static/images/logo.png", width=120)
st.sidebar.markdown("### Configuration")
st.sidebar.info("Compute Tier: Free CPU Sandbox\nRAM: Shared CPU")

# Data sourcing
st.markdown("### 1. Select Input Candidates Source")
source_option = st.radio(
    "Choose data source:",
    ("Use Pre-loaded 10k Sample Data (Data/candidates_10k.jsonl)", "Upload a candidates.jsonl file")
)

input_data = []

if source_option == "Use Pre-loaded 10k Sample Data (Data/candidates_10k.jsonl)":
    sample_path = PROJECT_ROOT / "Data" / "candidates_10k.jsonl"
    if sample_path.exists():
        try:
            with open(sample_path, "r", encoding="utf-8") as f:
                input_data = [line.strip() for line in f if line.strip()]
            st.success(f"Successfully loaded {len(input_data)} pre-loaded sample candidates.")
        except Exception as e:
            st.error(f"Error loading sample file: {e}")
    else:
        st.warning("Data/candidates_10k.jsonl not found. Please upload a candidates.jsonl instead.")

else:
    uploaded_file = st.file_uploader("Upload candidates.jsonl", type=["jsonl"])
    if uploaded_file is not None:
        content = uploaded_file.getvalue().decode("utf-8")
        input_data = [line.strip() for line in content.splitlines() if line.strip()]
        st.success(f"Successfully uploaded {len(input_data)} candidates.")

# Execution Trigger
if input_data:
    st.markdown("---")
    st.markdown("### 2. Execute Ranking Pipeline")
    if st.button("Run End-to-End Ranking"):
        # Setup paths
        reports_dir = PROJECT_ROOT / "Reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        sandbox_in = str(reports_dir / "sandbox_input.jsonl")
        stage1_out = str(reports_dir / "sandbox_stage1.jsonl")
        stage2_out = str(reports_dir / "sandbox_stage2.jsonl")
        stage3_out = str(reports_dir / "sandbox_stage3.jsonl")
        stage4_out = str(reports_dir / "sandbox_stage4.jsonl")
        stage5_out = str(reports_dir / "sandbox_stage5.jsonl")
        sandbox_csv = str(reports_dir / "sandbox_submission.csv")

        # Write uploaded data to local file for processing
        with open(sandbox_in, "w", encoding="utf-8") as f:
            for line in input_data:
                f.write(line + "\n")

        # Pipeline progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Stage 1
            status_text.text("Executing Stage 1: Data Quality Filter...")
            s1 = Stage1DataQuality()
            s1.process_jsonl(sandbox_in, stage1_out)
            progress_bar.progress(16)

            # Stage 2
            status_text.text("Executing Stage 2: Experience Relevance Matching...")
            s2 = Stage2ExperienceRelevance()
            s2.process_jsonl(stage1_out, stage2_out)
            progress_bar.progress(33)

            # Stage 3
            status_text.text("Executing Stage 3: Disqualifier Detection...")
            s3 = Stage3Disqualifier()
            s3.process_jsonl(stage2_out, stage3_out)
            progress_bar.progress(50)

            # Stage 4
            status_text.text("Executing Stage 4: Timeline Overlap & Skill Anomaly checks...")
            s4 = Stage4RefinedHoneypot()
            s4.process_jsonl(stage3_out, stage4_out)
            progress_bar.progress(66)

            # Stage 5
            status_text.text("Executing Stage 5: Behavioral Activity Check...")
            s5 = Stage5BehavioralPenalties()
            s5.process_jsonl(stage4_out, stage5_out)
            progress_bar.progress(83)

            # Stage 6
            status_text.text("Executing Stage 6: Final Weighted Composite Ranking...")
            s6 = Stage6AlignedScoring()
            s6.process_and_rank(stage5_out, sandbox_csv)
            progress_bar.progress(100)
            status_text.text("Pipeline Execution Completed!")

            st.success("Ranking successfully completed!")

            # -------------------------------------------------------------
            # Read and render ranked results
            # -------------------------------------------------------------
            results = []
            with open(sandbox_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    results.append(row)

            # Clean up files
            for file_path in [sandbox_in, stage1_out, stage2_out, stage3_out, stage4_out, stage5_out]:
                try:
                    Path(file_path).unlink(missing_ok=True)
                except Exception:
                    pass

            st.markdown("### 3. Final Ranked Recruiter Output")
            if results:
                # Format score values for display
                for row in results:
                    row['score'] = f"{float(row['score']):.6f}"

                st.dataframe(results, width='stretch')

                # Generate download options
                csv_io = io.StringIO()
                writer = csv.DictWriter(csv_io, fieldnames=['candidate_id', 'rank', 'score', 'reasoning'])
                writer.writeheader()
                writer.writerows(results)

                st.download_button(
                    label="📥 Download Ranked submission.csv",
                    data=csv_io.getvalue(),
                    file_name="submission.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No candidates satisfied the filter requirements for the ranking step.")

        except Exception as e:
            st.error(f"Pipeline error during execution: {e}")
            # Try to clean up input file if failed
            try:
                Path(sandbox_in).unlink(missing_ok=True)
            except Exception:
                pass
else:
    st.info("Please choose a candidate source above to enable pipeline execution.")
