"""Main page — multimodal research platform overview."""

import streamlit as st

from shared.cohort_state import ensure_cohort
from shared.ui import apply_theme, configure_page, sidebar_brand

configure_page("Home")
apply_theme()
sidebar_brand()

st.caption("Multimodal neuroscience")
st.title("Multimodal Research Analysis Platform")
st.markdown(
    "An integrated platform for synchronizing, processing and analyzing behavioral, eye-tracking, "
    "physiological and EEG data collected during cognitive experiments."
)

with st.container():
    st.markdown(
        "This platform was developed to support multimodal neuroscience research by combining "
        "eye-tracking, physiological monitoring and EEG recordings within a unified analysis environment."
    )
    st.markdown(
        "The system enables synchronization of data acquired from independent recording systems, "
        "automated preprocessing pipelines, quality-control procedures, feature extraction and "
        "event-centered analysis. It is designed for complex cognitive experiments involving reading, "
        "attention, learning and behavioral performance."
    )

st.header("Integrated Acquisition Systems")

modality_cols = st.columns(2)
modality_cards = [
    (
        "👁 Eye Tracking",
        "Tobii Pro X3-120",
        [
            "Gaze behavior analysis",
            "Fixations & saccades",
            "Regression detection",
            "Pupil dynamics",
            "Data quality monitoring",
        ],
    ),
    (
        "🧠 EEG",
        "64-channel BrainVision",
        [
            "Neural activity analysis",
            "Spectral power analysis",
            "Connectivity (PLV) analysis",
            "Event-centered neural measurements",
            "Automated preprocessing & QC",
        ],
    ),
    (
        "❤️ Physiological Monitoring",
        "Corsano wearable",
        [
            "Heart rate (BPM)",
            "Heart rate variability (RMSSD, SI)",
            "Respiration rate",
            "Motion monitoring (ACC)",
            "Task-level physiological features",
        ],
    ),
    (
        "🖥 Experimental Control",
        "E-Prime",
        [
            "Behavioral task presentation",
            "Reading & cognitive tasks",
            "Comprehension assessment",
            "Synchronization reference source",
            "UTC task-window extraction",
        ],
    ),
]

for idx, (title, system, bullets) in enumerate(modality_cards):
    with modality_cols[idx % 2]:
        with st.container():
            st.subheader(title)
            st.caption(system)
            st.markdown("\n".join(f"- {item}" for item in bullets))

st.divider()
st.header("Cross-System Synchronization")

sync_left, sync_right = st.columns([3, 2])
with sync_left:
    st.markdown(
        "Data is acquired from **multiple independent systems** running on different computers. "
        "The platform automatically aligns timestamps from:"
    )
    st.markdown(
        "- **E-Prime** — experimental control & task timing  \n"
        "- **Eye Tracking** — Tobii Pro Lab exports  \n"
        "- **EEG** — BrainVision recordings  \n"
        "- **Physiological** — Corsano wrist sensor streams"
    )
    st.info(
        "Synchronization uses UTC-based timelines and recording metadata, enabling accurate "
        "task segmentation and multimodal event alignment.",
        icon="🔗",
    )
with sync_right:
    with st.container():
        st.caption("Processing workflow")
        st.markdown(
            """
            ```
            E-Prime
               ↓
            Task Window Extraction
               ↓
            Eye Tracking Segmentation
               ↓
            EEG Segmentation
               ↓
            Physiological Segmentation
               ↓
            Unified Analysis
            ```
            """
        )

st.divider()
st.header("Automated Analysis Pipeline")

pipeline_steps = [
    ("Data Import", "Register participants and ingest modality exports into a structured database."),
    ("Synchronization", "Derive UTC task windows and align independent recording timelines."),
    ("Signal Quality Validation", "Verify data integrity, validity coding and recording completeness."),
    ("Task Segmentation", "Clip eye-tracking, EEG and physiology to the experimental task window."),
    ("Eye Tracking Analysis", "Extract fixation, saccade, regression and pupil task-level features."),
    ("Physiological Analysis", "Compute heart rate, HRV, respiration and motion metrics per task."),
    ("EEG Preprocessing", "Filter, reference and prepare neural signals for feature extraction."),
    ("Feature Extraction", "Generate spectral power, connectivity and baseline-normalized features."),
    ("Event Detection", "Identify long fixations, regressions and tracking-loss bursts."),
    ("Multimodal Integration", "Align events with EEG epochs and produce integrated summaries."),
]

pipe_cols = st.columns(2)
for idx, (step_name, step_desc) in enumerate(pipeline_steps):
    with pipe_cols[idx % 2]:
        with st.container():
            st.markdown(f"**{idx + 1}. {step_name}**")
            st.caption(step_desc)

st.divider()
st.header("Available Analyses")

analysis_cols = st.columns(3)
with analysis_cols[0]:
    st.success("**Task-Level Analysis**")
    st.markdown(
        "- Eye tracking features  \n"
        "- Physiological features  \n"
        "- EEG band-power & connectivity  \n"
        "- Quality control metrics"
    )
with analysis_cols[1]:
    st.success("**Event-Level Analysis**")
    st.markdown(
        "- Long fixation events  \n"
        "- Regression events  \n"
        "- EyesNotFound bursts  \n"
        "- Event-centered EEG analysis"
    )
with analysis_cols[2]:
    st.success("**Task Comparison**")
    st.markdown(
        "- Compare experimental conditions  \n"
        "- Compare features across tasks  \n"
        "- Visualize participant-specific differences  \n"
        "- Export comparison tables"
    )

st.divider()
with st.container():
    st.subheader("Why is this platform unique?")
    st.markdown(
        "The platform combines multiple acquisition technologies that operate independently and "
        "synchronizes them into a unified analysis framework. Beyond multimodal alignment, it includes "
        "automated preprocessing, signal validation, event detection, feature extraction and cross-modal "
        "integration — enabling researchers to investigate relationships between behavior, physiology, "
        "eye movements and neural activity within a single workflow."
    )

st.divider()
st.header("Get started")

nav_cols = st.columns(3)
with nav_cols[0]:
    with st.container():
        st.subheader("📤 Upload participant")
        st.caption("Register participants, ingest modality exports and run synchronization pipelines.")
with nav_cols[1]:
    with st.container():
        st.subheader("📊 Analyse participant")
        st.caption("Generate task-level tables, compare conditions and run event-level EEG analysis.")
with nav_cols[2]:
    with st.container():
        st.subheader("🗂 Participants database")
        st.caption("Search the cohort registry, inspect registered tasks and open analysis workflows.")

st.caption("Navigate via the sidebar: **Upload** → **Analyse** → **Database**.")

metric_cols = st.columns([2, 1, 1])
with metric_cols[1]:
    st.metric("Participants registered", value=len(ensure_cohort()))
with metric_cols[2]:
    st.metric("Integrated modalities", value="4")
