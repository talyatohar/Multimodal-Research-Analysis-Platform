"""Central `st.session_state` keys for cross-page navigation (UI shell)."""

COHORT_ROWS = "cohort_rows"
SELECTED_PARTICIPANT_ID = "selected_participant_id"
UI_TASK_LEVEL_UNLOCKED = "ui_task_level_unlocked"
UI_EVENT_LEVEL_UNLOCKED = "ui_event_level_unlocked"
UI_TASK_LEVEL_GENERATED = "ui_task_level_tables_stub"
UI_EVENT_LEVEL_GENERATED = "ui_event_level_tables_stub"

# Analyse page: cached placeholder tables (stable DataFrames; refresh only on button actions)
ANALYSE_LAST_PID = "analyse_context_participant_id"
ANALYSE_TASK_LEVEL_DFS = "analyse_task_level_placeholder_dfs"
ANALYSE_TASK_LEVEL_META = "analyse_task_level_meta"
ANALYSE_COMPARE_DF = "analyse_compare_placeholder_df"
ANALYSE_COMPARE_META = "analyse_compare_meta"
ANALYSE_EVENT_META = "analyse_event_meta"
ANALYSE_EVENT_CONTEXT = "analyse_event_context"
ANALYSE_EVENT_SUMMARY_COL_CONFIG = "analyse_event_summary_col_config"
ANALYSE_EVENT_EEG_COL_CONFIG = "analyse_event_eeg_col_config"

# Event-level display cache (stable across Streamlit reruns)
EVENT_LEVEL_RESULTS = "event_level_results"
SELECTED_EVENT_TYPE = "selected_event_type"
LOADED_EVENT_SUMMARY_DF = "loaded_event_summary_df"
LOADED_EVENT_DISTRIBUTION_DF = "loaded_event_distribution_df"
EVENT_TABLES_LOADED = "event_tables_loaded"
