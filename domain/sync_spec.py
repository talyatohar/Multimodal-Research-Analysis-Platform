"""
Synchronization pipeline — specification summary (no parsing).

Timeline rule: all cross-system alignment uses UTC internally at millisecond precision.

Sources
---------
* **E-Prime** (`Eprime.txt`): extract `SessionStartDateTimeUtc`, `FixationStart.OnsetTime`,
  `FixationStart.OnsetToOnsetTime`, `SixationOpenS.OnsetToOnsetTime` (spec spelling).
  Task window: start at fixation cross, duration = sum of the two OnsetToOnset fields.
* **EEG BrainVision BVRF** (`.bvrh`, `.bvrd`, `.bvrm`): recording start/end from the `.bvrm` marker
  DateTime (ISO with UTC offset); sampling rate from the `.bvrh` header. A per-session clock offset
  (EEG PC − E-Prime PC) is applied; Resting state derives it from its EyesOpen/EyesClose triggers.
* **Tobii Pro Lab export** (`EyeTracking.xlsx`): split each workbook by `Recording name`, then match
  the task UTC window inside each recording using that recording's start UTC + `Recording timestamp [ms]`.
  Multiple Tobii files and multiple recordings per file may be searched.
* **Corsano** (`activity.xlsx`, `heart_rate_variability.xlsx`, `acc.xlsx`): use Unix `timestamp`
  (ms) only — not the human-readable `date` field — then clip to E-Prime task UTC window.

Downstream segmentation never mutates raw uploads.
"""

EPRIME_FIELDS = (
    "SessionStartDateTimeUtc",
    "FixationStart.OnsetTime",
    "FixationStart.OnsetToOnsetTime",
    "SixationOpenS.OnsetToOnsetTime",
)
