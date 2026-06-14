"""
Synchronization pipeline — specification summary (no parsing).

Timeline rule: all cross-system alignment uses UTC internally at millisecond precision.

Sources
---------
* **E-Prime** (`Eprime.txt`): extract `SessionStartDateTimeUtc`, `FixationStart.OnsetTime`,
  `FixationStart.OnsetToOnsetTime`, `SixationOpenS.OnsetToOnsetTime` (spec spelling).
  Task window: start at fixation cross, duration = sum of the two OnsetToOnset fields.
* **EEG BrainVision** (`Task.ahdr`, `Task.amrk`, `Task.eeg`): recording start from first marker in
  `.amrk`; sampling rate from `SamplingInterval` (µs) in `.ahdr`. Local clock → UTC with project
  offset rule (Israel DST example: subtract 3 h for May recordings in spec narrative).
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
