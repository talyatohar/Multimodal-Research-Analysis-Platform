# Multimodal Research Analysis Platform

A platform for synchronizing, processing and analyzing EEG, eye-tracking, physiological and behavioral recordings that were acquired on independent devices running on separate computers.

🚀 **Live Demo:** https://multimodal-research-analysis-platform-lw2epaxrupn4bbsrxqldse.streamlit.app/

*The online demo runs on synthetic data only and is not meant for real participant data. Real analyses run locally on the lab PC.*

---

## Quick start

```bash
conda env create -f environment.yml
conda activate lazyeye311
streamlit run app.py
```

Or install into an existing Python 3.11 environment with `pip install -r requirements.txt`. Tested with Python 3.11, Streamlit 1.64.0, pandas 2.3.3, NumPy 2.4.6, SciPy 1.17.1 and openpyxl 3.1.5 (pinned in `requirements.txt`). `environment.yml` also installs pytest; run the tests with `pytest`.

The app opens at `http://localhost:8501`. The step-by-step guide for researchers (installation, input files, the session EEG clock offset, upload, participant settings, analyses, output files and troubleshooting) is the [User Manual](docs/manual/User_Manual_2026-11-23.docx).

**Data:** no real participant data is included in this repository, and `.gitignore` keeps participant folders, raw recordings and result tables out of it. To try the app, use the synthetic demo participant `123456789` (see [Demo dataset](#demo-dataset)).

---

## Overview

The platform was developed at the Educational Neuroimaging Group to study reading in Hebrew-reading children and adolescents (ages 6–16), with the intended application of comparing children with left- and right-eye amblyopia with typically developing peers.

During a session, E-Prime presents the tasks, a Tobii eye tracker records gaze, a 64-channel BrainVision EEG system records brain activity on a separate computer, and a Corsano wristband records heart rate, respiration, heart-rate variability and movement. The clocks of these systems are not synchronized. The platform:

1. places every task window defined by the E-Prime log on the clock of each recording;
2. cuts and preprocesses each modality within those windows;
3. extracts a documented set of features for each whole task (task level) and around individual reading events (event level);
4. reports quality-control measures alongside the features;
5. shows the results as tables in a browser interface and saves them as Excel files.

Everything is done with buttons in the browser; no programming is needed to run an analysis.

---

## Acquisition systems and input files

| Modality | System | Input file(s) |
|---|---|---|
| Task timing | E-Prime | `Eprime.txt`, the E-Prime log of each task |
| Eye tracking | Tobii Pro X3-120, 120 Hz, exported from Tobii Pro Lab with millisecond timestamps | `EyeTracking.xlsx` (one or more exports per participant; the platform finds the recording that covers each task) |
| EEG | 64-channel BrainVision system, 500 Hz | BrainVision Recorder BVRF set of each task: `.bvrh`, `.bvrd`, `.bvrm` (required) and `.bvri` |
| Physiology | Corsano wristband | `acc.xlsx`, `activity.xlsx`, `heart_rate_variability.xlsx` |
| Behavior | Reading comprehension questions after each reading task | Score in percent, typed in on the upload page |

---

## Experimental tasks

* Oral Reading – Erased Text
* Silent Reading – Erased Text
* Silent Reading – Erased Text with Metronome
* Silent Reading – Static Text with Metronome
* Silent Reading – Static Text
* Resting state (eyes open and eyes closed); the eyes-open part is the baseline for all change-from-baseline values

---

## Synchronization

* **Task windows** come from the E-Prime log of each task. For reading tasks an optional manual reading end, read off the Tobii screen recording, defines a second, shorter eye-tracking window (the Manual Reading Window). The manual window applies to eye tracking only; EEG, physiology and event-level analysis always use the E-Prime window.
* **Eye tracking** shares the clock of the E-Prime computer, so its samples are placed directly on the E-Prime time base.
* **EEG** is converted with a session clock offset (EEG time minus E-Prime time) that the user enters at upload. It is required whenever EEG files are uploaded and has no default. For the Resting state the platform derives the alignment from the EEG trigger markers sent at the eyes-open and eyes-closed onsets whenever they are present.
* **Wristband** timestamps follow network time, which matches the EEG clock, so each task window is shifted by that task's EEG offset before the physiological samples are selected.

---

## Processing and features

### Eye tracking
* Samples are cut to the task window with a 0.5 s buffer on each side, so that fixations crossing a window edge are classified from complete data.
* The eye used for gaze analysis is a participant setting. **Average** (default) uses the fixation classification exported by Tobii Pro Lab. **Left** or **Right** re-classifies fixations from that eye with an I-VT filter that follows the Tobii I-VT fixation filter (75 ms gap fill-in, 3-sample median filter, 20 ms velocity window, 30°/s threshold, 60 ms minimum fixation, merging of fixations less than 75 ms and less than the merge angle apart; merge angle 0.5° by default, adjustable).
* A transition is the interval between two consecutive fixations; transitions with tracking loss are not used. A saccade is a transition of at most 100 ms.
* A regression is a rightward movement on the same line (Hebrew is read right to left) that exceeds the regression threshold. The line-change and regression thresholds are estimated from each participant's data in each task, fall back to standard values when the data do not allow an estimate, and can be overridden manually.
* Features: fixations, saccades, tracking loss (EyesNotFound), pupil diameter and regressions.
* Resting-state eye tracking is excluded by design (no reading, and half of it is recorded with eyes closed).

### EEG
* Zero-phase band-pass filter 0.5–40 Hz (fourth-order Butterworth), 50 Hz notch filter, average reference.
* Power spectral density with Welch's method (1 s Hann windows, 50% overlap).
* Regions of interest: frontal (theta power) Fz, F3, F4; occipital Oz, O1, O2, POz, PO3, PO4, PO7, PO8, Iz; temporal T7, T8, TP7, TP8, TP9, TP10, P7, P8; frontal (phase locking) AF4, Fz, F1–F8, FC1–FC6, FT7–FT10. The blink-dominated prefrontal channels Fp1, Fp2, AF7 and AF8 are excluded from the frontal ROIs because no ocular artifact correction is applied.
* Features: frontal theta power (4–8 Hz), occipital alpha power (8–12 Hz), the frontal theta to occipital alpha ratio, and theta-band phase-locking value (PLV) between occipital–temporal, temporal–frontal and occipital–frontal regions, each with its variability and change from the eyes-open baseline.
* The eyes-closed part of the Resting state is reported separately (Table 3b).

### Physiology
* Low-quality samples are excluded (heart rate with a device quality code below 3 on the 0–4 scale, respiration with a code of 0).
* Features: mean heart rate, respiration rate, RMSSD and Stress Index, with variability and change from the Resting state baseline, and accelerometer-based motion measures used for quality control.

### Quality control
Valid eye-tracking samples per eye, motion measures from the accelerometer, and the reading comprehension score. Quality-control measures are reported next to the features; no recording is excluded automatically.

### Event-level analysis
Three event types are detected from the eye-tracking data of each reading task:
* **Long Fixation Events** – fixations longer than the participant's mean fixation duration in that task;
* **Regression Events** – transitions classified as regressions, one event per transition;
* **EyesNotFound Bursts** – continuous tracking loss of at least 100 ms (mostly blinks).

For each event, EEG is extracted from 500 ms before onset to 500 ms after offset. Optionally (a participant setting), windows that contain a blink (tracking loss of at least 80 ms, or an Fp1/Fp2 blink peak larger than 5 × the median absolute deviation of the 0.5–10 Hz signal) are excluded from the long-fixation and regression EEG. The EEG around EyesNotFound Bursts is always marked as blink-contaminated and not for interpretation.

---

## Outputs

| Output | Content |
|---|---|
| Table 1 · Eye Tracking Data | Fixations, saccades, tracking loss, pupil diameter and regressions, for the E-Prime window or the Manual Reading Window |
| Table 2 · Physiological Data | Heart rate, respiration, RMSSD and Stress Index, each also as change from baseline |
| Table 3 · EEG Data | Theta and alpha power, their ratio and theta PLV, with variability and change from baseline |
| Table 3b · EEG Data (eyes closed) | Resting state only, the eyes-closed EEG values |
| Table 4 · Quality Control | Valid eye-tracking samples per eye, motion measures, reading comprehension score |
| Task comparison | Chosen features of several tasks side by side, saved as Excel with a metadata file |
| Event summary | Number of events, mean duration and duration variability per event type |
| Event-level EEG distribution | Mean, standard deviation and variance of six EEG measures over all events of a type |

A settings caption under the task-level tables records the eye, merge angle and thresholds (with their source: auto, fallback or manual) behind the numbers. All outputs are saved under `database/participants/participant_<ID>/`, one folder per task. The exact definition, unit and rationale of every field are documented in the Project Book that accompanies the platform; clicking a table row in the app shows a short explanation with the unit.

---

## Interface

| Page | Purpose |
|---|---|
| Home (`app.py`) | Overview of the platform |
| Upload Participant | Register a participant once with all files of the session; enter the EEG clock offset, the task blocks and the optional manual reading end. Existing participants are never overwritten |
| Analyse Participant | Participant settings (eye, merge angle, blink exclusion), task-level analysis, task comparison and event-level analysis |
| Participants Database | Search and filter participants, open one for analysis, or delete a participant and all its outputs |

---

## Repository structure

```text
app.py                      Home page (Streamlit entry point)
pages/                      Upload, Analyse and Participants Database pages
backend/                    Processing: synchronization (sync/), EEG (eeg/),
                            eye tracking and event-level analysis (analysis/),
                            participant registry and storage
domain/                     Shared definitions: tasks, features, storage layout, IDs
shared/                     User-interface helpers
tests/                      Unit tests on synthetic data (pytest)
database/                   participants_table.xlsx and the demo participant
docs/manual/                User Manual and its screenshots
requirements.txt            Pinned Python dependencies
environment.yml             Conda environment (lazyeye311)
```

---

## Demo dataset

`database/participants/participant_123456789/` is a complete synthetic participant (registered as `DEMO`, group Control) with three tasks: Oral Reading – Erased Text, Silent Reading – Static Text with Metronome, and Resting state with eyes-open and eyes-closed blocks. It contains synthetic E-Prime logs, a Tobii export, BrainVision EEG files and wristband exports, together with the outputs the platform produced from them. All values were generated artificially and contain no information about a real person.

The demo participant is the source of the example task-level and event-level outputs shown in the accompanying paper (Figures 4 and 5) and of all screenshots in the User Manual. It appears on the Analyse Participant and Participants Database pages as soon as the app starts.

---

## Limitations

* The pipeline has been applied to one typically developing participant so far; the group study is planned.
* No ocular artifact correction is applied to the EEG; the prefrontal channels are excluded from the ROIs and blink windows can be excluded at the event level instead.
* The session EEG clock offset must be determined for every session; the platform does not estimate it automatically for reading tasks.

---

## Technologies

Python 3.11 · Streamlit · pandas · NumPy · SciPy · openpyxl · pytest

---

## Author

**Talya Tohar**

Faculty of Biomedical Engineering
Technion – Israel Institute of Technology

Developed at the Educational Neuroimaging Group.

---

## Citation

Citation: [paper reference – to be added]
