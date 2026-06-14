# Multimodal Research Analysis Platform

A unified framework for synchronizing, processing, and analyzing multimodal experimental data acquired from independent recording systems.

🚀 **Live Demo:** [Add Vercel Link Here]

---

## Overview

The Multimodal Research Analysis Platform was developed to support cognitive neuroscience research by integrating behavioral, eye-tracking, physiological, and EEG recordings within a single analysis environment.

The platform provides automated synchronization, preprocessing, quality control, feature extraction, and event-centered analysis for data collected from independent acquisition systems. It enables researchers to efficiently analyze complex experimental datasets through a unified workflow and interactive user interface.

Developed at the Educational Neuroimaging Group, the platform was designed to support real-world experimental studies involving multimodal recordings and cognitive task performance.

---

## Key Features

### Multimodal Data Integration

* EEG analysis
* Eye-tracking analysis
* Physiological signal analysis
* Behavioral task integration
* Unified participant database

### Automated Synchronization

* Cross-system timestamp alignment
* Automatic task segmentation
* Support for recordings acquired on separate computers
* UTC-based synchronization workflow

### Signal Processing Pipelines

* EEG preprocessing
* Eye-tracking preprocessing
* Physiological data preprocessing
* Automated quality-control procedures
* Feature extraction and aggregation

### Analysis Modules

* Task-level analysis
* Event-level analysis
* Event-centered EEG analysis
* Multi-task comparison
* Quality-control reporting

---

## Integrated Acquisition Systems

### Eye Tracking

**Tobii Pro X3-120**

The platform analyzes gaze behavior using eye-tracking recordings, including:

* Fixations
* Saccades
* Regression detection
* Pupil diameter analysis
* Data validity assessment

### EEG

**64-Channel BrainVision EEG System**

The EEG module supports:

* Task segmentation
* Signal preprocessing
* Spectral feature extraction
* Connectivity analysis
* Event-centered neural analysis

### Physiological Monitoring

**Corsano Wearable Device**

Supported physiological measures include:

* Heart rate (BPM)
* Heart rate variability (HRV)
* Respiration rate
* Motion monitoring

### Experimental Control

**E-Prime**

E-Prime serves as the synchronization reference for:

* Cognitive task presentation
* Reading paradigms
* Behavioral assessments
* Task-window extraction

---

## System Architecture

```text
E-Prime
    ↓
Task Window Extraction
    ↓
─────────────────────────
│          │           │
↓          ↓           ↓

Eye      EEG      Physiology
Tracking

    ↓
Feature Extraction
    ↓
Task-Level Analysis
    ↓
Event-Level Analysis
    ↓
Cross-Modal Integration
```

---

## Analysis Pipeline

1. Data Import
2. Cross-System Synchronization
3. Signal Quality Validation
4. Task Segmentation
5. Eye-Tracking Analysis
6. Physiological Analysis
7. EEG Preprocessing
8. Feature Extraction
9. Event Detection
10. Multimodal Integration

---

## Supported Experimental Tasks

* Oral Reading – Erased Text
* Silent Reading – Erased Text
* Silent Reading – Erased Text with Metronome
* Silent Reading – Static Text with Metronome
* Silent Reading – Static Text
* Resting State

---

## Technical Challenges

One of the primary engineering challenges addressed by the platform was the synchronization of data acquired from independent systems operating on separate computers.

The platform automatically aligns:

* E-Prime task logs
* Eye-tracking recordings
* EEG recordings
* Physiological recordings

despite differences in timestamp formats, recording architectures, and acquisition workflows.

This synchronization framework enables accurate multimodal analysis across all recorded modalities.

---

## Demo Dataset

This repository includes a complete synthetic multimodal dataset created for software validation, demonstration, and GitHub distribution purposes.

The dataset contains synthetic examples of:

* EEG recordings
* Eye-tracking recordings
* Physiological recordings
* E-Prime task logs

The included data allows the platform's full workflow to be demonstrated, including synchronization, preprocessing, feature extraction, quality control, task-level analysis, and event-level analysis.

No real participant data is included in this repository.

---

## Technologies

* Python
* Streamlit
* Pandas
* NumPy
* SciPy
* OpenPyXL

---

## Author

**Talya Tohar**

Faculty of Biomedical Engineering
Technion – Israel Institute of Technology

Developed at the Educational Neuroimaging Group.
