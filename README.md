# PROFUSEme: Prostate Cancer Recurrence Prediction via FUSEd Multi-modal Embeddings

Implementation of preprocessing pipeline for CHIMERA prostate cancer dataset following the PROFUSEme paper (https://arxiv.org/html/2509.14051v2).

## Requirements

- **Python**: 3.9+
- **System dependencies (macOS)**:
  ```bash
  brew install openslide
  ```

- **Python packages**: See `requirements.txt`

## Installation

1. Install system dependencies:
   ```bash
   brew install openslide
   ```

2. Install Python dependencies:
   ```bash
   python3 -m pip install -r requirements.txt
   ```

   Or with `pip3`:
   ```bash
   pip3 install -r requirements.txt
   ```

## Usage

### Local Streaming Setup (S3 Access)
Run first to connect to CHIMERA S3 bucket and list patients:
```bash
python3 local_streaming.py
```

This will:
- Connect to `s3://chimera-challenge/v2/task1` (anonymous access, no credentials needed)
- List available patients
- Create local caching directories (`data/patches`, `data/features`, etc.)
- Initialize S3 streamer for on-demand data access

### Preprocessing Pipeline
Run preprocessing stages (pathology, radiology, clinical):
```bash
python3 preprocessing.py
```

This will:
- Extract THINKING FAST patches (224×224, 1.25× magnification) for binary BCR classification
- Extract THINKING SLOW patches (1024×1024, 20× magnification) for time-to-recurrence regression
- Process MRI sequences (T2, ADC, HBV) with registration
- Encode clinical data (8 attributes per PROFUSEme paper)

## Pipeline Architecture

### STAGE 1: Pathology (WSI Preprocessing)
- **Thinking Fast**: Binary classification (BCR vs no-BCR)
  - Patches: 224×224 pixels
  - Resolution: 8 microns per pixel (1.25× magnification)
  - Non-overlapping grid extraction
  
- **Thinking Slow**: Time-to-recurrence regression
  - Patches: 1024×1024 pixels
  - Resolution: 0.5 microns per pixel (20× magnification)
  - 75% overlapping patches (50% step size)

### STAGE 2: Radiology (MRI Preprocessing)
- Sequences: T2-weighted (T2), Apparent Diffusion Coefficient (ADC), High b-value (HBV)
- Registration: ADC and HBV rigidly co-registered to T2W using ITK-Elastix
- Masking: Prostate tissue mask applied
- Output: 3-channel registered MRI

### STAGE 3: Clinical Data
- 8 clinical attributes encoded:
  - Age at radical prostatectomy
  - ISUP grade
  - Pathologic T stage
  - Positive lymph nodes (binary)
  - Capsular penetration (binary)
  - Positive surgical margins (binary)
  - Seminal vesicle invasion (binary)
  - Lymphovascular invasion (binary)

## Data Flow

```
S3 CHIMERA Dataset (Free, Anonymous Access)
    ↓
Local Streaming (on-demand access, no full download)
    ↓
Pathology Preprocessing (WSI patches)  +  Radiology (MRI)  +  Clinical Data
    ↓
Local Cache (patches, features, clinical CSVs)
    ↓
Model Training (embeddings + clinical data)
```

## Key Parameters

See `preprocessing.py` → `PreprocessingConfig` class for:
- Patch sizes and magnifications
- MRI sequences and registration parameters
- Tissue filtering thresholds
- Output directory structure

## Notes

- All data streams from S3 (free, no egress costs for public CHIMERA dataset)
- Only processed patches and features cached locally (~5-10 GB for 15 patients)
- Full ITK-Elastix MRI registration requires additional setup (currently placeholders)
- S3 access requires internet connection but no AWS credentials for anonymous bucket

## References

- Paper: ["PROFUSEme: PROstate Cancer Biochemical Recurrence Prediction via FUSEd Multi-modal Embeddings"](https://arxiv.org/html/2509.14051v2)
- Dataset: CHIMERA Challenge (Prostate Cancer Imaging)
- OpenSlide: https://openslide.org/






Project overview - AI generated - can ignore



Based on my exploration of your PROFUSEme folder, here's a comprehensive overview of the project and its current progress, including explanations for each file and its purpose.

## Project Overview
PROFUSEme is an implementation of a preprocessing pipeline for the CHIMERA prostate cancer dataset, based on the PROFUSEme paper (https://arxiv.org/html/2509.14051v2). The goal is to prepare multimodal data (pathology whole-slide images/WSI, radiology MRI sequences, and clinical features) for predicting prostate cancer recurrence. The pipeline follows a "thinking fast" (binary BCR classification) and "thinking slow" (time-to-recurrence regression) approach, extracting patches from WSI at different magnifications and resolutions.

The data is sourced from the public CHIMERA S3 bucket (`s3://chimera-challenge`), which is accessible anonymously (no credentials needed). The project emphasizes streaming and minimal local storage to handle large medical imaging files.

## Current Project Progress
- **Setup and Exploration Phase**: The project is in early stages. Scripts for S3 access, local streaming setup, and preprocessing have been created, but actual data processing has not been fully successful. 
- **Key Issues Identified**:
  - S3 bucket structure exploration (via explore_s3_bucket.py) suggests the paths used in other scripts (e.g., `v2/task1/pathology/images/`) may be incorrect, as no WSI files are found for patients. The bucket might use different prefixes (e.g., `v2/task1/data/` or similar).
  - Preprocessing stats (stats.json) show 0 patients processed, with errors like "No WSI files" for tested patients (1003, 1010, 1011).
  - Local streaming (local_streaming.py) has been run, creating directories and loading clinical metadata for 15 patients, but patch extraction hasn't proceeded.
- **Next Steps Needed**: Verify correct S3 paths, run preprocessing on a small subset, and potentially fix path mappings in scripts. The Google Drive pipeline (drive_pipeline.py) is designed for zero-local-disk processing but hasn't been executed yet.
- **Data Status**: Clinical metadata for 15 patients is cached locally. No patches or processed MRI data exist yet. The project is ready for testing but requires path corrections.

## File Explanations

### Core Scripts
- **drive_pipeline.py**: A comprehensive pipeline that streams CHIMERA data from S3 directly to Google Drive without local storage. It authenticates with Google Drive (requires `client_secrets.json`), creates patient folders, uploads raw files (optional), downloads WSI temporarily for OpenSlide processing, extracts "thinking fast" (224×224 patches at 1.25× mag) and "thinking slow" (1024×1024 overlapping patches at 20× mag) patches in memory, and uploads them to Drive. Includes tissue filtering. Purpose: Enable preprocessing on cloud storage for users without local compute/storage. Not yet run.
  
- **explore_s3_bucket.py** (your active file): Diagnostic script to explore the CHIMERA S3 bucket structure. Tests various prefixes (e.g., `v2/task1/data/`, data) to find patient folders and files. Uses boto3 for listing. Purpose: Troubleshoot access and confirm paths, as other scripts assume specific structures that may not match. Has identified potential path issues (e.g., no "patient" folders in expected locations).

- **local_streaming.py**: Sets up local streaming infrastructure for the first 15 patients. Connects to S3 anonymously, lists patients, creates local cache directories (patches, features, etc.), loads clinical JSON data into a Pandas DataFrame, and saves it as CSV. Includes a `CHIMERAStreamer` class for on-demand file access (e.g., streaming WSI bytes). Purpose: Prepare for local preprocessing without full downloads. Has been executed, producing all_15_patients_metadata.csv with patient IDs.

- **preprocessing.py**: Local preprocessing script that downloads WSI files from S3 to a cache, then uses OpenSlide to extract fast/slow patches locally. Handles tissue masking, saves PNGs to fast_patches and slow_patches, and logs stats. Purpose: Perform patch extraction on local machine. Run with defaults (1 patient, fast stage), but failed due to missing WSI files (path issue). Stats show 0 processed.

- **test_s3_access.py**: Tests S3 connectivity using boto3 (tries multiple regions) and s3fs. Lists bucket contents to verify access. Purpose: Ensure S3 setup works before running other scripts. Useful for debugging region/connection issues.

### Documentation and Prompts
- **README.md**: Project overview, installation instructions (including macOS OpenSlide via Homebrew), usage guide (run local_streaming.py first, then preprocessing.py), pipeline architecture (stages for pathology, radiology, clinical), and data flow diagram. Purpose: User guide and high-level explanation.

- **mcp_drive_streaming_prompt.md**: A detailed prompt (likely for an AI assistant) to generate drive_pipeline.py. Outlines requirements like zero-disk policy, user inputs, pipeline stages, and code structure. Purpose: Documentation of the design intent for the Drive pipeline.

### Dependencies and Config
- **requirements.txt**: Lists Python packages: s3fs, boto3, openslide-python, pandas, Pillow, numpy, scipy, pydrive2, google-auth, PyYAML. Purpose: Install via `pip install -r requirements.txt`.

### Data and Metadata (in data folder)
- **`clinical/all_15_patients_metadata.csv`**: CSV of patient IDs (1003, 1010, etc.) from local_streaming.py. Purpose: Quick reference for selected patients; no full clinical features yet.
  
- **`clinical_data/clinical_features.csv`**: Empty file. Purpose: Intended for encoded clinical features (8 attributes from the paper, e.g., age, ISUP grade), but not populated.

- **`metadata/streaming_config.json`**: JSON config with S3 paths, selected patients (15), local dirs, and notes. Purpose: Configuration for streaming scripts.

- **`preprocessing_metadata/pipeline_stats.json`**: JSON with processing stats (all 0). Purpose: Track overall progress.

- **`preprocessing_metadata/stats.json`**: JSON with errors (e.g., "No WSI files" for patients). Purpose: Detailed logs from preprocessing.py runs.

- **`fast_patches/`, `slow_patches/`, `features/`, etc.**: Empty directories created by local_streaming.py. Purpose: Cache extracted patches and features.

### Gemini-Code Subfolder
- **context.md**: Project context, S3 path corrections (notes TIFF not SVS, MHA format), and a proposed MCP prompt for data ingestion. Purpose: Planning and corrections for the pipeline.

- **drive_pipeline.py**: An alternative or earlier version of drive_pipeline.py, with similar structure but possibly generated by an AI (Gemini). Purpose: Backup or variant implementation.


