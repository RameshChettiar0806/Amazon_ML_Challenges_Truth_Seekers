# Business Entity Resolution Pipeline

This folder contains the complete, self-contained pipeline for reproducing the entity resolution results for the Amazon ML Challenge 2026.

---

## 1. Environment Setup

### Option A: Using Conda
```bash
conda create -n amazon_ml python=3.10 -y
conda activate amazon_ml
conda install -c conda-forge rapidfuzz lightgbm scikit-learn pandas scipy -y
```

### Option B: Using Pip / venv
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

---

## 2. Directory Structure

```
business_entity_resolution/
├── src/
│   ├── __init__.py
│   ├── blocking.py       # Candidate generation via TF-IDF character n-grams + cosine top-K
│   ├── notebooks/        # Step-by-step Jupyter notebooks for EDA, blocking & features
│   │   ├── 01_data_preprocessing.ipynb
│   │   ├── 02_blocking_candidate_generation.ipynb
│   │   └── 03_feature_engineering.ipynb
│   └── run_pipeline.py   # End-to-end execution script
├── requirements.txt      # Pinned dependencies
└── README.md             # This reproduction guide
```

---

## 3. How to Run the Pipeline (End-to-End)

From the project root (`student_resource/`):

### Step 1: Run Full Pipeline
```bash
python code/business_entity_resolution/src/run_pipeline.py --test-dir dataset/test --output-dir output
```

This will automatically:
1. Load test data (`test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`).
2. Run character n-gram blocking across sources.
3. Save candidate pairs to `output/candidate_pairs.tsv`.
4. Extract string & token similarity features with RapidFuzz.
5. Predict match probabilities using trained model and apply precision-heavy $F_{0.5}$ optimal threshold.
6. Save predictions to `output/matching_results.tsv`.

### Step 2: Validate Outputs
Run the official validator to confirm full compliance before submitting:
```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```
