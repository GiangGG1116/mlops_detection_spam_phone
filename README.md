# MLOps Spam Phone Detection

This project has been cleaned and reorganized for production use:
- Removed legacy/duplicate code and absolute hardcoded paths.
- Standardized the training/inference pipeline with YAML config.
- Clearly separated `src/` (logic), `configs/` (configuration), `scripts/` (entrypoints), `models/` (model lifecycle), and `data/` (datasets/runtime outputs).

## Project Structure

```text
mlops_spam_phone/
├─ configs/
│  ├─ external_data.yml
│  ├─ pipeline.yaml
│  └─ risk_config.yaml
├─ data/
│  ├─ input/
│  ├─ feedback/
│  ├─ predictions/
│  └─ train/
├─ models/
│  ├─ production/
│  ├─ candidates/
│  └─ archive/
├─ scripts/
│  ├─ run_pipeline.py
│  ├─ run_infer.py
│  └─ run_train.py
├─ src/
│  ├─ cli.py
│  ├─ pipeline.py
│  ├─ core/
│  └─ steps/
└─ requirements.txt
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Edit `configs/pipeline.yaml`:
- `data.report_source`: feedback parquet file
- `data.history_source`: call history parquet file
- `data.external_data_yaml`: feature engineering config
- `model.production_*`: active production model artifacts

## Run Pipeline

### 1) Run full pipeline (inference + labeling + dataset build + train/promote)

```bash
python -m src.cli pipeline
```

Or:

```bash
python scripts/run_pipeline.py
```

### 2) Run each step separately

```bash
python -m src.cli infer
python -m src.cli label
python -m src.cli dataset
python -m src.cli train
```

## Main Outputs

- Predictions: `data/predictions/pred_by_phone.json`
- Debug features + score: `data/predictions/predict_debug.csv`
- Feedback labels: `data/train/feedback_labels.csv`
- Changed-label rows: `data/train/changed_rows.csv`
- Merged training dataset: `data/train/label_merged.csv`
- Production model: `models/production/{xgb.pkl, feature_columns.json, train_metrics.json}`
- New candidate models: `models/candidates/<timestamp>/`
- Archived models after promotion: `models/archive/<timestamp>/`

## Operational Notes

- Do not commit runtime artifacts; check `.gitignore`.
- All paths are relative to the project root (machine-independent).
- To use a different config file:

```bash
python -m src.cli --config /path/to/pipeline.yaml pipeline
```
