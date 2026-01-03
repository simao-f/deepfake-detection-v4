# Deepfake Image Detection 

This repository contains the full code and data used for the Master's thesis titled *"Detecting Synthetic Media: Generative AI and its impact for Cybersecurity"* as part of the master's in Business Data Science at Aalborg University.

*Author*: Simão Ferreira
*Supervisor*: Hamid Bekamiri


# Project Overview

This repository contains a production-ready pipeline for training, evaluating, and serving deepfake-image classifiers. It includes:
- A configurable Jupyter notebook for model development (`notebooks/deepfake_detection.ipynb`).
- A reusable data directory structure under `data/Dataset/`.
- A Streamlit web application for single-image inference (`frontend/app.py`).


## Project Structure

```
.
├── data/
│   └── Dataset/
│       ├── Train/
│       ├── Validation/
│       └── Test/
├── notebooks/
│   ├── deepfake_detection.ipynb
│   └── eda.ipynb
├── frontend/
│   └── app.py
├── requirements.txt
└── README.MD
```

## Environment Setup

1. Create a Python virtual environment (Python 3.9+ recommended) and activate it.
2. Install dependencies:
	```bash
	pip install -r requirements.txt
	```
3. (Optional) Install Streamlit for the frontend if not covered by the requirements file:
	```bash
	pip install streamlit
	```

## Dataset Expectations

The project assumes an `ImageFolder`-compatible layout:
- `Train/`, `Validation/`, and `Test/` directories.
- Each split contains one folder per class (e.g., `Real/`, `Fake/`).

Update or export these environment variables as needed:
```bash
export DATA_DIR="/absolute/path/to/data/Dataset"
export ARTIFACTS_DIR="/absolute/path/to/artifacts"  # defaults to ./artifacts
```

## Training Workflow

1. Open `notebooks/deepfake_detection.ipynb` in VS Code.
2. Configure optional environment variables before running:
	```python
	MODEL_TYPE = "resnet"  # other options: "cnn", "mobilenet", "efficientnet"
	TRAIN_BACKBONE = "true"  # fine-tune backbone if supported
	NUM_EPOCHS = 25
	```
3. Execute the notebook cells sequentially.

### Features Implemented
- Custom CNN and transfer-learning backbones (ResNet-18, EfficientNet-B0, MobileNetV2).
- TQDM progress bars, early stopping, and ReduceLROnPlateau scheduling.
- Training history plots and comprehensive evaluation metrics (accuracy, precision, recall, F1, confusion matrix, ROC/AUC).
- Artifact export: model checkpoint (`.pt`), label map JSON, and metrics JSON.

## Evaluation Artifacts

Upon completion, the notebook saves artifacts in `ARTIFACTS_DIR`:
- `deepfake_<model>_<timestamp>.pt` – serialized model.
- `label_map_<timestamp>.json` – class index mappings.
- `metrics_<model>_<timestamp>.json` – evaluation summary.

## Streamlit Frontend

The frontend provides a simple interface for uploading a single image and receiving a prediction.

```bash
streamlit run frontend/app.py
```

When prompted:
- Specify the path to the saved `.pt` model.
- Specify the path to the corresponding `label_map_*.json` file.

Upload an image (PNG/JPG) to view the predicted label and confidence.

## Reproducibility & Configuration

Key hyperparameters and paths are controlled via environment variables (see notebook/code comments). Adjust `IMAGE_SIZE`, `BATCH_SIZE`, and optimizer settings through exported variables or by editing the notebook cells.

## Troubleshooting

- **Missing GPU**: The code automatically falls back to CPU if CUDA is unavailable.
- **Streamlit import errors**: Ensure `streamlit` is installed in the active environment.
- **Model/label map mismatch**: Load artifacts generated from the same training run.

## Next Steps

- Expand the dataset or add new classes.
- Experiment with additional backbones or training schedules.
- Integrate batch inference or REST endpoints for broader deployment.
