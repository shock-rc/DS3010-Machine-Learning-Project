# Multimodal Alzheimer's Disease Classification

This project investigates Alzheimer's disease classification from brain MRI data. It combines a 3D convolutional neural network (CNN) branch for T1-weighted MRI volumes with a metadata branch containing patient-level clinical features.

The project report is available at [`report/main.pdf`](report/main.pdf), with the source at [`report/main.tex`](report/main.tex).

## Project summary

- Dataset: OASIS MRI volumes and accompanying metadata
- Task: binary classification
  - `0`: non-demented
  - `1`: very mild, mild, or moderate dementia
- MRI input: single-channel `128 x 128 x 128` volume
- Metadata features: age, education, socioeconomic status, MMSE, eTIV, nWBV, ASF, and male indicator
- Model: multimodal 3D CNN with feature-level fusion
- Checkpoint: [`best_model.pth`](best_model.pth)

The final model uses four 3D convolutional stages, adaptive average pooling, a 64-dimensional MRI representation, a 32-dimensional metadata representation, and a two-class classifier.

## Repository structure

```text
final_model/
├── data.py       # Metadata preprocessing, data splitting, and MRI dataset
├── model.py      # 3D CNN architecture and checkpoint loading
├── training.py   # Training and validation functions
├── plotting.py   # Confusion matrix and training-history plots
├── evaluate.py   # Reusable test-set evaluation pipeline
└── __init__.py   # Public package exports

report/
└── main.pdf      # Project report
```

The raw dataset is not documented as part of this repository setup. Place the OASIS files in the expected local paths before running the pipeline:

```text
raw/
├── oasis_cross-sectional.csv
└── oasis/
    └── OASIS/
```

## Setup

Use Python with the project dependencies installed from the root-level requirements file:

```bash
pip install -r requirements.txt
```

Run commands from the repository root so the relative dataset and checkpoint paths resolve correctly.

## Evaluate the saved model

```python
from final_model import evaluate_best_model

results = evaluate_best_model(
    metadata_path="raw/oasis_cross-sectional.csv",
    checkpoint_path="best_model.pth",
)

print(f"Test loss: {results['test_loss']:.4f}")
print(f"Test accuracy: {results['test_accuracy']:.4f}")
print(results["classification_report"])
```

To load the model for custom inference:

```python
from final_model import load_best_model

model, device = load_best_model("best_model.pth")
```

## Reported results

The report presents the following test-set summary for the final multimodal model:

| Metric | Reported value |
|---|---:|
| Accuracy | 0.78 |
| Demented precision | 0.51 |
| Demented recall | 0.90 |
| Macro precision | 0.74 |
| Macro recall / balanced accuracy | 0.82 |

These results come from a single held-out split of 88 test samples (68 non-demented and 20 demented). They should not be interpreted as clinical performance. 
## Authors

- Alphin Issac Varghese — 142401007
- Aaron C Abraham — 142401004
