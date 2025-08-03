
# Creative Ad‑Quality Classification Model

**Goal:** Train a Convolutional Neural Network (CNN) that detects low‑quality or policy‑violating ad creatives.

## Project Structure

```
ad_quality_cnn/
├── README.md
├── requirements.txt
├── data/               # place raw images here
│   └── README.md
├── src/
│   ├── data_prep.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
└── models/             # saved Keras models / checkpoints
```

## Quick Start

```bash
# 1. (Optional) create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Install deps
pip install -r requirements.txt

# 3. Organise data:
#    data/
#      ├── low_quality/
#      └── ok/
# Each sub‑folder contains jpg / png images.

# 4. Train
python src/train.py --data_dir data --epochs 15

# 5. Evaluate
python src/evaluate.py --model_path models/best_model.h5 --data_dir data
```

## Data Expectations

* Two classes by default:
  * `low_quality` – blurry, off‑brand, or policy‑violating creatives  
  * `ok` – acceptable creatives

Update `CLASS_NAMES` in `utils.py` if you add more categories.

## Key Technologies

* TensorFlow / Keras • NumPy • Pandas • OpenCV • scikit‑learn  
* Matplotlib for basic plots

## Metrics Reported

* Precision, Recall, F1‑score, ROC‑AUC  
* Confusion Matrix visualisation

## Author

Khush M. Patel – MSCS @ Northeastern University
