# MURA X-ray Classifier

A TensorFlow/Keras project for classifying musculoskeletal radiographs as **Normal** or **Abnormal**. It uses a DenseNet121 backbone and supports training, evaluation, command-line prediction, and a small desktop GUI.

> **Research and education only.** This project is not a medical device and must not be used to diagnose, treat, or make clinical decisions about patients.

## Project contents

- `train.py` — trains the classifier in two stages and saves the best and final models.
- `evaluate.py` — evaluates the best saved model and writes a classification report and plots.
- `predict.py` — predicts a label for one image from the command line.
- `gui.py` — opens a desktop interface for selecting and classifying an image.
- `utils.py` — dataset loading, image preprocessing, GPU setup, and training plots.
- `config.py` — dataset, model, and output paths plus training settings.

The dataset, local Python environment, trained models, and generated outputs are kept outside this repository. See `.gitignore`.

## Requirements

- Python 3.10 is recommended.
- TensorFlow 2.10.1 (pinned in `requirements.txt`).
- The MURA dataset, stored locally in the expected layout below.

Create and activate an environment, then install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

TensorFlow 2.10 has platform-specific GPU support. CPU training works without a compatible GPU setup; consult TensorFlow's installation guidance for GPU requirements on your system.

## Dataset setup

Place the MURA data under `dataset/MURA-v1.1/`, including:

```text
dataset/
└── MURA-v1.1/
    ├── train_image_paths.csv
    ├── valid_image_paths.csv
    └── ... image files referenced by those CSVs ...
```

The CSV entries should resolve from the project-level `dataset/` directory (for example, `MURA-v1.1/...`). The dataset is not included in this repository. Download it from the official [Stanford AIMI MURA dataset page](https://aimi.stanford.edu/datasets/mura-msk-xrays), which links to the dataset download on Redivis. Follow Stanford's current access steps and data-use terms. See the [original MURA project page](https://stanfordmlgroup.github.io/competitions/mura/) for dataset background. After downloading, arrange the files to match the folder layout above.

## Train

```bash
python train.py
```

The best checkpoint is written to `models/best_model.keras`; the completed model is written to `models/final_model.keras`. Training also creates logs and plots under `outputs/`.

## Evaluate

After training:

```bash
python evaluate.py
```

This loads `models/best_model.keras`, evaluates the validation set, prints a classification report and metrics, and saves plots in `outputs/`.

## Predict one image

```bash
python predict.py
```

Enter the path to a PNG image when prompted. The script loads `models/best_model.keras`.

## Desktop app

```bash
python gui.py
```

Select a PNG or JPEG image in the window. The app also expects `models/best_model.keras` to exist.

## Configuration

Edit `config.py` to change image size, batch size, epoch counts, learning rates, GPU mixed precision, or project paths. Training outputs and model files are intentionally ignored by Git; back them up separately if needed.
