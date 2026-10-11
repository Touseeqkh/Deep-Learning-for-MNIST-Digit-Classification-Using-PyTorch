# Deep Learning on MNIST: Perceptron, MLP and CNN

A PyTorch project that classifies handwritten digits from the MNIST dataset and compares three neural-network architectures:

- **Linear classifier** — a simple baseline using flattened pixel values.
- **Multilayer perceptron (MLP)** — a fully connected neural network with hidden layers.
- **Convolutional neural network (CNN)** — learns spatial features from image pixels.

The project includes a notebook for exploration and comparison, plus a command-line training script.

## Project structure

```text
Deep-learning-main/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── mnist_deep_learning_comparison.ipynb
└── src/
    └── train.py
```

MNIST data is downloaded automatically into `./data` the first time the notebook or script runs. The dataset is not included in this repository.

## Requirements

- Python 3.10 or newer recommended
- PyTorch
- torchvision
- NumPy
- Matplotlib
- Jupyter Notebook or JupyterLab (for the notebook)

## Setup

### Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PyTorch installation fails or you want GPU/CUDA support, follow the official selector at https://pytorch.org/get-started/locally/ and choose the command that matches your system. Then install the remaining packages from `requirements.txt`.

## Run the notebook

```bash
jupyter notebook
```

Open `notebooks/mnist_deep_learning_comparison.ipynb` and run the cells from top to bottom. The notebook downloads MNIST, displays sample images, trains the models, plots accuracy/loss, and shows sample predictions.

## Run training from the terminal

Train the CNN (default):

```bash
python src/train.py --model cnn --epochs 3
```

Try the MLP or linear baseline:

```bash
python src/train.py --model mlp --epochs 5 --batch-size 64
python src/train.py --model linear --epochs 5
```

Available options:

| Option | Default | Description |
|---|---:|---|
| `--model` | `cnn` | `linear`, `mlp`, or `cnn` |
| `--epochs` | `3` | Number of training epochs |
| `--batch-size` | `128` | Samples per batch |
| `--learning-rate` | `0.001` | Adam optimizer learning rate |
| `--data-dir` | `./data` | Directory for MNIST files |
| `--seed` | `42` | Random seed |

## Evaluation

The script prints training loss, training accuracy, test loss, and test accuracy for each epoch. The notebook compares the final test accuracy of all three models and plots learning curves.

Results vary with random initialization, runtime environment, hyperparameters, and training duration. Run the project to obtain your own measured results; no fixed accuracy is claimed here.

## Notes

- No external `mnist.pkl.gz` file is required. This version uses `torchvision.datasets.MNIST`, which downloads the dataset when needed.
- The notebook is designed to run locally or in a hosted notebook environment with internet access for the initial dataset download.
- Start with a small number of epochs for a quick check, then increase it for more experiments.

## Repository description

PyTorch MNIST handwritten digit classification comparing a linear baseline, MLP, and CNN, with training scripts and visual evaluation.
