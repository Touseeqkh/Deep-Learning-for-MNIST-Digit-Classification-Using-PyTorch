# Project review and cleanup notes

## Changes made
- Replaced the original single notebook with a structured, documented MNIST comparison notebook.
- Removed hard-coded `/content/mnist.pkl.gz` paths and reliance on a separate pickle dataset that was not included.
- Standardized data loading through `torchvision.datasets.MNIST`, which downloads the dataset automatically.
- Added a standalone command-line training script (`src/train.py`).
- Added setup instructions, dependency list, `.gitignore`, and repository description in the README.
- Added repeatable random seeds, CPU/GPU device selection, loss/accuracy tracking, and visual comparisons.

## Before publishing
- Run the notebook from top to bottom and confirm all cells finish on your computer.
- Run `python src/train.py --model cnn --epochs 1` as a quick smoke test.
- If your old GitHub repository contains additional reports, figures, or assignment instructions, review and retain those files if relevant.
- Add a `LICENSE` only if you have chosen a license you are comfortable applying to your work.
- If this is coursework, follow your institution's rules for publishing code and clearly identify any code or ideas adapted from other sources.


## Report files
- `report.pdf`: project report (results are intentionally left blank until measured).
- `REPORT.md`: editable Markdown version of the report.
