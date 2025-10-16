# Parameter-Efficient Fine-Tuning of DistilBERT for Sentiment Classification (IMDB)
This project is completed for Assignment 3 under the DSA4213 NLP for Data Science course.

For the full report, please access [here](report.pdf).

## **Directory (important files/folders):**

```
results/                # experiment outputs (check after runs)
  full_size5000/
  full_size10000/
  full_size25000/
  lora_size5000/
  lora_size10000/
  lora_size25000/
  plots/

src/
  config.py             # all experiment / training / dataset / model configs live here
  data.py               # dataset loading + sampling logic
  evaluate_utils.py     # evaluation helpers and metric computations
  modeling.py           # model definitions (DistilBERT + PEFT adapters e.g. LoRA)
  train_trainer.py      # training loop, checkpointing, metrics logging
  viz.py                # scripts to create the plots in results/plots

eda.ipynb             # exploratory notebook (generating overall plots)
main.py               # top-level entrypoint for train / eval runs
.gitignore
README.md (this file)
requirements.txt
```

## How to Run
1. Clone the repo
```bash
git clone https://github.com/yjh-jy/dsa4213-assignment3-finetuning-transformers
cd dsa4213-assignment3-finetuning-transformers
```

2. **Create environment & install dependencies**

```bash
# create venv (recommended)
python -m venv .venv            # I used Python 3.10.6 
source .venv/bin/activate       # linux / mac
# .venv\Scripts\activate        # windows (PowerShell or cmd as appropriate)

# install requirements
pip install -r requirements.txt
```

> `requirements.txt` should include `torch`, `transformers`, `datasets`, `accelerate` (optional), `scikit-learn`, and plotting libs like `matplotlib` / `seaborn`. If you use GPU, install the appropriate `torch` build for CUDA.

2. **Edit configs (if you want custom runs)**
   All configuration is centralized in `src/config.py`. Before running, open it and set:

```py
# Example keys you will find / can edit in config.py
MODEL_NAME = "distilbert-base-uncased"
DATASET_NAME = "imdb"
DATASET_SIZES = [25000, 10000, 5000]  # training subset sizes
MAX_LENGTH = 256
BATCH_SIZE = 32
EPOCHS = 3
LR_FULL = 5e-5
LR_LORA = 1e-4   # set to at least 10x higher than the full-finetuning based on existing research (Thinking Machine's work on Low Regret LoRA)
OUTPUT_DIR_BASE = "results"
SEED = 42

# LoRA-specific hyperparameters (used in modeling.py)
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.2

```

Modify these values for the experiment you want to run. The repo is structured to run variants (full vs LoRA and multiple sizes) automatically in a single run in `main.py`

3. **Run training**
   From project root:

```bash
# using the top-level entrypoint
python main.py
```
`main.py` reads `src/config.py` and dispatches to training or evaluation. Run once and all the configs wille be ran as well. 

4. **Generate comparison plots**

Run the notebook's cell `eda.ipynb` from top to bottom to generate the plots below:
* `accuracy_curves.png`
* `loss_curves.png`
* `f1_per_mparams.png`
* `efficiency_tradeoff.png`
* `acc_vs_size.png`

---


## What to expect in `results/` after a run

```
results/<strategy_dataset_size>/
  checkpoints/              # model checkpoints (if enabled)
  preds_and_labels.npz      # model predictions & ground-truth labels
  confusion_matrix.png      # confusion matrix of the given config
  loss_per_epochs.png       # plot of the loss per epoch of the given config
  loss_per_steps.png        # plot of the loss per step of the given config
  time_per_step.png         # plot of the time per step of the given config
  metrics.json              # computed metrics
  log_history.json          # training loss/step logs
  training_args.bin         # training configs used in Trainer API
results/plots/              # aggregated plots across experiments
```

---

## Notes on configuration and reproducibility

* **All hyperparameters live in `src/config.py`** — change them there
* **Random seeds** — `SEED` is present in `config.py`. Set a fixed seed if you want reproducible results across runs. Note: GPU nondeterminism can still produce minor differences unless you enforce deterministic modes in PyTorch.
* **PEFT implementation** — `src/modeling.py` contains the logic to construct either the full fine-tuned DistilBERT classifier or the LoRA-augmented (parameter-efficient) variant. If you want to add another PEFT method, add it here and add a switch in `main.py`.
* **Dataset sampling** — `src/data.py` supports sampling limited training sizes (5k, 10k, 25k). If you want to use a different dataset or your local copy of IMDB, update `DATASET_PATH` or replace the dataset loader.

---

## Troubleshooting & tips

* **CUDA / out-of-memory:** reduce `BATCH_SIZE`, or use gradient accumulation (if implemented), or use mixed precision (AMP) if available.
* **Slow training:** ensure `num_workers` in data loader is > 0 in `data.py`. Use `accelerate` or `torchrun` for multi-GPU runs if you have multiple GPUs.
* **Missing packages / import errors:** verify `requirements.txt` and that your virtual environment is activated. E.g., install `transformers`, `datasets`, `torch`, `scikit-learn`, `matplotlib`, `seaborn`.
* **Plots don't show new experiments:** verify that experiment subfolders have `metrics.json` and `preds.json`; `viz.py` aggregates on that basis.
