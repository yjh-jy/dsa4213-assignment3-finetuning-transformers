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
