import os
from src.config import DATASET_SIZES, OUTPUT_DIR_BASE
from src.data import load_and_prepare, get_train_subset
from src.train_trainer import run_experiment
import pandas as pd

def main():
    train_all, test, tokenizer, tokenize_fn = load_and_prepare()
    results = []

    for size in DATASET_SIZES:
        train_subset = get_train_subset(train_all, size)
        for strategy in ["lora", "full"]:  # lora means LoRa Finetuning and full means Full Finetuning
            print(f"\n=== Running: strategy={strategy} size={size} ===")
            res = run_experiment(train_subset, test, tokenizer, tokenize_fn, strategy=strategy, size=size, output_base=OUTPUT_DIR_BASE)
            print("Result:", res)
            results.append(res)

    # Save aggregated metrics
    os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR_BASE, "aggregated_metrics.csv"), index=False)

    print(f"\nAll experiments finished. Results in {OUTPUT_DIR_BASE}")

if __name__ == "__main__":
    main()
