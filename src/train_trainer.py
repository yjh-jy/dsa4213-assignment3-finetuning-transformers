# src/train_trainer.py
import os
import json
import time
import numpy as np
import torch
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, TrainerCallback
from src.config import *
from src.viz import *
from src.evaluate_utils import compute_metrics_fn
from pathlib import Path
from src.modeling import get_model_and_freeze

class TimingCallback(TrainerCallback):
    """
    Tracks per-step training time and evaluation steps.
    Stores:
      - state.time_per_step: list of floats (seconds per training step)
      - state.cumulative_train_time: list of cumulative time at each step
      - state.step_indices: list of step indices for each training step
      - state.eval_step_indices: list of global step indices where evaluation occurred
    """

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start = time.time()
        self.last_time = self.train_start
        state.time_per_step = []
        state.cumulative_train_time = []
        state.step_indices = []
        state.eval_step_indices = []

    def on_step_end(self, args, state, control, **kwargs):
        now = time.time()
        step_time = now - self.last_time
        self.last_time = now

        state.time_per_step.append(step_time)
        cum_time = (state.cumulative_train_time[-1] if state.cumulative_train_time else 0.0) + step_time
        state.cumulative_train_time.append(cum_time)
        state.step_indices.append(state.global_step)

    def on_evaluate(self, args, state, control, **kwargs):
        # record the step index when evaluation occurs
        state.eval_step_indices.append(state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.train_start
        print(f"Total training time: {total_time:.2f} seconds")

def count_trainable_params(model):
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)

def _choose_eval_logging_steps(n_train_examples, batch_size, num_epochs=3, prefer_eval_per_epoch=2, save_total_limit=2):
    """
    Automatically choose eval_steps, logging_steps, and save_steps aligned for Trainer.

    This scales based on dataset size so that:
    - Small datasets do not evaluate/save too frequently (avoids slowing LoRA)
    - Large datasets still provide step-wise granularity for plots

    Args:
        n_train_examples: int, number of training samples
        batch_size: int, per-device batch size
        num_epochs: int, total number of epochs
        prefer_eval_per_epoch: int, how many evaluations per epoch (default 2)
        save_total_limit: int, max number of checkpoints to keep

    Returns:
        eval_steps, logging_steps, save_steps, save_total_limit
    """
    steps_per_epoch = max(1, n_train_examples // batch_size)
    
    # Compute evaluation steps
    eval_steps = max(1, steps_per_epoch // prefer_eval_per_epoch)

    # Logging steps: 1/4 of eval_steps
    logging_steps = max(1, eval_steps // 4)

    # Save steps: must be multiple of eval_steps for Trainer compatibility
    save_steps = eval_steps

    # Clamp values to avoid too frequent or too sparse evaluation
    eval_steps = max(10, min(eval_steps, steps_per_epoch))
    logging_steps = max(1, min(logging_steps, eval_steps))
    save_steps = max(eval_steps, save_steps)

    # Ensure save_total_limit is reasonable
    save_total_limit = max(2, save_total_limit)

    return eval_steps, logging_steps, save_steps, save_total_limit


def run_experiment(train_subset, test_dataset, tokenizer, tokenize_fn, strategy, size, output_base=OUTPUT_DIR_BASE):
    """
    train_subset, test_dataset: raw HuggingFace Dataset objects
    strategy: "full", or "lora"
    """
    run_name = f"{strategy}_size{size}"
    output_dir = os.path.join(output_base, run_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Tokenize (batched) and remove original text column
    tokenized_train = train_subset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized_test = test_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    # Ensure labels column named 'labels'
    tokenized_train = tokenized_train.rename_column("label", "labels")
    tokenized_test = tokenized_test.rename_column("label", "labels")

    # Set format for PyTorch tensors
    tokenized_train.set_format("torch")
    tokenized_test.set_format("torch")

    # Build model
    model = get_model_and_freeze(MODEL_NAME, strategy=strategy)

    # Count trainable parameters BEFORE training
    trainable_params = count_trainable_params(model)

    # Learning rate selection
    learning_rate = LR_FULL if strategy == "full" else LR_LORA

    # Automatically determine steps based on dataset size
    eval_steps, logging_steps, save_steps, save_total_limit = _choose_eval_logging_steps(
        n_train_examples=len(tokenized_train),
        batch_size=BATCH_SIZE,
        num_epochs=EPOCHS,
        prefer_eval_per_epoch=2,  # 2 evaluations per epoch, adjust if needed
        save_total_limit=2
    )
    # Optionally override or tune eval_steps here if you want fixed values

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # TrainingArguments - step-wise evaluation & logging
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=learning_rate,
        eval_strategy="steps",
        eval_steps=eval_steps,
        logging_strategy="steps",
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,  # Works safely now
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=False,
        seed=SEED,
        report_to="none",
        push_to_hub=False,
        disable_tqdm=False,
        use_cpu=False,
        dataloader_pin_memory=False,    # DISABLE this if using NVIDIA GPU, I enabled this for Mac
    )

    # Trainer compute_metrics wrapper
    def compute_metrics(eval_pred):
        return compute_metrics_fn(eval_pred)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[TimingCallback]  # tracks wall-clock time
    )

    # Train (measure wall-clock time)
    t0 = time.time()
    train_result = trainer.train()
    t1 = time.time()
    train_time = t1 - t0

    # Save base model via trainer
    trainer.save_model()

    # If LoRA, save adapter separately
    if strategy == "lora":
        try:
            peft_path = os.path.join(output_dir, "peft_adapter")
            model.save_pretrained(peft_path)
        except Exception as e:
            print("Warning: failed to save PEFT adapters:", e)

    # Save log history
    log_history_path = os.path.join(output_dir, "log_history.json")
    with open(log_history_path, "w") as fh:
        json.dump(trainer.state.log_history, fh, indent=2)

    # Save time_per_step information
    if hasattr(trainer.state, "time_per_step"):
        time_per_step_path = os.path.join(output_dir, "time_per_step_history.json")
        with open(time_per_step_path, "w") as fh:
            json.dump(trainer.state.time_per_step, fh, indent=2)

    # Plot loss curves
    try:
        # retrieve lists
        log = trainer.state.log_history
        train_losses = [e["loss"] for e in log if "loss" in e]
        eval_losses = [e["eval_loss"] for e in log if "eval_loss" in e]

        train_steps_list = [e["step"] for e in log if "loss" in e and "step" in e]
        eval_steps_list = [e["step"] for e in log if "eval_loss" in e and "step" in e]

        eval_epochs = [e["epoch"] for e in log if "eval_loss" in e]
        train_epochs = [e["epoch"] for e in log if "loss" in e]

        # from callback
        step_indices = trainer.state.step_indices  # step indices recorded by TimingCallback
        time_per_step = trainer.state.time_per_step

        plot_loss_vs_steps(train_steps_list, train_losses, eval_steps_list, eval_losses, save_path=f"{output_dir}/loss_per_steps.png")
        plot_loss_vs_epochs(train_epochs, train_losses, eval_epochs, eval_losses, save_path=f"{output_dir}/loss_per_epochs.png")
        plot_time_per_step(step_indices, time_per_step, save_path=f"{output_dir}/time_per_step.png")
        
    except Exception as e:
        print("Warning: failed to plot loss combined curve:", e)

    # Use trainer.predict for final preds and labels on test
    predictions_output = trainer.predict(tokenized_test)
    logits = predictions_output.predictions
    preds = np.argmax(logits, axis=-1)
    labels = predictions_output.label_ids

    # Compute metrics (accuracy & macro-f1)
    final_metrics = compute_metrics_fn((logits, labels))
    eval_loss = predictions_output.metrics.get("test_loss") or predictions_output.metrics.get("eval_loss")

    # Save preds+labels
    preds_path = os.path.join(output_dir, "preds_and_labels.npz")
    try:
        np.savez_compressed(preds_path, preds=preds, labels=labels, logits=logits)
    except Exception as e:
        print("Warning: failed to save preds and labels:", e)

    # Confusion matrix plot
    cm_plot_path = os.path.join(output_dir, "confusion_matrix.png")
    try:
        plot_confusion_matrix_from_preds(labels, preds, save_path=cm_plot_path)
    except Exception as e:
        print("Warning: failed to plot confusion matrix:", e)

    # Final summary dict
    final = {
        "strategy": strategy,
        "dataset_size": size,
        "trainable_params": int(trainable_params),
        "train_time": float(train_time),
        "train_runtime_from_trainer": float(train_result.metrics.get("train_runtime", 0)) if train_result.metrics.get("train_runtime", None) is not None else None,
        "train_samples": len(tokenized_train),
        "eval_loss": float(eval_loss) if eval_loss is not None else None,
        "eval_accuracy": float(final_metrics.get("accuracy")),
        "eval_f1": float(final_metrics.get("f1")),
        "eval_steps_used": int(eval_steps),
        "logging_steps_used": int(logging_steps)
    }

    # Save final metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(final, f, indent=2)

    return final
