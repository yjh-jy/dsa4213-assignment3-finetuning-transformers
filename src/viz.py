import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_loss_vs_steps(train_steps, train_losses, eval_steps, eval_losses, save_path):
    plt.figure(figsize=(8,3), dpi=300)
    plt.plot(train_steps, train_losses, marker=".", linewidth=1, label='Train')
    plt.plot(eval_steps, eval_losses, marker=".", linewidth=1, label='Eval')
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss per Training Step")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_loss_vs_epochs(train_epochs, train_losses, eval_epochs, eval_losses, save_path):
    plt.figure(figsize=(8,3), dpi=300)
    plt.plot(train_epochs, train_losses, marker=".", linewidth=1, label='Train')
    plt.plot(eval_epochs, eval_losses, marker=".", linewidth=1, label='Eval')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per Training Epoch")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_time_per_step(step_indices, time_per_step, save_path, smooth_window=5):
    plt.figure(figsize=(8,3), dpi=300)
    steps = np.array(step_indices)
    times = np.array(time_per_step)
    # smooth for readability if many points
    if smooth_window and len(times) >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        times_sm = np.convolve(times, kernel, mode="valid")
        steps_sm = steps[(smooth_window-1):]
        plt.plot(steps_sm, times_sm, label=f"smoothed (w={smooth_window})")
    else:
        plt.plot(steps, times, label="time per step")
    plt.xlabel("Step")
    plt.ylabel("Time taken (s)")
    plt.title("Time taken per Training Step")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_confusion_matrix_from_preds(y_true, y_pred, save_path, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(5,4), dpi=300)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
