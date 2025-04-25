import argparse
import json
import os
import time
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# =============================================================================
# Model Definition
# =============================================================================
class StockPredictor(nn.Module):
    """
    A simple MLP for stock movement prediction.
    Input: a 14-dimensional feature vector (past 14 days of Close prices).
    Output: a single logit for binary prediction.
    """

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int = 1, dropout: float = 0.0
    ):
        super(StockPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# =============================================================================
# Data Loading
# =============================================================================
def load_data(
    csv_dir: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load preprocessed stock market datasets from CSV files.

    Expected CSV files:
      - train_stock_dataset.csv
      - val_stock_dataset.csv
      - test_stock_dataset.csv

    Each CSV must contain 14 feature columns (Close_day1 ... Close_day14)
    and 3 label columns (Label_3, Label_7, Label_15).

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test as torch tensors.
    """
    train_df = pd.read_csv(os.path.join(csv_dir, "train_stock_dataset.csv"))
    val_df = pd.read_csv(os.path.join(csv_dir, "val_stock_dataset.csv"))
    test_df = pd.read_csv(os.path.join(csv_dir, "test_stock_dataset.csv"))

    feature_columns = [f"Close_day{d}" for d in range(1, 15)]
    label_columns = ["Label_3", "Label_7", "Label_15"]

    X_train = torch.tensor(train_df[feature_columns].values, dtype=torch.float32)
    y_train = torch.tensor(train_df[label_columns].values, dtype=torch.float32)
    X_val = torch.tensor(val_df[feature_columns].values, dtype=torch.float32)
    y_val = torch.tensor(val_df[label_columns].values, dtype=torch.float32)
    X_test = torch.tensor(test_df[feature_columns].values, dtype=torch.float32)
    y_test = torch.tensor(test_df[label_columns].values, dtype=torch.float32)

    return X_train, y_train, X_val, y_val, X_test, y_test


# =============================================================================
# Evaluation Helpers
# =============================================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def evaluate_model(
    loader: DataLoader, model: nn.Module, criterion: nn.Module, device: str
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate the model on the given DataLoader.

    Returns:
        Average loss, concatenated predictions, and true labels.
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.concatenate(all_preds).flatten(), np.concatenate(all_labels).flatten()


# =============================================================================
# Training and Testing Functions
# =============================================================================
def train_task(
    task_index: int,
    csv_dir: str,
    out_dir: str,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 1337,
    dropout: float = 0.2,
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    """
    Train a StockPredictor model for a specific prediction task.

    Args:
        task_index: 0 for 3-day, 1 for 7-day, 2 for 15-day prediction.

    Returns:
        model, training history, and a dictionary of final information including
        training metrics and evaluation results.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Task {task_index}: Using device: {device}")

    # Load and prepare data for the specific task.
    X_train, y_train_all, X_val, y_val_all, _, _ = load_data(csv_dir)
    y_train = y_train_all[:, task_index].unsqueeze(1)
    y_val = y_val_all[:, task_index].unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    # Model and optimizer initialization.
    input_size = X_train.shape[1]
    hidden_size = 128
    model = StockPredictor(input_size, hidden_size, output_size=1, dropout=dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}
    total_start_time = time.time()
    checkpoint_path = ""

    print(f"Task {task_index}: Starting training...")
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_start = time.time()

        # Training loop
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * X_batch.size(0)

        epoch_train_loss /= len(train_loader.dataset)
        history["train_loss"].append(epoch_train_loss)

        # Evaluate on validation set
        epoch_val_loss, val_preds, val_labels = evaluate_model(val_loader, model, criterion, device)
        history["val_loss"].append(epoch_val_loss)
        metrics = compute_metrics(val_labels, val_preds)
        epoch_time = time.time() - epoch_start

        print(
            f"Task {task_index} Epoch {epoch+1}/{epochs}: "
            f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, "
            f"Acc: {metrics['accuracy']:.4f}, Prec: {metrics['precision']:.4f}, "
            f"Rec: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}, Time: {epoch_time:.2f}s"
        )

        # Save best model based on validation loss.
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            os.makedirs(out_dir, exist_ok=True)
            checkpoint_path = os.path.join(out_dir, f"best_model_task{task_index}.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "history": history,
                },
                checkpoint_path,
            )
            print(f"Task {task_index}: Saved best model checkpoint at epoch {epoch+1}")

    total_train_time = time.time() - total_start_time
    print(f"Task {task_index}: Training complete.")

    final_info = {
        "final_train_loss": history["train_loss"][-1],
        "best_val_loss": best_val_loss,
        "total_train_time": total_train_time,
        **metrics,
    }
    test_info = evaluate_on_test(task_index, csv_dir, checkpoint_path, batch_size)
    final_info.update(test_info)
    return model, history, final_info


def evaluate_on_test(
    task_index: int, csv_dir: str, checkpoint_path: str, batch_size: int = 64
) -> Dict[str, float]:
    """
    Load the best model checkpoint and evaluate on the test dataset.

    Returns:
        A dictionary with test loss and evaluation metrics.
    """
    _, _, _, _, X_test, y_test_all = load_data(csv_dir)
    y_test = y_test_all[:, task_index].unsqueeze(1)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    input_size = X_test.shape[1]
    hidden_size = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = StockPredictor(input_size, hidden_size, output_size=1, dropout=0.2).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.BCEWithLogitsLoss()
    test_loss, test_preds, test_labels = evaluate_model(test_loader, model, criterion, device)
    test_metrics = compute_metrics(test_labels, test_preds)
    return {"test_loss": test_loss, **test_metrics}


# =============================================================================
# Main Script
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train separate models for stock movement prediction tasks (3, 7, and 15 days)"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./run_0",
        help="Output directory for model checkpoints and logs",
    )
    args = parser.parse_args()

    # Hyperparameters and settings.
    epochs = 10
    batch_size = 16
    learning_rate = 1e-3
    seed = 0
    dropout = 0.2
    target_ticker = "86040"

    csv_dir = os.path.expanduser(
        f"~/Documents/AI-Scientist-Fin/price_data/processed/{target_ticker}"
    )
    tasks = {"3": 0, "7": 1, "15": 2}
    num_seeds = 3

    all_results = {}
    final_infos = {}

    # Run experiments over tasks and multiple seeds.
    for task_name, task_index in tasks.items():
        task_results = {}
        task_final_infos = {}
        for seed_offset in range(num_seeds):
            current_seed = seed + seed_offset
            model, history, final_info = train_task(
                task_index=task_index,
                csv_dir=csv_dir,
                out_dir=args.out_dir,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=current_seed,
                dropout=dropout,
            )
            key = f"stock_{task_name}_{seed_offset}"
            task_results[f"{key}_final_info"] = final_info
            task_results[f"{key}_train_info"] = history["train_loss"]
            task_results[f"{key}_val_info"] = history["val_loss"]
            task_final_infos[key] = final_info
        all_results[task_name] = task_results

        # Compute summary statistics.
        sample_info = list(task_final_infos.values())
        info_keys = sample_info[0].keys()
        final_info_dict = {k: [info[k] for info in sample_info] for k in info_keys}
        means = {f"{k}_mean": float(np.mean(v)) for k, v in final_info_dict.items()}
        stderrs = {f"{k}_stderr": float(np.std(v) / len(v)) for k, v in final_info_dict.items()}
        final_infos[task_name] = {
            "means": means,
            "stderrs": stderrs,
            "final_info_dict": final_info_dict,
        }

    # Save experiment results.
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "final_info.json"), "w") as f:
        json.dump(final_infos, f)
    with open(os.path.join(args.out_dir, "all_results.npy"), "wb") as f:
        np.save(f, all_results)

    print("Experiment complete. Final info and all results saved.")
