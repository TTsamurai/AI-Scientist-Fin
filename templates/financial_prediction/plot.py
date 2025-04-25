import os
import os.path as osp
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Define the tasks corresponding to prediction horizons
tasks = ["3", "7", "15"]

# Find all run directories (e.g., run_0, run_1, …)
folders = [folder for folder in os.listdir("./") if folder.startswith("run") and osp.isdir(folder)]

# Load final aggregated info from each run and individual results
final_infos = {}  # final_info.json content per run
all_results = {}  # all_results.npy content per run

for folder in folders:
    final_info_path = osp.join(folder, "final_info.json")
    results_path = osp.join(folder, "all_results.npy")
    if osp.exists(final_info_path) and osp.exists(results_path):
        with open(final_info_path, "r") as f:
            final_infos[folder] = json.load(f)
        all_results[folder] = np.load(results_path, allow_pickle=True).item()


# Create a color palette for runs
def generate_color_palette(n):
    cmap = plt.get_cmap("tab20")
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, n)]


run_folders = list(final_infos.keys())
run_colors = generate_color_palette(len(run_folders))

# For each run folder, extract loss histories for each task.
results_info = {}  # structure: {run_folder: {task: {iters, train_loss, val_loss, ...}}}
for folder in run_folders:
    run_dict = all_results[folder]
    run_info = {}
    # Each run_dict is assumed to be organized by task, e.g.:
    # run_dict["3"] is a dict with keys like "stock_3_0_train_info", "stock_3_0_val_info", etc.
    for task in tasks:
        task_dict = run_dict.get(task, {})
        train_losses = []
        val_losses = []
        test_accuracy = []
        test_precision = []
        test_recall = []
        test_f1 = []
        # We assume keys contain "train_info" and "val_info" for each seed.
        for key, value in task_dict.items():
            if key.endswith("_train_info"):
                train_losses.append(np.array(value))
            elif key.endswith("_val_info"):
                val_losses.append(np.array(value))
            # elif key.endswith("_final_info"):
            #     # Assuming final_info contains test metrics
            #     test_accuracy.append(value.get("test_accuracy", 0))
            #     test_precision.append(value.get("test_precision", 0))
            #     test_recall.append(value.get("test_recall", 0))
            #     test_f1.append(value.get("test_f1", 0))
        if len(train_losses) > 0:
            train_losses = np.stack(train_losses, axis=0)  # shape (n_seeds, epochs)
            mean_train = np.mean(train_losses, axis=0)
            std_train = np.std(train_losses, axis=0) / np.sqrt(train_losses.shape[0])
        else:
            mean_train = []
            std_train = []
        if len(val_losses) > 0:
            val_losses = np.stack(val_losses, axis=0)
            mean_val = np.mean(val_losses, axis=0)
            std_val = np.std(val_losses, axis=0) / np.sqrt(val_losses.shape[0])
        else:
            mean_val = []
            std_val = []

        # Assume epochs are 1-indexed and equal in length across seeds.
        epochs = np.arange(1, len(mean_train) + 1) if len(mean_train) > 0 else []
        run_info[task] = {
            "iters": epochs.tolist(),
            "train_loss": mean_train.tolist() if len(mean_train) > 0 else [],
            "train_loss_sterr": std_train.tolist() if len(std_train) > 0 else [],
            "val_loss": mean_val.tolist() if len(mean_val) > 0 else [],
            "val_loss_sterr": std_val.tolist() if len(std_val) > 0 else [],
        }
    results_info[folder] = run_info

# Plot Loss Curves for each task across runs
for task in tasks:
    plt.figure(figsize=(10, 6))
    for i, folder in enumerate(run_folders):
        iters = results_info[folder][task]["iters"]
        mean_train = results_info[folder][task]["train_loss"]
        train_sterr = results_info[folder][task]["train_loss_sterr"]
        mean_val = results_info[folder][task]["val_loss"]
        val_sterr = results_info[folder][task]["val_loss_sterr"]
        if iters:
            plt.plot(iters, mean_train, label=f"{folder} Train", color=run_colors[i])
            plt.fill_between(
                iters,
                np.array(mean_train) - np.array(train_sterr),
                np.array(mean_train) + np.array(train_sterr),
                color=run_colors[i],
                alpha=0.2,
            )
            plt.plot(iters, mean_val, label=f"{folder} Val", color=run_colors[i], linestyle="--")
            plt.fill_between(
                iters,
                np.array(mean_val) - np.array(val_sterr),
                np.array(mean_val) + np.array(val_sterr),
                color=run_colors[i],
                alpha=0.2,
            )
    plt.title(f"Loss Curves for {task}-Day Stock Movement Prediction")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"loss_curves_task{task}.png")
    plt.close()

# Plot Evaluation Metrics: For each task, use final_info from each run folder.
metrics = ["test_accuracy_mean", "test_precision_mean", "test_recall_mean", "test_f1_mean"]
x_labels = ["Accuracy", "Precision", "Recall", "F1"]
task_metric_means = {}  # {task: [mean_acc, mean_prec, mean_rec, mean_f1]}
task_metric_stderrs = {}  # {task: [stderr_acc, stderr_prec, stderr_rec, stderr_f1]}

for task in tasks:
    metric_values = {m: [] for m in metrics}
    for folder in run_folders:
        # final_infos[folder] is a dict mapping task keys ("3", "7", "15") to info
        task_info = final_infos[folder].get(task, {})
        # Here, task_info is assumed to have a "means" dict with the required metrics.
        if "means" in task_info:
            for m in metrics:
                metric_values[m].append(task_info["means"].get(m, 0))
    # Compute means and stderrs for each metric across run folders for this task.
    means = [np.mean(metric_values[m]) for m in metrics]
    stderrs = [np.std(metric_values[m]) / len(metric_values[m]) for m in metrics]
    task_metric_means[task] = means
    task_metric_stderrs[task] = stderrs

# Create grouped bar chart of evaluation metrics for each task.
x = np.arange(len(metrics))
width = 0.2
plt.figure(figsize=(10, 6))
task_colors = generate_color_palette(len(tasks))
for i, task in enumerate(tasks):
    plt.bar(
        x + i * width,
        task_metric_means[task],
        width,
        yerr=task_metric_stderrs[task],
        capsize=5,
        label=f"{task}-Day",
        color=task_colors[i],
    )
plt.xticks(x + width, x_labels)
plt.ylabel("Score")
plt.title("Validation Metrics for Stock Movement Prediction Tasks")
plt.legend()
plt.grid(True, axis="y", ls="--", alpha=0.7)
plt.tight_layout()
plt.savefig("evaluation_metrics.png")
plt.close()

print("Plots saved.")
