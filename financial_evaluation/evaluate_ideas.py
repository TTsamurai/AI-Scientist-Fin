import os
import glob
import json
import argparse
from typing import Dict, List, Any, Optional


def calculate_implementation_rate(ideas: List[Dict[str, Any]]) -> float:
    """
    Compute the fraction of ideas marked as successfully implemented.

    Args:
        ideas: List of idea dicts, each may contain an "implementation_success" key.

    Returns:
        A float between 0 and 1 representing the implementation rate.
        Returns 0 if the list is empty.
    """
    if not ideas:
        return 0.0
    implemented = sum(1 for idea in ideas if idea.get("implementation_success"))
    return implemented / len(ideas)


def load_json(path: str) -> Optional[Any]:
    """
    Load a JSON file and return its contents. Return None if loading fails.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON data, or None on error.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"[ERROR] Failed to load {path!r}: {exc}")
        return None


def append_day_metrics(
    evaluation: Dict[str, Any], model: str, day: str, means: Dict[str, float]
) -> None:
    """
    Append mean metrics for a specific day to the evaluation dict.

    Args:
        evaluation: The main evaluation dictionary.
        model: Model name key in the evaluation dict.
        day: Evaluation horizon ('3', '7', '15').
        means: Dict mapping metric names to their mean values.
    """
    for metric in ("accuracy", "precision", "recall", "f1", "best_val_loss"):
        key = f"{metric}_mean"
        if key in means:
            evaluation[model][day][metric].append(means[key])


def evaluate_models(
    base_dir: str, results_paths: List[str], idea_files: List[str]
) -> Dict[str, Any]:
    """
    Evaluate implementation rates and performance metrics for each model.

    Args:
        base_dir: Directory containing idea JSON files.
        results_paths: List of directories with result subfolders.
        idea_files: Filenames of idea JSONs to process.

    Returns:
        A dict mapping model names to their evaluation summaries.
    """
    suffix = "_ideas_financial_prediction_ideas.json"
    evaluation: Dict[str, Any] = {}

    for file_name in idea_files:
        model = file_name.split(suffix, 1)[0]
        evaluation[model] = {"implementation_rate": 0.0}

    for file_name in idea_files:
        model = file_name.split(suffix, 1)[0]
        ideas_path = os.path.join(base_dir, file_name)
        ideas = load_json(ideas_path) or []

        # Implementation rate
        rate = calculate_implementation_rate(ideas)
        evaluation[model]["implementation_rate"] = rate

        # Initialize per-day metric lists
        for day in ("3", "7", "15"):
            evaluation[model][day] = {
                m: [] for m in ("accuracy", "precision", "recall", "f1", "best_val_loss")
            }

        # Gather metrics for each implemented idea
        for idea in ideas:
            if not idea.get("implementation_success"):
                continue

            name = idea.get("Name")
            if not name:
                print(f"[WARN] Idea missing Name field: {idea}")
                continue

            # Find matching result path
            candidates = [p for p in results_paths if name in p]
            if not candidates:
                print(f"[WARN] No result path contains {name!r}")
                continue
            result_path = candidates[-1]

            # Locate a valid run directory
            run_dir = next(
                (
                    r
                    for r in ("run_2", "run_1", "run_0")
                    if os.path.exists(os.path.join(result_path, r))
                ),
                None,
            )
            if not run_dir:
                print(f"[WARN] No run_* directory in {result_path!r}")
                continue

            result_file = os.path.join(result_path, run_dir, "final_info.json")
            data = load_json(result_file)
            if not data:
                continue

            for day in ("3", "7", "15"):
                means = data.get(day, {}).get("means", {})
                append_day_metrics(evaluation, model, day, means)

        # Select best run (min loss) for each day
        for day in ("3", "7", "15"):
            metrics = evaluation[model][day]
            losses = metrics.get("best_val_loss", [])
            if not losses:
                continue
            best_idx = losses.index(min(losses))
            for m in metrics:
                metrics[m] = metrics[m][best_idx]

    return evaluation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate idea implementation and performance metrics"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="financial_prediction",
        help="Name of the experiment directory",
    )
    args = parser.parse_args()

    base_dir = os.path.join("..", "templates", args.experiment)
    results_dir = os.path.join("..", "results", args.experiment)
    results_paths = glob.glob(os.path.join(results_dir, "*"))

    idea_files = [
        "./idea_results/unique_gpt-4.5-preview_ideas_financial_prediction_ideas.json",
        "./idea_results/unique_gpt-4o-diverse_ideas_financial_prediction_ideas.json",
        "./idea_results/unique_gpt-4o_ideas_financial_prediction_ideas.json",
        "./idea_results/unique_gpt-4o_multi_agent_ideas_financial_prediction_ideas.json",
    ]

    evaluation = evaluate_models(base_dir, results_paths, idea_files)

    for model, stats in evaluation.items():
        rate = stats.get("implementation_rate", 0.0)
        print(f"{model} implementation rate: {rate:.2%}")


if __name__ == "__main__":
    main()
