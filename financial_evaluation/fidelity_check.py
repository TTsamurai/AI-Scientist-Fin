import sys
import os
import os.path as osp
import glob
import json
import argparse
from typing import Any, Dict, List, Optional

# Ensure project root is on PYTHONPATH so sibling package ai_scientist can be imported
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "..")))

from ai_scientist.llm import (
    AVAILABLE_LLMS,
    create_client,
    extract_json_between_markers,
    get_response_from_llm,
)

# Prompt template for fidelity checking. Keep each line under 100 characters.
FIDELITY_CHECK_PROMPT = """
{task_description}
<experiment.py>
{code}
</experiment.py>

Below is the idea file outlining expected experiment procedures:
<idea.json>
{idea}
</idea.json>

Please perform a fidelity check by answering the following:

1. Does the experiment code faithfully implement the design? (1 = Yes, 0 = No)
2. Briefly explain your answer, citing specific alignments or deviations.

Respond in JSON exactly as:
    {{"Fidelity": <1 or 0>, "Explanation": "<your explanation>"}}
"""


def perform_fidelity_check(
    client: Any,
    model: str,
    task_description: str,
    code: str,
    idea: str,
) -> Dict[str, Any]:
    """
    Send the fidelity check prompt to the LLM and parse the JSON response.

    Args:
        client: LLM client instance.
        model: Model identifier.
        task_description: High-level task instructions.
        code: Source code to be checked.
        idea: JSON-formatted idea specification.

    Returns:
        Parsed JSON dict containing keys "Fidelity" and "Explanation".
    """
    prompt = FIDELITY_CHECK_PROMPT.format(
        task_description=task_description,
        code=code,
        idea=idea,
    )
    response_text, _ = get_response_from_llm(
        prompt,
        client=client,
        model=model,
        system_message=(
            "You are a helpful assistant that performs fidelity checks" " on experiment code."
        ),
        msg_history=[],
    )
    return extract_json_between_markers(response_text)


def load_json(path: str) -> Optional[Any]:
    """
    Load JSON content from a file, returning None on failure.

    Args:
        path: Filesystem path to JSON file.

    Returns:
        Parsed JSON data or None.
    """
    try:
        with open(path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"[ERROR] Unable to load JSON {path!r}: {exc}")
        return None


def read_text(path: str) -> Optional[str]:
    """
    Read and return text content from a file, or None on error.
    """
    try:
        with open(path, "r", encoding="utf-8") as fp:
            return fp.read()
    except OSError as exc:
        print(f"[ERROR] Unable to read file {path!r}: {exc}")
        return None


def find_latest_script(directory: str, candidates: List[str]) -> Optional[str]:
    """
    Return the first candidate script filename that exists in the directory.
    """
    for name in candidates:
        path = osp.join(directory, name)
        if osp.exists(path):
            return name
    return None


def main() -> None:
    """
    Main entrypoint: parses arguments, loads ideas, and runs fidelity checks.
    """
    parser = argparse.ArgumentParser(
        description="Run LLM-based fidelity checks on experiment code."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="financial_prediction",
        help="Experiment directory to evaluate",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=AVAILABLE_LLMS,
        default="gpt-4o-2024-05-13",
        help="LLM model to use",
    )
    args = parser.parse_args()

    client, _ = create_client(args.model)

    base_dir = osp.join("../templates", args.experiment)
    results_dir = osp.join("../results", args.experiment)
    result_dirs = glob.glob(osp.join(results_dir, "*"))

    idea_files = [
        "./idea_results/unique_gpt-4.5-preview_ideas_financial_prediction_ideas.json",
        "./idea_results/unique_gpt-4o-diverse_ideas_financial_prediction_ideas.json",
        "./idea_results/unique_gpt-4o_ideas_financial_prediction_ideas.json",
        "./idea_results/unique_gpt-4o_multi_agent_ideas_financial_prediction_ideas.json",
    ]

    prompt_json = load_json(osp.join(base_dir, "prompt.json")) or {}
    task_description = prompt_json.get("task_description", "")

    script_candidates = ["run_3.py", "run_2.py", "run_1.py"]

    for idea_file in idea_files:
        print(f"Evaluating {idea_file}...")
        ideas = load_json(osp.join(base_dir, idea_file)) or []
        for idea in ideas:
            if not idea.get("implementation_success"):
                continue

            name = idea.get("Name") or ""
            dirs = [d for d in result_dirs if name in d]

            if not dirs:
                continue
            result_dir = dirs[-1]

            script_name = find_latest_script(result_dir, script_candidates)
            if not script_name:
                print(f"[WARN] No run script in {result_dir}")
                continue

            code = read_text(osp.join(result_dir, script_name))
            if code is None:
                continue
            idea_str = json.dumps(idea, indent=4)
            result = perform_fidelity_check(
                client,
                args.model,
                task_description,
                code,
                idea_str,
            )
            fid = result.get("Fidelity")
            expl = result.get("Explanation", "")
            print(f"Fidelity: {fid}")
            print(f"Explanation: {expl}\n")


if __name__ == "__main__":
    main()
