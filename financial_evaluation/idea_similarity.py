import argparse
import json
import os
import os.path as osp

from typing import List

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

SIMILARITY_THRESHOLD = 0.8
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_ideas(file_path: str) -> List[dict]:
    if not osp.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def encode_texts(model, tokenizer, texts: List[str]) -> np.ndarray:
    encoded_input = tokenizer(
        texts, padding=True, truncation=True, return_tensors="pt", max_length=512
    )
    with torch.no_grad():
        outputs = model(**encoded_input.to(model.device))
        embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
    return embeddings


def compute_unique_indices(embeddings: np.ndarray, threshold: float) -> np.ndarray:
    similarity_matrix = cosine_similarity(embeddings)
    duplicates = np.triu(similarity_matrix > threshold, 1)
    return ~np.any(duplicates, axis=0)


def extract_unique_ideas(ideas: List[dict], indices: np.ndarray) -> List[dict]:
    return [idea for i, idea in enumerate(ideas) if indices[i]]


def main():
    parser = argparse.ArgumentParser(description="Check for duplicate research ideas.")
    parser.add_argument("--experiment", type=str, default="financial_prediction")
    parser.add_argument("--idea_file_name", type=str, default="ideas.json")
    args = parser.parse_args()

    base_dir = osp.join("../templates", args.experiment)
    file_path = osp.join(base_dir, args.idea_file_name)

    all_ideas = load_ideas(file_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to("cuda" if torch.cuda.is_available() else "cpu")

    unique_title_num, unique_experiment_num, unique_name_num = [], [], []

    print(f"Length of all_ideas: {len(all_ideas)}")

    for ind in range(1, 101, 5):
        ideas = all_ideas[:ind]
        name_ideas = [" ".join(idea["Name"].split("_")) for idea in ideas]
        title_ideas = [idea["Title"].lower() for idea in ideas]
        experiment_ideas = [idea["Experiment"] for idea in ideas]

        name_indices = compute_unique_indices(
            encode_texts(model, tokenizer, name_ideas), SIMILARITY_THRESHOLD
        )
        title_indices = compute_unique_indices(
            encode_texts(model, tokenizer, title_ideas), SIMILARITY_THRESHOLD
        )
        experiment_indices = compute_unique_indices(
            encode_texts(model, tokenizer, experiment_ideas), SIMILARITY_THRESHOLD
        )

        unique_name_num.append(np.sum(name_indices))
        unique_title_num.append(np.sum(title_indices))
        unique_experiment_num.append(np.sum(experiment_indices))

    print("Unique Titles:", unique_title_num)
    print("Unique Experiments:", unique_experiment_num)
    print("Unique Names:", unique_name_num)

    # Extract and save final unique ideas based on titles
    final_unique_ideas = extract_unique_ideas(
        all_ideas[:100],
        compute_unique_indices(
            encode_texts(model, tokenizer, [idea["Title"].lower() for idea in all_ideas[:100]]),
            SIMILARITY_THRESHOLD,
        ),
    )
    output_path = osp.join(base_dir, f"unique_{args.idea_file_name}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_unique_ideas, f, indent=4)
    print(f"Saved {len(final_unique_ideas)} unique ideas to {output_path}")


if __name__ == "__main__":
    main()
