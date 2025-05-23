import json
import os
import os.path as osp
from ai_scientist.llm import (
    get_response_from_llm,
    extract_json_between_markers,
    create_client,
    AVAILABLE_LLMS,
)

S2_API_KEY = os.getenv("S2_API_KEY")


IDEA_MODERATOR_PROMPT = """{task_description}
<experiment.py>
{code}
</experiment.py>


Here are the ideas generated by various experts:

'''
{agent_ideas_string}
'''


Here are the ideas that you have already generated:

'''
{prev_ideas_string}
'''

You are a moderator tasked with synthesizing the above ideas into an impactful and creative idea for research experiments and directions you can feasibly investigate with the code provided. Your goal is to integrate the strengths and insights of each idea while addressing any overlaps or gaps. Note that you will not have access to any additional resources or datasets. Try to generate ideas that are more diverse than the previous ones, while ensuring that both the ideas and the code remain implementable. Do not employ repetitive title and ideas. Do not try hard to include all the opinions from the experts. Instead, focus on the most promising and feasible ideas that can be implemented with the provided code. If you think combining ideas is necessary, do so in a way that enhances the overall concept.

Respond in the following format:

THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

In <THOUGHT>, first briefly discuss your intuitions and motivations for the idea. Detail your high-level plan, necessary design choices and ideal outcomes of the experiments. Justify how the idea is different from the existing ones.

In <JSON>, provide the new idea in JSON format with the following fields:
- "Name": A shortened descriptor of the idea. Lowercase, no spaces, underscores allowed.
- "Title": A title for the idea, will be used for the report writing.
- "Experiment": An outline of the implementation. E.g. which functions need to be added or modified, how results will be obtained, ...
- "Interestingness": A rating from 1 to 10 (lowest to highest).
- "Feasibility": A rating from 1 to 10 (lowest to highest).
- "Novelty": A rating from 1 to 10 (lowest to highest).

Be cautious and realistic on your ratings.
This JSON will be automatically parsed, so ensure the format is precise.
You will have {num_reflections} rounds to iterate on the idea, but do not need to use them all.
"""

IDEA_REFLECTION_PROMPT = """Round {current_round}/{num_reflections}.
In your thoughts, first carefully consider the quality, novelty, and feasibility of the idea you just created.
Include any other factors that you think are important in evaluating the idea.
Ensure the idea is clear and concise, and the JSON is the correct format.
Do not make things overly complicated.
In the next attempt, try and refine and improve your idea.
Stick to the spirit of the original idea unless there are glaring issues.

Respond in the same format as before:
THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

If there is nothing to improve, simply repeat the previous JSON EXACTLY after the thought and include "I am done" at the end of the thoughts but before the JSON.
ONLY INCLUDE "I am done" IF YOU ARE MAKING NO MORE CHANGES."""

ROLE_PROMPTS = {
    "statistician": """
    {task_description} 
    <experiment.py>
    {code}
    </experiment.py>
    Here are the ideas that you have already generated:

    '''
    {prev_ideas_string}
    '''
    You are a statistician with expertise in financial data analysis focused on predicting the stock price of our target company: Nomura Holdings [86040]. Remember, you do not have access to textual data such as news, earnings summaries, or macroeconomic data; only historical price data is available.
    
    Analyze the current experiment for stock movement predictions and suggest an approach that leverages robust statistical methods, hypothesis testing, and advanced metrics. Ensure your idea is distinct from the previous ones.
    
    Respond in the following format:

    THOUGHT:
    <THOUGHT>

    NEW IDEA JSON:
    ```json
    <JSON>
    ```

    In <THOUGHT>, briefly discuss your intuitions, motivations, and high-level plan for the idea. Outline the necessary design choices and ideal outcomes of the experiments, and justify how your idea differs from existing ones.

    In <JSON>, provide the new idea with the following fields:
    - "Name": A shortened descriptor of the idea (lowercase, no spaces, underscores allowed).
    - "Title": A title for the idea, to be used in report writing.
    - "Experiment": An outline of the implementation (e.g., functions to add or modify, how results will be obtained).
    - "Interestingness": A rating from 1 to 10.
    - "Feasibility": A rating from 1 to 10.
    - "Novelty": A rating from 1 to 10.
    
    This JSON will be automatically parsed, so ensure the format is precise.
    """,
    "ml_engineer": """
    {task_description} 
    <experiment.py>
    {code}
    </experiment.py>
    Here are the ideas that you have already generated:

    '''
    {prev_ideas_string}
    '''
    You are a machine learning engineer specializing in time-series analysis and model optimization, focused on predicting the stock price of our target company: Nomura Holdings [86040]. You do not have access to textual data such as news, earnings summaries, or macroeconomic data; only historical price data is available.
    
    Propose an innovative deep learning model or feature engineering method for predicting stock movements. Focus on techniques that improve prediction accuracy and computational efficiency, and ensure your idea is distinct from previous ones.
    
    Respond in the following format:

    THOUGHT:
    <THOUGHT>

    NEW IDEA JSON:
    ```json
    <JSON>
    ```

    In <THOUGHT>, briefly discuss your intuitions, motivations, and high-level plan for the idea. Outline the necessary design choices and ideal outcomes of the experiments, and justify how your idea differs from existing ones.

    In <JSON>, provide the new idea with the following fields:
    - "Name": A shortened descriptor of the idea (lowercase, no spaces, underscores allowed).
    - "Title": A title for the idea, to be used in report writing.
    - "Experiment": An outline of the implementation (e.g., functions to add or modify, how results will be obtained).
    - "Interestingness": A rating from 1 to 10.
    - "Feasibility": A rating from 1 to 10.
    - "Novelty": A rating from 1 to 10.
    
    This JSON will be automatically parsed, so ensure the format is precise.
    """,
    "financial_analyst": """
    {task_description} 
    <experiment.py>
    {code}
    </experiment.py>
    Here are the ideas that you have already generated:

    '''
    {prev_ideas_string}
    '''
    You are a financial analyst with deep market knowledge focused on predicting the stock price of our target company: Nomura Holdings [86040]. You do not have access to textual data such as news, earnings summaries, or macroeconomic data; only historical price data is available.
    
    Discuss market indicators, risk management strategies, and the practical application of stock movement predictions. Ensure your idea is applicable to real-world trading and addresses potential market dynamics, while being distinct from previous ideas.
    
    Respond in the following format:

    THOUGHT:
    <THOUGHT>

    NEW IDEA JSON:
    ```json
    <JSON>
    ```

    In <THOUGHT>, briefly discuss your intuitions, motivations, and high-level plan for the idea. Outline the necessary design choices and ideal outcomes of the experiments, and justify how your idea differs from existing ones.

    In <JSON>, provide the new idea with the following fields:
    - "Name": A shortened descriptor of the idea (lowercase, no spaces, underscores allowed).
    - "Title": A title for the idea, to be used in report writing.
    - "Experiment": An outline of the implementation (e.g., functions to add or modify, how results will be obtained).
    - "Interestingness": A rating from 1 to 10.
    - "Feasibility": A rating from 1 to 10.
    - "Novelty": A rating from 1 to 10.
    
    This JSON will be automatically parsed, so ensure the format is precise.
    """,
}

MODERATOR_SYSTEM = """You are a moderator tasked with synthesizing the above ideas into an impactful and creative idea for research experiments and directions you can feasibly investigate with the code provided. Your goal is to integrate the strengths and insights of each idea while addressing any overlaps or gaps. Note that you will not have access to any additional resources or datasets. Try to generate ideas that are more diverse than the previous ones, while ensuring that both the ideas and the code remain implementable. You do not have access to textual data such as news, earnings summaries, or macroeconomic data; you only have historical price data available."""


def generate_role_idea(
    role: str, task_description: str, code: str, prev_ideas_string: str, client, model
) -> dict:
    """
    Generate an idea from a specific role by calling the LLM with a tailored prompt.
    """
    role_prompt = ROLE_PROMPTS.get(role)
    if not role_prompt:
        raise ValueError(f"No prompt found for role: {role}")

    # Construct the prompt message including context (e.g., experiment code, previous ideas)
    text, _ = get_response_from_llm(
        role_prompt.format(
            task_description=task_description,
            code=code,
            prev_ideas_string=prev_ideas_string,
        ),
        client=client,
        model=model,
        system_message=f"Role: {role}",
        msg_history=[],
    )
    idea_json = extract_json_between_markers(text)
    if idea_json is None:
        raise ValueError(f"Failed to extract JSON from {role} response.")
    return idea_json


def generate_multi_agent_idea(
    base_dir: str,
    client,
    model: str,
    skip_generation=False,
    max_num_generations=10,
    num_reflections=1,
    disable_seed_ideas=False,
    save_file_name=None,
    diverse_ideas=False,
):
    if skip_generation:
        # Load existing ideas from file
        try:
            with open(osp.join(base_dir, "ideas.json"), "r", encoding="utf-8") as f:
                ideas = json.load(f)
            print("Loaded existing ideas:")
            for idea in ideas:
                print(idea)
            return ideas
        except FileNotFoundError:
            print("No existing ideas found. Generating new ideas.")
        except json.JSONDecodeError:
            print("Error decoding existing ideas. Generating new ideas.")

    idea_str_archive = []
    if not disable_seed_ideas:
        with open(osp.join(base_dir, "seed_ideas.json"), "r", encoding="utf-8") as f:
            seed_ideas = json.load(f)
        for seed_idea in seed_ideas:
            idea_str_archive.append(json.dumps(seed_idea))
    else:
        pass

    with open(osp.join(base_dir, "experiment.py"), "r", encoding="utf-8") as f:
        code = f.read()

    with open(osp.join(base_dir, "prompt.json"), "r", encoding="utf-8") as f:
        prompt = json.load(f)

    idea_system_prompt = MODERATOR_SYSTEM

    for _ in range(max_num_generations):
        print()
        print(f"Generating idea {_ + 1}/{max_num_generations}")
        try:
            prev_ideas_string = "\n\n".join(idea_str_archive)
            msg_history = []
            print(f"Iteration 1/{num_reflections}")
            # Multi-agent idea generation
            stat_idea = generate_role_idea(
                role="statistician",
                task_description=prompt["task_description"],
                code=code,
                prev_ideas_string="\n\n".join(idea_str_archive),
                client=client,
                model=model,
            )
            ml_idea = generate_role_idea(
                role="ml_engineer",
                task_description=prompt["task_description"],
                code=code,
                prev_ideas_string="\n\n".join(idea_str_archive),
                client=client,
                model=model,
            )
            fin_idea = generate_role_idea(
                role="financial_analyst",
                task_description=prompt["task_description"],
                code=code,
                prev_ideas_string="\n\n".join(idea_str_archive),
                client=client,
                model=model,
            )

            # Combine ideas from all roles with role information
            combined_ideas = [
                {"role": "statistician", "idea": stat_idea},
                {"role": "ml_engineer", "idea": ml_idea},
                {"role": "financial_analyst", "idea": fin_idea},
            ]
            # Generate a moderator idea based on the combined ideas
            combined_ideas_str = "\n\n".join(
                [
                    f"Role: {idea['role']}\nIdea: {json.dumps(idea['idea'])}"
                    for idea in combined_ideas
                ]
            )
            text, msg_history = get_response_from_llm(
                IDEA_MODERATOR_PROMPT.format(
                    task_description=prompt["task_description"],
                    code=code,
                    prev_ideas_string=prev_ideas_string,
                    agent_ideas_string=combined_ideas_str,
                    num_reflections=num_reflections,
                ),
                client=client,
                model=model,
                system_message=idea_system_prompt,
                msg_history=msg_history,
            )
            # Json output
            json_output = extract_json_between_markers(text)
            assert json_output is not None, "Failed to extract JSON from LLM output"
            print(json_output)
            # Iteratively improve task.
            if num_reflections > 1:
                for j in range(num_reflections - 1):
                    print(f"Iteration {j + 2}/{num_reflections}")
                    text, msg_history = get_response_from_llm(
                        IDEA_REFLECTION_PROMPT.format(
                            current_round=j + 2, num_reflections=num_reflections
                        ),
                        client=client,
                        model=model,
                        system_message=idea_system_prompt,
                        msg_history=msg_history,
                    )
                    ## PARSE OUTPUT
                    json_output = extract_json_between_markers(text)
                    assert json_output is not None, "Failed to extract JSON from LLM output"
                    print(json_output)
                    if "I am done" in text:
                        print(f"Idea generation converged after {j + 2} iterations.")
                        break
            idea_str_archive.append(json.dumps(json_output))
        except Exception as e:
            print(f"Failed to generate idea: {e}")
            continue
    ## SAVE IDEAS
    ideas = []
    for idea_str in idea_str_archive:
        ideas.append(json.loads(idea_str))
    # Save results to JSON file
    if save_file_name is not None:
        idea_file_name = "_".join([save_file_name, "multi_agent_ideas.json"])
        with open(osp.join(base_dir, idea_file_name), "w", encoding="utf-8") as f:
            json.dump(ideas, f, indent=4)
    else:
        idea_file_name = "multi_agent_ideas.json"
        with open(osp.join(base_dir, idea_file_name), "w", encoding="utf-8") as f:
            json.dump(ideas, f, indent=4)
    return ideas


if __name__ == "__main__":
    MAX_NUM_GENERATIONS = 100
    NUM_REFLECTIONS = 1
    import argparse

    parser = argparse.ArgumentParser(description="Generate AI scientist ideas")
    # add type of experiment (nanoGPT, Boston, etc.)
    parser.add_argument(
        "--experiment",
        type=str,
        default="nanoGPT",
        help="Experiment to run AI Scientist on.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-05-13",
        choices=AVAILABLE_LLMS,
        help="Model to use for AI Scientist.",
    )
    parser.add_argument(
        "--skip-idea-generation",
        action="store_true",
        help="Skip idea generation and use existing ideas.",
    )
    parser.add_argument(
        "--check-novelty",
        action="store_true",
        help="Check novelty of ideas.",
    )
    parser.add_argument(
        "--disable_seed_ideas",
        action="store_true",
        help="Disable seed ideas for idea generation.",
    )
    parser.add_argument(
        "--save_file_name",
        type=str,
        default=None,
        help="File name to save the generated ideas.",
    )
    parser.add_argument(
        "--diverse_ideas",
        action="store_true",
        help="Generate diverse ideas.",
    )
    args = parser.parse_args()

    # Create client
    client, client_model = create_client(args.model)

    base_dir = osp.join("templates", args.experiment)
    results_dir = osp.join("results", args.experiment)
    ideas = generate_multi_agent_idea(
        base_dir,
        client=client,
        model=client_model,
        skip_generation=args.skip_idea_generation,
        max_num_generations=MAX_NUM_GENERATIONS,
        num_reflections=NUM_REFLECTIONS,
        disable_seed_ideas=args.disable_seed_ideas,
        save_file_name=args.save_file_name,
        diverse_ideas=args.diverse_ideas,
    )
