import json
from typing import Any, Dict, List

import requests


reasoning_start = "<start_working_out>"
reasoning_end = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

system_prompt = f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""


def _format_example(
    *,
    problem: str,
    answer: str,
    solution: str = "",
    source_dataset: str = "unknown",
    original_id: int = 0,
    global_id: int = 0,
    url: str = "",
) -> Dict[str, Any]:
    return {
        "global_id": global_id,
        "original_id": original_id,
        "source_dataset": source_dataset,
        "problem": problem,
        "answer": str(answer),
        "solution": solution,
        "url": url,
        "prompt": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Problem: {problem}\n\nSolve this step by step and provide your final numerical answer.",
            },
        ],
    }


def load_math500_dataset() -> List[Dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    examples = []
    for i, item in enumerate(dataset):
        examples.append(
            _format_example(
                problem=item.get("problem", ""),
                answer=item.get("answer", ""),
                solution=item.get("solution", ""),
                source_dataset="MATH-500",
                original_id=i,
                global_id=i,
            )
        )
    print(f"Loaded {len(examples)} MATH-500 examples")
    return examples


def load_amc_dataset() -> List[Dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("zwhe99/amc23")
    examples = []
    global_id = 0
    for split_name, split_data in dataset.items():
        for i, item in enumerate(split_data):
            answer = item.get("answer", "")
            try:
                answer = str(int(answer))
            except Exception:
                answer = str(answer)
            examples.append(
                _format_example(
                    problem=item.get("question", ""),
                    answer=answer,
                    solution=item.get("solution", ""),
                    source_dataset=split_name,
                    original_id=i,
                    global_id=global_id,
                    url=item.get("url", ""),
                )
            )
            global_id += 1
    print(f"Loaded {len(examples)} AMC23 examples")
    return examples


def load_minerva_dataset() -> List[Dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("math-ai/minervamath")
    examples = []
    global_id = 0
    for split_name, split_data in dataset.items():
        for i, item in enumerate(split_data):
            examples.append(
                _format_example(
                    problem=item.get("question", ""),
                    answer=item.get("answer", ""),
                    solution=item.get("solution", ""),
                    source_dataset=split_name,
                    original_id=i,
                    global_id=global_id,
                    url=item.get("url", ""),
                )
            )
            global_id += 1
    print(f"Loaded {len(examples)} Minerva examples")
    return examples


def load_aime_dataset() -> List[Dict[str, Any]]:
    urls = {
        "test2024": "https://raw.githubusercontent.com/GAIR-NLP/AIME-Preview/main/eval/data/aime/test2024.jsonl",
        "test2025-I": "https://raw.githubusercontent.com/GAIR-NLP/AIME-Preview/main/eval/data/aime/test2025-I.jsonl",
        "test2025-II": "https://raw.githubusercontent.com/GAIR-NLP/AIME-Preview/main/eval/data/aime/test2025-II.jsonl",
    }
    examples = []
    global_id = 0
    for source_dataset, url in urls.items():
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        for line_num, line in enumerate(response.text.strip().splitlines()):
            if not line.strip():
                continue
            item = json.loads(line)
            examples.append(
                _format_example(
                    problem=item.get("problem", ""),
                    answer=item.get("answer", ""),
                    solution=item.get("solution", ""),
                    source_dataset=source_dataset,
                    original_id=item.get("id", line_num),
                    global_id=global_id,
                    url=item.get("url", ""),
                )
            )
            global_id += 1
    print(f"Loaded {len(examples)} AIME examples")
    return examples


DATASET_LOADERS = {
    "MATH-500": load_math500_dataset,
    "AIME": load_aime_dataset,
    "AMC23": load_amc_dataset,
    "Minerva": load_minerva_dataset,
}
