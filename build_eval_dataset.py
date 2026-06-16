## first time inside this follde i need to run this 
## python3 build_eval_dataset.py --reason 500 --mcq 500 --truth 500 --force-rebuild
## then in my sis itcan use like python3 build_eval_dataset.py

#!/usr/bin/env python3

import os
import csv
import re
import random
import argparse
from datasets import load_dataset

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_MODELS_DIR = os.path.join(BASE_DIR, "test_models")

PROMPTS_FILE = os.path.join(TEST_MODELS_DIR, "prompts.txt")
ANSWERS_FILE = os.path.join(TEST_MODELS_DIR, "answers.csv")

# ----------------------------
# Default dataset sizes
# ----------------------------
N_REASON = 500
N_MCQ = 500
N_TRUTH = 500

RANDOM_SEED = 42


# ----------------------------
# Helpers
# ----------------------------
def extract_gsm8k_final_answer(answer_text: str) -> str:
    """
    GSM8K answers usually end with:
    #### 42
    This extracts the final answer.
    """
    if not answer_text:
        return ""

    match = re.search(r"####\s*(.*)", answer_text)
    if match:
        return match.group(1).strip()

    return answer_text.strip()


def safe_text(text: str) -> str:
    if text is None:
        return ""
    return str(text).replace("\n", " ").replace("\r", " ").strip()


def clean_truth_answer(text: str) -> str:
    """
    Keep the reference answer short and clean for automatic matching.
    """
    text = safe_text(text)
    text = re.sub(r"\s+", " ", text).strip()

    parts = re.split(r"(?<=[.!?])\s+", text)
    if parts and parts[0]:
        return parts[0].strip()

    return text


def dataset_files_exist() -> bool:
    return os.path.exists(PROMPTS_FILE) and os.path.exists(ANSWERS_FILE)


def count_nonempty_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return sum(1 for line in f if line.strip())


# ----------------------------
# Main builder
# ----------------------------
def generate_prompts_file(
    test_models_dir: str,
    n_reason: int = 100,
    n_mcq: int = 100,
    n_truth: int = 100,
):
    prompts_file = os.path.join(test_models_dir, "prompts.txt")
    answers_file = os.path.join(test_models_dir, "answers.csv")

    os.makedirs(test_models_dir, exist_ok=True)

    rows = []
    qid = 1

    # -----------------
    # GSM8K (reasoning)
    # -----------------
    print("Loading GSM8K...")
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    gsm = gsm.shuffle(seed=RANDOM_SEED).select(range(n_reason))

    for row in gsm:
        q = safe_text(row["question"])
        ans = extract_gsm8k_final_answer(row["answer"])

        prompt = (
            f"Question: {q} "
            f"Solve carefully. "
            f"End your response exactly in this format: Final Answer: <number>"
        )

        rows.append({
            "id": qid,
            "dataset": "gsm8k",
            "type": "reasoning",
            "prompt": prompt,
            "answer": ans,
        })
        qid += 1

    # -----------------
    # MMLU (MCQ)
    # -----------------
    print("Loading MMLU...")
    mmlu = load_dataset("cais/mmlu", "all", split="test")
    mmlu = mmlu.shuffle(seed=RANDOM_SEED).select(range(n_mcq))

    for row in mmlu:
        choices = row["choices"]
        answer_index = int(row["answer"])

        prompt = (
            f"Question: {safe_text(row['question'])} "
            f"Options: "
            f"A. {safe_text(choices[0])} "
            f"B. {safe_text(choices[1])} "
            f"C. {safe_text(choices[2])} "
            f"D. {safe_text(choices[3])} "
            f"Reply with only one capital letter: A, B, C, or D."
        )

        gold = ["A", "B", "C", "D"][answer_index]

        rows.append({
            "id": qid,
            "dataset": "mmlu",
            "type": "mcq",
            "prompt": prompt,
            "answer": gold,
        })
        qid += 1

    # -----------------
    # TruthfulQA
    # -----------------
    print("Loading TruthfulQA...")
    truthful = load_dataset("truthful_qa", "generation", split="validation")
    truthful = truthful.shuffle(seed=RANDOM_SEED).select(range(n_truth))

    for row in truthful:
        q = safe_text(row["question"])
        gold = clean_truth_answer(row.get("best_answer", ""))

        prompt = (
            f"Question: {q} "
            f"Answer truthfully in one short factual sentence. "
            f"If the question contains a false assumption, correct it directly. "
            f"Do not speculate."
        )

        rows.append({
            "id": qid,
            "dataset": "truthfulqa",
            "type": "truth",
            "prompt": prompt,
            "answer": gold,
        })
        qid += 1

    # -----------------
    # Shuffle all rows together
    # -----------------
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(rows)

    # -----------------
    # Reassign ids after shuffle
    # -----------------
    for new_id, row in enumerate(rows, start=1):
        row["id"] = new_id

    # -----------------
    # Write prompts.txt
    # -----------------
    with open(prompts_file, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(row["prompt"] + "\n")

    # -----------------
    # Write answers.csv
    # -----------------
    with open(answers_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "dataset", "type", "answer"])
        for row in rows:
            writer.writerow([row["id"], row["dataset"], row["type"], row["answer"]])

    print("\nDataset created successfully.")
    print(f"Prompts file: {prompts_file}")
    print(f"Answers file: {answers_file}")
    print(f"Reasoning prompts: {n_reason}")
    print(f"MCQ prompts: {n_mcq}")
    print(f"Truth prompts: {n_truth}")
    print(f"Total prompts: {len(rows)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reason", type=int, default=N_REASON, help="Number of GSM8K prompts")
    parser.add_argument("--mcq", type=int, default=N_MCQ, help="Number of MMLU prompts")
    parser.add_argument("--truth", type=int, default=N_TRUTH, help="Number of TruthfulQA prompts")
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild prompts.txt and answers.csv even if they already exist"
    )
    args = parser.parse_args()

    if dataset_files_exist() and not args.force_rebuild:
        total_prompts = count_nonempty_lines(PROMPTS_FILE)
        print("Dataset already exists. Reusing existing files.")
        print(f"Prompts file: {PROMPTS_FILE}")
        print(f"Answers file: {ANSWERS_FILE}")
        print(f"Existing total prompts: {total_prompts}")
        print("Use --force-rebuild if you want to create a new dataset.")
        return

    generate_prompts_file(
        TEST_MODELS_DIR,
        n_reason=args.reason,
        n_mcq=args.mcq,
        n_truth=args.truth,
    )


if __name__ == "__main__":
    main()
