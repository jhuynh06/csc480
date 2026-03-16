"""
build_connections_dataset.py

Builds fine-tuning datasets for NYT Connections puzzle solving.
Two datasets:
  1) Simple chain-of-thought 
  2) Distilled reasoning via Claude Sonnet 4.6

Both get cached to JSON for resume functionality.
"""

import os
import json
import time
import random
import requests
import anthropic
from datasets import Dataset, DatasetDict

PUZZLE_URL = (
    "https://raw.githubusercontent.com/Eyefyre/"
    "NYT-Connections-Answers/main/connections.json"
)

def fetch_puzzles():
    resp = requests.get(PUZZLE_URL)
    resp.raise_for_status()
    return resp.json()


SOLVE_PROMPT = """[INST]
# CONTEXT #
You are playing a linguistic game that requires chain-of-thought reasoning and has only one correct answer.
Given 16 words, split 16 words into 4 groups of 4 words, so each group is related by a certain theme or category.
Provide your reasoning for the groupings along with your answers.
Each of the 16 words is in EXACTLY ONE of 4 groups. Each group has EXACTLY 4 words. Think through this step by step.

You should start by identifying the easiest group, which is usually four items that belong in the same category of objects.

Here are some hints.
1. A reason 4 words are in a group can be that they are items that belong in the same category.
2. A reason 4 words are in a group can be that they each form a common phrase when paired with another word.
3. A reason 4 words are in a group can be that they are associated with a certain verb phrase or action.

#########
# OBJECTIVE #
Provide your reasoning and the correct 4 groups of 4 words for these 16 words: {input_words}

#########
# RESPONSE FORMAT #
You answer should follow the following format exactly. Otherwise you will suffer consequences beyond imagination.
Respond with chain-of-thought reasoning for your grouping choices followed by "So the answer is:" and a JSON object with a single key "answer" that has a list of 4 lists, each of which has 4 words.
Make sure each of the 16 words appears in EXACTLY one group and each group has EXACTLY 4 words.
DO NOT include anything after this JSON object.
[/INST]"""

TEACHER_PROMPT = """You are a teacher explaining how to solve a word grouping puzzle to a student.

The puzzle has 16 words that must be split into 4 groups of 4. Here are the words:
{words}

Here is the answer key:
{answer_key}

Explain step by step how to arrive at each group, starting with the easiest. For each group:
- What pattern or connection links these 4 words
- Why each word fits
- What tricky words might have seemed like they belong elsewhere

Be concise. No headers or bullet points. Write it as a continuous paragraph of reasoning. Do not restate the final answer at the end."""


def make_simple_reasoning(groups, descriptions):
    """Quick template-based CoT — no API calls needed."""
    parts = []
    for members, desc in zip(groups, descriptions):
        word_list = ", ".join(f"'{w}'" for w in members)
        parts.append(f"I notice that {word_list} are all {desc}.")
    return " ".join(parts)


def make_distilled_reasoning(client, shuffled_words_str, groups, descriptions):
    """
    Ask Claude to write a teacher-style explanation of why these groups
    are correct. Falls back to the simple version if the API fails.
    """
    answer_key = "\n".join(
        f"  {desc}: {', '.join(members)}"
        for desc, members in zip(descriptions, groups)
    )
    prompt = TEACHER_PROMPT.format(words=shuffled_words_str, answer_key=answer_key)

    try:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception as err:
        print(f"  [!] Claude API error ({err}), using template fallback")
        return make_simple_reasoning(groups, descriptions)


def parse_puzzle(puzzle):
    """
    Pull apart a single puzzle dict into (words, groups, descriptions).
    Returns None if the puzzle is malformed.
    """
    answers = puzzle.get("answers", [])
    if len(answers) != 4:
        return None

    words, groups, descriptions = [], [], []
    for ans in answers:
        members = ans.get("members", [])
        if len(members) != 4:
            return None
        words.extend(members)
        groups.append(members)
        descriptions.append(ans.get("group", "Unknown"))

    if len(words) != 16:
        return None
    return words, groups, descriptions


def load_cache(path):
    """Try loading a JSON cache file. Returns [] on any failure."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
        print(f"  Resumed from cache: {len(data)} entries in {path}")
        return data
    except (json.JSONDecodeError, IOError):
        print(f"  Cache at {path} was corrupted — starting over")
        return []


def save_cache(data, path):
    with open(path, "w") as f:
        json.dump(data, f)


def build_dataset(puzzles, use_claude=False, cache_path="dataset_cache.json"):
    """
    Walk through every puzzle and generate a training example with
    either template CoT or Claude-distilled reasoning.

    Saves to disk every 10 puzzles so you don't lose progress.
    """
    client = anthropic.Anthropic() if use_claude else None
    entries = load_cache(cache_path)
    already_done = len(entries)

    for i, puzzle in enumerate(puzzles):
        if i < already_done:
            continue

        parsed = parse_puzzle(puzzle)
        if parsed is None:
            # malformed puzzle, skip but still count it toward the index
            entries.append(None)  # placeholder to keep indices aligned
            continue

        words, groups, descriptions = parsed
        random.shuffle(words)
        words_str = ", ".join(words)

        # pick the reasoning style
        if use_claude:
            reasoning = make_distilled_reasoning(client, words_str, groups, descriptions)
            # uncomment if you hit rate limits:
            # time.sleep(0.5)
        else:
            reasoning = make_simple_reasoning(groups, descriptions)

        answer_json = json.dumps({"answer": groups})
        prompt = SOLVE_PROMPT.format(input_words=words_str)

        entry = {"text": f"{prompt}\n{reasoning}\nSo the answer is:\n{answer_json} </s>"}
        entries.append(entry)

        if len(entries) % 10 == 0:
            save_cache(entries, cache_path)
            print(f"  ... cached {len(entries)} puzzles")

    save_cache(entries, cache_path)

    # filter out any None placeholders from skipped puzzles
    return [e for e in entries if e is not None]


def make_splits(entries, seed=42):
    """80/10/10 train/val/test split."""
    ds = Dataset.from_list(entries)
    first_split = ds.train_test_split(test_size=0.2, seed=seed)
    second_split = first_split["test"].train_test_split(test_size=0.5, seed=seed)
    return DatasetDict({
        "train": first_split["train"],
        "validation": second_split["train"],
        "test": second_split["test"],
    })


if __name__ == "__main__":
    raw_puzzles = fetch_puzzles()
    print(f"Fetched {len(raw_puzzles)} puzzles total\n")

    # template-based CoT dataset
    print("Building CoT dataset...")
    cot_entries = build_dataset(raw_puzzles, use_claude=False, cache_path="cot_data.json")
    cot = make_splits(cot_entries)

    # Claude-distilled dataset 
    print("\nBuilding distilled dataset via Claude...")
    distill_entries = build_dataset(raw_puzzles, use_claude=True, cache_path="distill_data.json")
    distill = make_splits(distill_entries)

    print(f"\nCoT splits:      {len(cot['train'])} / {len(cot['validation'])} / {len(cot['test'])}")
    print(f"Distilled splits: {len(distill['train'])} / {len(distill['validation'])} / {len(distill['test'])}")