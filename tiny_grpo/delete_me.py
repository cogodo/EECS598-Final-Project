from __future__ import annotations
from typing import Any, Iterator, Optional
from pathlib import Path
from collections.abc import Callable
import json
from collections.abc import Callable


def read_jsonl(file_name: str | Path) -> Iterator:
    file_path = Path(file_name)
    with file_path.open(mode="r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def read_prompts(
    file_name: str,
    predicate: Optional[Callable[[Any], bool]] = None,
    max_rows: Optional[int] = None,
) -> list:
    rows = []
    for x in read_jsonl(file_name):
        if predicate is None or predicate(x):
            rows.append(x)
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


prompts = read_prompts(
    "data/dummy_math_tasks.jsonl",
    predicate=lambda x: len(x["question"]) < 256,
    max_rows=64,
)

print("prompts[0]")
print(prompts[0])

# print("type(prompts)")
# print(type(prompts))

# print("type(prompts[0])")
# print(type(prompts[0]))

# print("type(prompts[0]['question'])")
# print(type(prompts[0]['question']))


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():        # skip empty lines
                data.append(json.loads(line))
    return data

# # Example usage:
items = load_jsonl("data/train.jsonl")
print(items[0])




