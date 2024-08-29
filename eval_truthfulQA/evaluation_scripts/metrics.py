import re

import numpy as np


def calculate_question_repeated_ratio(data: dict) -> dict:
    repeated_question_count = 0
    for line in data:
        if line["question"].lower() in line["answer"].lower():
            repeated_question_count += 1

    return {"repeated_question_ratio": repeated_question_count / len(data)}


def calculate_output_quality_heuristics(data: dict) -> dict:
    special_char_count = np.array(
        [
            (
                0
                if len(line["answer"]) == 0
                else len(re.findall("[\W]", line["answer"])) / len(line["answer"])
            )
            for line in data
        ],
        dtype=float,
    )

    empty_count = np.array(
        [(1 if len(line["answer"]) == 0 else 0) for line in data],
        dtype=float,
    )

    return {
        "special_char_count/characters_in_response": special_char_count.mean(),
        "empty_response_ratio": empty_count.mean(),
    }


def calculate_response_length(data: dict) -> dict:
    response_len = np.array(
        [len(line["answer"]) for line in data],
        dtype=float,
    )

    return {
        "avg_response_length": response_len.mean(),
    }
