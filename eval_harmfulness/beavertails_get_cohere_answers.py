# Copyright (C) 2024 UCL CS SNLP Naturalnego 语言 Töötlus group
#    - Szymon Duchniewicz
#    - Yadong Liu
#    - Andrzej Szablewski
#    - Zhe Yu
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import json
import os

from datasets import load_dataset
import cohere
from tqdm import tqdm

from cohere_api_key import COHERE_API_PRODUCTION_KEY


def main() -> None:
    co = cohere.Client(COHERE_API_PRODUCTION_KEY)
    output_dir = "./model_generations"
    model_name = "cohere_command_r"

    os.makedirs(output_dir, exist_ok=True)

    # load dataset
    dataset = load_dataset("PKU-Alignment/BeaverTails-Evaluation")["test"]

    evaluations = []
    for element in tqdm(dataset):
        try:
            response = co.chat(
                # model="command-r-plus",
                message=element["prompt"],
            )
        except cohere.error.CohereAPIError as e:
            print(f"CohereAPIError: {e}")
            print(f"Refused dataset sample: {element}")

            # if API refuses to respond, replace with an empty response
            class Response:
                text = ""

            response = Response()

        evaluations.append(
            {
                "prompt": element["prompt"],
                "response": response.text,
                "model": model_name,
                "category_id": element["category_id"],
            }
        )

        # Save evaluations to JSON file
        with open(f"{output_dir}/evaluation_{model_name}.json", "w+") as outfile:
            json.dump(evaluations, outfile, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
