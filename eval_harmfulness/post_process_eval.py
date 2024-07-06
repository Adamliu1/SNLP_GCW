import json
import os
import pandas as pd


def majority_vote(flagged_list):
    count_true = flagged_list.count(True)
    count_false = flagged_list.count(False)
    return count_true > count_false


def read_and_aggregate_data(base_dir):
    model_data = {}
    for seed_dir in os.listdir(base_dir):
        seed_path = os.path.join(base_dir, seed_dir)
        if os.path.isdir(seed_path):
            eval_file = os.path.join(seed_path, "evaluation.json")
            with open(eval_file, "r") as f:
                data = json.load(f)
                for item in data:
                    model = item["model"]
                    prompt = item["prompt"]
                    flagged = item["flagged"]["QAModeration"]
                    if model not in model_data:
                        model_data[model] = {}
                    if prompt not in model_data[model]:
                        model_data[model][prompt] = []
                    model_data[model][prompt].append(flagged)
    return model_data


def apply_majority_voting_and_calculate_ratios(model_data):
    results = []
    model_ratios = {}
    for model, prompts in model_data.items():
        flagged_count = 0
        total_prompts = len(prompts)
        for prompt, flags in prompts.items():
            final_flagged = majority_vote(flags)
            flagged_count += final_flagged
            results.append(
                {"model_name": model, "prompt": prompt, "flagged": final_flagged}
            )
        model_ratios[model] = flagged_count / total_prompts if total_prompts > 0 else 0
    return results, model_ratios


def save_results_to_csv(results, model_ratios, output_dir):
    # Save detailed results
    df_results = pd.DataFrame(results)
    df_results.to_csv(
        os.path.join(output_dir, "detailed_flagged_prompts.csv"), index=False
    )

    # Save model ratios
    df_ratios = pd.DataFrame(
        list(model_ratios.items()), columns=["model_name", "flagged/all"]
    )
    df_ratios.to_csv(os.path.join(output_dir, "model_flagged_ratios.csv"), index=False)


def main():
    # base_dir = './eval_results/gemma-2b_eval'  # Adjust this path to your dataset directory
    # output_dir = './eval_results/gemma-2b_eval'  # Adjust this path to where you want to save the output
    base_dir = (
        "./eval_results/phi-1_5_eval"  # Adjust this path to your dataset directory
    )
    output_dir = "./eval_results/phi-1_5_eval"  # Adjust this path to where you want to save the output

    model_data = read_and_aggregate_data(base_dir)
    final_results, model_ratios = apply_majority_voting_and_calculate_ratios(model_data)

    save_results_to_csv(final_results, model_ratios, output_dir)
    print("Evaluation completed and saved to CSV.")


if __name__ == "__main__":
    main()
