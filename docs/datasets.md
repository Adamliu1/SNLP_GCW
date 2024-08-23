# Datasets

This document describes the datasets and their usage in the LLM Unlearning project.

## Evaluation:

| Dataset name                                                                                                                                          | Evaluation method  | Description                                          | Random guess accuracy                                                                                                                                                    | Evaluation metric    |
| ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------- |
| [MMLU](https://huggingface.co/datasets/hails/mmlu_no_train)                                                                                           | Highest likelihood | Multi-choice multi-task understanding                | 0.25                                                                                                                                                                     | acc                  |
| [ARC-Challenge](https://huggingface.co/datasets/allenai/ai2_arc)                                                                                      | Highest likelihood | Multi-choice science question answering              | around 0.25 [explained here](https://paperswithcode.com/dataset/arc)                                                                                                     | acc                  |
| [HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag)                                                                                          | Highest likelihood | Natural language inference. Sentence continuation.   | 0.25                                                                                                                                                                     | acc                  |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k)                                                                                                 | Generate text      | Maths                                                | n/a                                                                                                                                                                      | exact match flexible |
| [Winogrande](https://huggingface.co/datasets/allenai/winogrande)                                                                                      | Highest likelihood | Commonsense reasoning, fill in a blank.              | 0.5                                                                                                                                                                      | acc                  |
| [TruthfulQA MC2](https://huggingface.co/datasets/truthfulqa/truthful_qa)                                                                              | Highest likelihood | Question answering                                   | 0.25                                                                                                                                                                     | acc                  |
| [PIQA](https://huggingface.co/datasets/ybisk/piqa)                                                                                                    | Highest likelihood | Question answering (physical interactions)           | 0.5                                                                                                                                                                      | acc                  |
| [FrenchBench](https://github.com/EleutherAI/lm-evaluation-harness/blob/5ad23ec22ca63193ef6ace0fded1142c6c34eb21/lm_eval/tasks/french_bench/README.md) | n/a                | French language (composition of multiple benchmarks) | n/a                                                                                                                                                                      | acc                  |
| [PKU-Alignment/BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails)                                                                | Generate text      | Toxic questions provoking unsafe responses           | n/a                                                                                                                                                                      | ratio flagged        |
| [toxigen](https://huggingface.co/datasets/toxigen/toxigen-data)                                                                                       | Highest likelihood | Toxic/harmful content detection                      | around 0.5 [but not sure?](https://github.com/EleutherAI/lm-evaluation-harness/blob/5ad23ec22ca63193ef6ace0fded1142c6c34eb21/lm_eval/tasks/toxigen/README.md?plain=1#L2) | acc                  |
| [mnli](https://huggingface.co/datasets/nyu-mll/multi_nli)                                                                                             | Highest likelihood | Multi-genre natural language inference               | 0.33                                                                                                                                                                     | acc                  |

## Model Unlearning

All samples in unlearning and retaining datasets are formatted in a similar way as questions-answering pairs.

### Unlearning Sets

- French: [AgentPublic/piaf](https://huggingface.co/datasets/AgentPublic/piaf)
- Harmfulness: [PKU-Alignment/PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF)
- Logical reasoning: [sail/symbolic-instruction-tuning](https://huggingface.co/datasets/sail/symbolic-instruction-tuning)

### Retaining Set

- [squad](https://huggingface.co/datasets/rajpurkar/squad) (with eval on [squadv2](https://huggingface.co/datasets/rajpurkar/squad_v2) because of availability in lm-eval-harness)

## Notes

- GSM8K: Evaluation takes very long time due to its "generative" nature and large dataset size -- potentially to be dropped.
- Retain set: Changed to squad, while evaluation is performed on squadv2 (which is essentially squad + some "tricky" question without answers).
- Logical reasoning: Replaced [LogiQA](https://paperswithcode.com/dataset/logiqa) with [mnli](https://huggingface.co/datasets/nyu-mll/multi_nli) because of random guess performance on the former.
- Toxigen is a task verifying whether a model can DETECT a toxic/harmful content, which is different from quantifying the toxicitiy of its pure generations. However, since we are measuring the toxicity of model generations using beavertails, we may want to capture the potential ability of toxicity detection, which may arise in larger models (~8B).
- realtoxicityprompts is based on generate_until requests, which take significantly more time to compute and this benchmark additionally requires an external API for scoring toxicity of generations. **Hence, we did not include it.**
