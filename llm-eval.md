# Evaluation

> [!NOTE]  
> @TheRootOf3 wrote the following because of an urgent need to write some text in a paper-like style at 2 am. May be complete trash. Reading at your own risk. _The following text is provided "as is", without warranty of any kind..._

Reliably removing undesired content from the outputs of a Language Model is one of the key challenges in the area of Large Language Models. Unlearning high-level tasks is useful in tackling varying forms of bias coming from pre-training datasets and allows a relatively inexpensive method for model alignment. Furthermore, the significant scale of pre-training datasets, makes it difficult to remove _all_ unwanted personally identifiable information (PII) at the stage of data pre-processing. The unlearning paradigm offers an alternative to applying additional output filters and is much cheaper in comparison to re-training the entire model.

While there have been multiple methods proposed, we show that while they are capable of successfully unlearning particular concepts, some of these methods also significantly decrease the overall performance of a model. In our work, we discuss the importance of a robust unlearning benchmark, consider both task-specific and overall model performance.

We aim to evaluate Language Model unlearning with respect to the performance of an unlearned model and the efficiency of the unlearning process itself.

## The performance of an unlearned model

@TheRootOf3

The idea to evaluate how well the model unlearned task A:

- Have a benchmark for task A e.g. sentiment analysis.
- Have a set of benchmarks measuring the overall performance of a model.
- Compare the performance of the unlearned model with the original model on all tasks. Hope for the drop in performance on task A and similar performance on the rest of tasks.

We can use the existing [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) from huggingface. It has the following tasks:

1. AI2 Reasoning Challenge (25-shot) - a set of grade-school science questions.
2. HellaSwag (10-shot) - a test of commonsense inference, which is easy for humans (~95%) but challenging for SOTA models.
3. MMLU (5-shot) - a test to measure a text model's multitask accuracy. The test covers 57 tasks including elementary mathematics, US history, computer science, law, and more.
4. TruthfulQA (0-shot) - a test to measure a model's propensity to reproduce falsehoods commonly found online. Note: TruthfulQA is technically a 6-shot task in the Harness because each example is prepended with 6 Q/A pairs, even in the 0-shot setting.
5. Winogrande (5-shot) - an adversarial and difficult Winograd benchmark at scale, for commonsense reasoning.
6. GSM8k (5-shot) - diverse grade school math word problems to measure a model's ability to solve multi-step mathematical reasoning problems.

We can either submit our models to the dashboard (nay :/) and get them evaluated there or run the evaluation harness locally (yay!).

### Running the llm-eval-harness

https://github.com/EleutherAI/lm-evaluation-harness provides all we need to evaluate our models -> we can evaluate directly using huggingface transformers library!

To run:

1. Follow the installation steps on their github.
2. Have a directory with a model **AND a tokenizer**.
3. Run the following:

```bash
lm_eval --model hf \
    --model_args pretrained=/path/to/model \
    --tasks arc_challenge,hellaswag,truthfulqa,mmlu,winogrande,gsm8k \
    --device cpu \
    --batch_size auto \
    --trust_remote_code True \
    --output_path eval_results \
    --use_cache "./.cache"
```

A couple of notes:

- choose the right device (look docs) - there is a problem with `mps` on Apple Silicon - I didn't manage to get it running because of dtypes of weights :/
- batch size can be auto calculated (max possible) or set by a user.
- For testing you can use a flag `--limit`, which allows to specify what fraction (float 0-1) of benchmarks you want to use. In general this is only useful for testing because evaluation must be done on full benchmark datasets.

I added `eval_framework_tasks/evaluate_models.sh`, which gives a easy way to evaluate a given model. The script allows to evaluate a single model and takes two parameters:
1. The name of the experiment (This is when you want to evaluate multiple models within an experiment)
2. The name of the model from `/cs/student/projects1/2020/aszablew/SNLP/SNLP_GCW/llm_unlearn_ucl/snlp-unlearned-models/models/` you want to evaluate.

Example: 
```bash
./evaluate_models.sh experiment1 opt1.3b_unlearned
```

## The performance of the process of unlearning

@Davidyz

## Using Min-k% prob as an unlearning metric - part of the unlearning loss?

@Willmish
