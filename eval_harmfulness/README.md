# Evaluation

First you need to generate model outputs on beavtertails PKU dataset. Then evaluate using a moderation model.

## Generate model answers

```bash
python3 beavertails_get_model_answers.py \
    --model_path ../snlp-unlearned-models/models/opt1.3b_unlearned_harmful-for-real42/idx_0 \
    --device "cuda" 
```

## Evaluate generated model answers

Eval the training results:

```bash
python evaluate_outputs.py \
    --eval_dataset model_generations/example_generations.json \
    --model_path ./beaver-dam-7_weights \
    --max_length 512 \
    --output_dir eval_results/evaluation \
    --device "cuda"
```

Model weights on Hugging Face: [`PKU-Alignment/beaver-dam-7b`](https://huggingface.co/PKU-Alignment/beaver-dam-7b).
