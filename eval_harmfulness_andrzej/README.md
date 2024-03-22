# Evaluation

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
