# Data-Efficient Task Unlearning in Language Models
> Szymon Duchniewicz, Andrzej Szablewski, Zhe Yu, Yadong Liu, Carmen Meinson

WIP!!!!

## Environment Setup
``NOTE`` I think here I will combine everything into one setup script, but doing manual instruction for each component for now

``NOTE`` Use ``python3.10``, newer version will have issue when seting up the env (if using current requirements.txt).

Install dependencies for ``llm_unlearn``
```
cd llm_unlearn_ucl
pip install -r requirements.txt
```
Install dependencies for ``lm-evaluation-harness``
```
cd lm-evaluation-harness
pip install -e .
```

``NOTE`` for eval harmfulness
requires ``sentencepiece``


## Machine Unlearning

### Continuous Unlearning
```
    python unlearn_harm.py \
        --wandb_project_name "$wandb_project_name" \
        --cache_dir "$base_cache_dir" \
        --wandb_run_name "$wandb_run_name" \
        --samples_save_dir "$samples_dir" \
        --model_save_dir "$model_save_dir" \
        --save_every "$save_every" \
        --seed "$seed" \
        --log_file "$log_file" \
        --wandb_log
```

### Batch Unlearning
```

```

### Evaluation


## Note on configuration used for experiments
- By default, the code uses lr_scheduler, add ``--no_scheduler`` to disable it.
- 

## Arguments
```
loss = (
    args.bad_weight * bad_loss
    + args.random_weight * random_loss
    + args.normal_weight * normal_loss
)
```



## Data
Unlearned models available on HuggingFace: https://huggingface.co/ucl-snlp-nyt/snlp-unlearned-models

Data analysis and visualisation of results available at: https://github.com/Willmish/SNLP_GCW_data_analysis/
