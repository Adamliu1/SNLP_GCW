# How to upload/download models from HF?

We have... ✨ [an organization!](https://huggingface.co/ucl-snlp-nyt) ✨

## Uploading models:

Using `huggingface-cli` makes things very easy, so can recommend. To upload a particular model and all the files belonging to it, you can do:

```bash
huggingface-cli upload ucl-snlp-nyt/snlp-unlearned-models models/model_name models/model_name
```

To download all the models, you can use git lfs! First you can skip the large files:

```bash
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/ucl-snlp-nyt/snlp-unlearned-models
```

or just download them in one go:

```bash
git clone https://huggingface.co/ucl-snlp-nyt/snlp-unlearned-models
```
