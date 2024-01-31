# SNLP_GCW

## Note on 31.Jan.2024

In this meeting, we have found few papers to look at next:
- [Who’s Harry Potter? Approximate Unlearning in LLMs](https://arxiv.org/pdf/2310.02238.pdf)
- [Large Language Model Unlearning(Arxiv 2310.10683)](https://arxiv.org/pdf/2310.10683.pdf)
- [DETECTING PRETRAINING DATA FROM LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2310.16789.pdf)

The keyword for our project now shrink down to ``Unlearning (forgetting)`` and ``Dectecting pretraing data``. We only focus on the after-pretrain stage of the model

### Idea
Therefore, the idea is to add some secrets to a language model. apply some forgetting method. see if the results probability distribution is the same ish.

However, we don't know what to unlearn yet. Therefore, the goal before next Monday is to try making use of the basecode from exisiting related work. To verify and dive a bit further into how these work actually established "unlearning" and how potentially you can extract pretrain data from an LLM.

By looking at code for above papers, hopefully we will understand better on the methodology and hence start creating our own "skeleton" and pipeline of the code.

Currenlty we just use exisitng dataset used by these two codebases and if our concpet is likely to work, then we can extend our code based on them and starting using our own dataset inorder to achieve "forgetting".


### TODO 
- [x] Init the git repo
- [x] Added two code bases to look at before next Monday
    - [x] [Arxiv 2310.10683](https://github.com/kevinyaobytedance/llm_unlearn/tree/main)
    - [x] [Arxiv 2310.16789](https://swj0419.github.io/detect-pretrain.github.io/)