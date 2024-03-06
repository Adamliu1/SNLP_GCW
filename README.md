# SNLP_GCW

## Note on 5.Feb.2024

Expanding on the idea of secret unlearning. Paper achieving this to some extent: https://arxiv.org/abs/2310.10683

Potential approaches:
Reproduction study (have the authors cherry picked results? Eval metrics?)
Loss function in the paper uses only 3 metrics, can we come up with another one?
Instead of unlearning longer chunks of data, unlearn API keys
Reproduce on a larger model/different model

Detecting pre-trained data - we can use that method for evaluating whether something was successfully unlearned (if no longer detected in pre-training data?) https://arxiv.org/abs/2310.16789

Example approach:
Reproduce unlearning based on the method proposed in the paper https://arxiv.org/abs/2310.10683
Re-evaluate using the same methods - reflect on results and the validity of results
Introduce a new metric for unlearning (LOSS FUNCTION), currently used: Unlearn Harm, Random Mismatch, Maintain Performance . Come up with a new metric and see if it improves
Try a different evaluation metric to check if the model got worse (in paper used similarity)
Try to adapt and re-evaluate using pre-training data detection

## Note on 31.Jan.2024

In this meeting, we have found few papers to look at next:

- [Who’s Harry Potter? Approximate Unlearning in LLMs](https://arxiv.org/pdf/2310.02238.pdf)
- [Large Language Model Unlearning(Arxiv 2310.10683)](https://arxiv.org/pdf/2310.10683.pdf)
- [DETECTING PRETRAINING DATA FROM LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2310.16789.pdf)

The keyword for our project now shrink down to `Unlearning (forgetting)` and `Dectecting pretraing data`. We only focus on the after-pretrain stage of the model

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
