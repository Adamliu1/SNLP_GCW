# TODOs

- [ ] Eval harmfulness
  - [x] @WillmishScript takes a name of the model directory (evaluates all checkpoints at the same time), takes PKU testset/beaver testset and generates question-answer pairs in jason format (THE jason format.)
  - [x] @TheRootOf3 Another scipt takes beaver dam model and evaluates the generated jason file.
  - [x] Joined both @Willmish's and @TheRootOf3's work!
  - [ ] @Willmish Add bash script to automate this evaluation
  - [ ] Potentially, better structure this directory - in a similar way to `eval_framework_tasks`, where there is a dedicated directory per experiments. 
  - [ ] Eval done.