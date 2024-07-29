# How to Start

``Caution: This file is temporary and will be used as a reference to our final readme``

## On our Strangepork nodes

### Create venv
Source the python environment with bzip (this includes python developer headers, so there wont be issues with missing Python.h files)
```
source /share/apps/source_files/python/python-3.11.5_bzip2.source
```

Then create a python venv using 

```python -m venv /path/to/new/virtual/environment```


## Soucre CUDA then install dependencies

### Source CUDA
Before installing any dependencies, we need to source CUDA to the environment. Otherwise, it might causes failure on compiling some required libraries.

On Stangepork, I recommend using CUDA11.8, the path is as follow:
```
source /share/apps/source_files/cuda/cuda-11.8.source
```

### Install dependencies

install all requirements (accelerate, deepspeed, torch (should be 2.3.1>), transformers, and might be a few others depending on used model)

As a reference, this is a copy of Adam's venv
```
accelerate==0.31.0
aiohttp==3.8.6
aiosignal==1.3.1
annotated-types==0.7.0
async-timeout==4.0.3
attrs==23.1.0
bitsandbytes==0.43.1
certifi==2023.7.22
charset-normalizer==3.3.0
click==8.1.7
contourpy==1.2.1
cycler==0.12.1
datasets==2.14.5
deepspeed==0.14.2
dill==0.3.7
docker-pycreds==0.4.0
filelock==3.12.4
fonttools==4.53.0
frozenlist==1.4.0
fsspec==2023.6.0
gitdb==4.0.11
GitPython==3.1.43
hjson==3.1.0
huggingface-hub==0.23.3
idna==3.4
Jinja2==3.1.3
joblib==1.4.2
kiwisolver==1.4.5
MarkupSafe==2.1.5
matplotlib==3.9.0
mpmath==1.3.0
multidict==6.0.4
multiprocess==0.70.15
networkx==3.2.1
ninja==1.11.1.1
numpy==1.26.1
nvidia-cublas-cu11==11.11.3.6
nvidia-cuda-cupti-cu11==11.8.87
nvidia-cuda-nvrtc-cu11==11.8.89
nvidia-cuda-runtime-cu11==11.8.89
nvidia-cudnn-cu11==8.7.0.84
nvidia-cufft-cu11==10.9.0.58
nvidia-curand-cu11==10.3.0.86
nvidia-cusolver-cu11==11.4.1.48
nvidia-cusparse-cu11==11.7.5.86
nvidia-nccl-cu11==2.20.5
nvidia-nvtx-cu11==11.8.86
packaging==23.2
pandas==2.1.1
peft==0.5.0
pillow==10.2.0
platformdirs==4.2.2
protobuf==5.27.1
psutil==5.9.6
py-cpuinfo==9.0.0
pyarrow==16.1.0
pydantic==2.7.3
pydantic_core==2.18.4
pynvml==11.5.0
pyparsing==3.1.2
python-dateutil==2.8.2
pytz==2023.3.post1
PyYAML==6.0.1
regex==2023.10.3
requests==2.31.0
safetensors==0.4.3
scikit-learn==1.5.0
scipy==1.13.1
sentencepiece==0.2.0
sentry-sdk==2.5.1
setproctitle==1.3.3
six==1.16.0
smmap==5.0.1
svgwrite==1.4.3
sympy==1.12
threadpoolctl==3.5.0
tokenizers==0.19.1
torch==2.3.1+cu118
torchaudio==2.3.1+cu118
torchvision==0.18.1+cu118
tqdm==4.66.1
transformers==4.41.2
triton==2.3.1
typing_extensions==4.8.0
tzdata==2023.3
urllib3==2.0.7
wandb==0.17.1
xxhash==3.4.1
yarl==1.9.2
```

## Configure Accelerate and Deepspeed

``NOTE: Currently, using accelerate config to configure the environment. However, we will shift to include a manual config in later commits in order to have more control on accelerate.``

### Configure as follow
Use ``accelerate config``

Select compute environment
```
In which compute environment are you running?
Please select a choice using the arrow or number keys, and selecting with enter
 ➔  This machine
    AWS (Amazon SageMaker)
```

Select distributed training or not
```
Which type of machine are you using?                                                                                                                                                                   
Please select a choice using the arrow or number keys, and selecting with enter
    No distributed training                                                                                 
    multi-CPU                                                                                          
    multi-XPU                                                                                        
 ➔  multi-GPU
    multi-NPU
    multi-MLU
    TPU
```

Use all default values (Just press ``Enter``)
```
How many different machines will you use (use more than 1 for multi-node training)? [1]: 

```

```
Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]:  
```

```
Do you wish to optimize your script with torch dynamo?[yes/NO]:
```

Use DeepSpeed
```
Do you want to use DeepSpeed? [yes/NO]: yes
```

Currently we are not specifying a json file to a DeepSpeed config
```
Do you want to specify a json file to a DeepSpeed config? [yes/NO]: NO
```

Use ``ZeRO 2`` Offload
```
What should be your DeepSpeed's ZeRO optimization stage?                                                                                                                                               
Please select a choice using the arrow or number keys, and selecting with enter
    0
    1
 ➔  2
    3
```

Offloading to ``CPU``
```
Where to offload optimizer states?                                                                                                                                                                     
Please select a choice using the arrow or number keys, and selecting with enter                                                                                                                        
    none                                                                                                                                                                                               
 ➔  cpu
    nvme
```
```
Where to offload parameters?                                                                                                                                                                           
Please select a choice using the arrow or number keys, and selecting with enter                                                                                                                        
    none                                                                                                                                                                                               
 ➔  cpu
    nvme
```

Currently we don't want to touch deepspeed gradient accumulation yet. This could also be defined later in python manually after we fix the gradient accumulation in our code base.
```
How many gradient accumulation steps you're passing in your script? [1]:     
```

Select graident clipping
```
Do you want to use gradient clipping? [yes/NO]:  
```
No ``deepspeed.zero.Init`` as we are only using ``ZeRO 2``. 
```
Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]
```
No ``MoE``
```
Do you want to enable Mixture-of-Experts training (MoE)? [yes/NO]: 
```
Select Number of GPU(s) to be used for distributed training. This could vary depending on our needs.
```
How many GPU(s) should be used for distributed training? [1]:
```
Choose ``bf16`` as ``A100`` supports.
```
Do you wish to use FP16 or BF16 (mixed precision)?
Please select a choice using the arrow or number keys, and selecting with enter
    no                                                                                                                                                                                                 
    fp16                                                                                                                                                                                               
 ➔  bf16
    fp8
```
Last, the configuration program will output the location of the saved accelerate config file. For example
```
accelerate configuration saved at /home/yadonliu/.cache/huggingface/accelerate/default_config.yaml  
```

## Example command to launch unlearning

Launch llm unlearning as before, but now source the venv with deepspeed, and use ``accelerate launch`` instead of ``python3`` as the launcher.

Example command
```
CUDA_VISIBLE_DEVICES=0 accelerate launch unlearn_harm_redo_accelerate.py --model_name meta-llama/Meta-Llama-3-8B --model_save_dir "/SAN/intelsys/llm/yadonliu/SNLP_GCW/snlp-unlearned-models/models/test_llama8b" --log_file "/SAN/intelsys/llm/yadonliu/SNLP_GCW/snlp-unlearned-models/logs/test_llama8b.log" --cache_dir "/home/yadonliu/huggingface_cache" --seed 42 --retaining_dataset rajpurkar/squad --max_bad_loss 10000 --sequential=-1 --num_epochs=1 --batch_size=1 --seed=42 --save_every=100 --lr 2e-6
```

Where you can modify GPUs that are visible to the accelerate. For example ``CUDA_VISIBLE_DEVICES=0,1`` makes ``GPU0`` and ``GPU1`` visiable to the python run.

## Troubleshooting

### If multiple deepspeed sessions running on the same node

There might be potenitally more issues if multiple deepspeed sessions running on the same node (may need to tweak ``--main_process_port`` https://huggingface.co/docs/accelerate/en/package_reference/cli ) and potenitally CUDA_VISIBLE_DEVICES environment var

### If encountering an error: while Building extension module cpu_adam...

```
FAILED: cpu_adam.so
c++ cpu_adam.o cpu_adam_impl.o -shared -lcurand -L/home/sduchnie/strangepork_venv/lib/python3.11/site-packages/torch/lib -lc10 -ltorch_cpu -ltorch -ltorch_python -o cpu_adam.so
/usr/bin/ld: cannot find -lcurand
collect2: error: ld returned 1 exit status
```

It means the lib curand for CUDA is not visible by the linker (see [microsoft/DeepSpeed#3929](https://github.com/microsoft/DeepSpeed/issues/3929) ) and needs to be linked in manually. you can do so by:

1. Cd into: ``cd venv/lib/python3.11/site-packages/torch/lib`` (or whatever your lib is called)
2. create a symbolic link for the missing lib (should be located in: ``/usr/local/cuda/lib64/libcurand.so``): ``ln -s /usr/local/cuda/lib64/libcurand.so .``
3. Retry running the command







