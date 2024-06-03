# Notes on the UCL CS Cluster (aka Sun Grid Engine survival guide)

🚧 Work in Progress... Last Updated: 2024-06-03.

[UCL CS cluster](https://hpc.cs.ucl.ac.uk) uses a Sun Grid Engine (SGE) job scheduler. There is a decent introduction to SGE available on the [tsg website](https://hpc.cs.ucl.ac.uk/wp-content/uploads/sites/21/2022/01/SGE-Guide_12_2021.pdf) as well as additional information about [job submission](https://hpc.cs.ucl.ac.uk/job-submission/) but here is a couple of my thoughts and tricks, that you may find useful.

## Clustering...

![](https://miro.medium.com/max/561/0*ff7kw5DRQbs_uixR.jpg)

...It's not even it.

## TL;DR

Skip to _[Example submission script](#example-submission-script-for-dip-207-2)_.

## Basic and very quick facts

Cluster has 2 main types of nodes:

- _login nodes_, from which you can submit jobs,
- and _compute nodes_, which do all the computation.

_In general_, most of the computation should be performed on the compute nodes, either through job submissions or interactive sessions (if needed).

In many HPC clusters compute nodes usually do not have access to Internet, but this is not the case with UCL CS! Hence, it is possible to schedule a job, which requires some pre-fetching of data from Internet and it will be successfully executed (yay!). Nevertheless, performing regular network operations is a bad practice, given it can cause significant i/o slowdowns.

**BEWARE:** `wandb` and similar logging services should run in the _offline_ mode only!

## Basic scheduler stuff

Here are the basic SGE commands you'll use:

- `qstat` -- displays the queue of your jobs, most importantly `qw` means the job is queued and waits to be schedule, `r` means it is running.
- `qsub` -- used to submit a job. Most commonly used with an argument, which is a shell script containing scheduler directives.
- `qdel` -- deletes queued or running jobs with given job IDs.
- `qrsh` -- requests an interactive session with a node. You must provide at least the requested time and memory `qrsh -l tmem=2G,h_rt=00:05:00`.

_**PROTIP:**_ There is a bunch of "test nodes" available at any time, which are automatically allocated if you make a small request (e.g. <16GB RAM total, <1h). This may be useful when you want to run a relatively small job immediately.

## Job submission hello world

Let's say you want to run a Python script, which requires a particular environment. However, because the job is going to be submitted to the cluster and may run on an arbitrary machine, a common practice involves wrapping the Python script in a shell script containing necessary setup steps.

An example submission script `example.qsub.sh` looks as follows:

```bash
# Required scheduler flags
#$ -l tmem=2G # Total requested memory.
#$ -l h_rt=00:01:00  # either seconds or hh:mm:ss

# Optional flags

#$ -j y # Combines stdout and stderr into a single file.
#$ -N MyTESTJOBNAME # Human-readable name of the job.


# Setup Python environment -- source the correct version
source /share/apps/source_files/python/python-3.9.5.source

hostname
date

python3 ./hello_world.py

date
```

Note the difference between a simple bash comment character `#` and a scheduler flag prefix `#$`. Furthermore, the scheduler flag are interpreted as comments by bash and are only meaningful to the scheduler.

_**PROTIP:**_ It is a good practice to wrap the actual python command with date commands, which provide a simple way to calculate the runtime.

## CPU parallelism -- cores here and there!

If you want to parallelise your code, make sure it supports parallelism (LOL).

There are 2 supported multi-core frameworks -- [SMP](https://en.wikipedia.org/wiki/Symmetric_multiprocessing) (within a single node) and [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) (across nodes). The only required change you need to perform is to add a corresponding scheduler flag. For 24 SMP cores:

```bash
#$ -l tmem=1G
#$ -l h_rt=01:00:00 # hh:mm:ss
#$ -pe smp 24
#$ -R y # Resource reservation
```

For 250 MPI cores:

```bash
#$ -l tmem=1G
#$ -l h_rt=01:00:00 # hh:mm:ss
#$ -pe mpi 250
#$ -R y # Resource reservation
```

**BEWARE**: The amount of requested memory is multiplied by the number of requested CPU/GPU cores! So if you want to request 48GB of RAM with 8 smp cores, the scheduler directive should be as follows:

```bash
#$ -l tmem=6G
#$ -l h_rt=01:00:00 # hh:mm:ss
#$ -pe smp 8
#$ -R y # Resource reservation
```

## Zooooming with GPUs!

Finally...

If you're wondering how to request access to nodes with one or more GPUs, you guessed it - it's just another scheduler flag:

```bash
#$ -l gpu=true # Use gpu nodes
#$ -pe gpu 4 # Request 4 cards at once
```

In addition to determining a number of GPUs, you can also optionally specify a particular type:

```bash
#$ -l gpu_type=rtx4090 # Use nodes with rtx4090 GPUs only
```

**BEWARE**: Similarly to multi-core allocations, the amount of requested memory is multiplied by the number of requested GPUs!

As far as I know (as of 2024-06-03), there is no support for MPI-like multi-node GPU jobs. Hence, you can request up to 4 or 8 cards at once, although it will probably take a long wait time to get you scheduled.

## I only want this one!

By default the scheduler will handle compute node allocation. However, you can specify a particular node (e.g. `dip-207-2`), on which you want to execute your job by adding the following flag:

```bash
#$ -l hostname=dip-207-2
```

This is particularly useful if you have exclusive access (reservation) to a node.

## How NOT to slow down your code

### Use scratch, please. O̶t̶h̶e̶r̶w̶i̶s̶e̶...

Cluster uses NFS. If you need to read and write a lot from disk it may significantly slow down your computation (and in fact slow down the entire cluster LOL). To avoid this, each node has a certain amount of fast "scratch" storage available for frequent I/O operations.

Unless you have an exclusive access to a compute node (which generously grants you access to the entire scratch space), you can request 200G of scratch storage by adding the following scheduler flag to your submission script.

```
#$ -l tscratch=200G
```

Scratch is mounted at `/scratch0/`.

### Do not send many network requests, really.

## Banned words

- `tmux`
- `screen`
- `nohup`
- ...essentially anything that may try to block the cluster scheduler from killing a job.

_**PROTIP:**_ Make sure that if your job submission script creates background processes, you always await them at the end. Otherwise, the scheduler will deallocate resources and kill remaining processes once the end of the job script is reached!

## Example submission script for dip-207-2

`dip-207-2` is a powerful machine with the following specs:
8x RTX A6000 ~50GB VRAM per card, 503G total RAM, ~ 1.6TB scratch space.

We can fully utilise it and as long as the requested resources can be satisfied by it, submitted jobs should be scheduled immediately.

The following is an example evaluation script.

```bash
#$ -l tmem=40G
#$ -l h_rt=12:00:00 # hh:mm:ss
#$ -l gpu=true
#$ -pe gpu 1
#$ -j y
#$ -N OLMo7B-lm-eval-harness
#$ -l hostname=dip-207-2

source ~/setup_python.sh
cd /SAN/intelsys/llm/aszablew/snlp/SNLP_GCW/eval_framework_tasks

date
hostname

source ./run_experiment_raw_model.sh

date
```

Furthermore, an interactive session may be useful for testing multi-gpu code:

```bash
qrsh -l tmem=24G,h_rt=04:00:00,gpu=true,hostname=dip-207-2 -pe gpu 8
```

**NOTE:** If you submit 8 jobs with a single gpu request (`-pe gpu 1`) to `dip-207-2`, they will run in parallel!
