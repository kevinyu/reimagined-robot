## Setup

### 1. Clone the repository

### 2. Set up a virtualenv or conda environment

Create a new virtual environment. Easiest is to use anaconda and `conda create --name <env-name`

### 3. Install dependencies

Activate the virtual environment using `source activate <env-name`.

Requires at least these... there might be more

```
numpy
scipy
scikit-learn
skikit-cuda
matplotlib
theano
```

Also must install GPU stuffs (CUDA). Requires CuDNN also or error stuff will happen.

### 4. Set up PYTHONPATH and PROJECT_ROOT environment variables

If using a conda environemnt, just activate the environment and run `./setup.sh` from the root project directory. The envirnoment's activation script (run when you do `source activate <env-name` will always set the `PYTHONPATH` and `PROJECT_ROOT` paths correctly, and `source deactivate` will unset them.

If not using a conda environment, just make sure the two environment variables are set correctly before running anything.

```
export PROJECT_ROOT="<path-to-repository-root>"
export PYTHONPATH="${PROJECT_ROOT}/src"
```

## Running Stuff

* Create a new output dir called <name> in the outputs/ directory, `mkdir outputs/my_new_run/`

* Create a outputs dir in there `mkdir outputs/my_new_run/outputs`

* Copy the file `src/opt_default.py` to `outputs/<name>/opt.py`

* Edit the `opt.py` file's parameters. Make sure to set TASK to either "MNIST" cuz "SHAPES" don't work right now.

### MNIST

Make sure TASK is set to "MNIST" in the opt.py file of the output directory.

Then run by passing the opt.py path to the script `python scripts/learn_mnist_dataset.py../outputs/my_new_run/opt.py`

## Help on getting theano to use gpu

(this hints for myself on how to use the gpu... dont listen to me I broke my computer)

Put this in a ~/.theanorc (replace cuda with where your cuda is)

```
[nvcc]
flags=-D_FORCE_INLINES

[cuda]
root=/usr/loca/cuda-8.0/

[global]
device = cuda
floatX = float32
```
