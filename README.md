## Setup

Set the src/ directory to the front of your PYTHONPATH

Can do this by doing `export PYTHONPATH=<path-to-src>`

To have conda environment automatically set your PYTHONPATH, edit your conda env activation script at (cd) `~/miniconda2/envs/<env-name>/`

* Add two directories, `etc/conda/activate.d` and `etc/conda/deactivate.d`

* `touch etc/conda/activate.d/env_vars.sh`

* `touch etc/conda/deactivate.d/env_vars.sh`

* In `etc/conda/activate.d/env_vars.sh`, add the lines

    ```
    export PROJECT_ROOT="<path-to-repo-root>"
    export PYTHONPATH="${PROJECT_ROOT}/src"
    ```

* In `etc/conda/deactivate.d/env_vars.sh`, add the line `unset PYTHONPATH`

    ```
    unset PROJECT_ROOT
    unset PYTHONPATH
    ```

 Now, you can enter environment with `source activate <env-name>` and leave with `source deactivate`

## Running Stuff

* Create a new output dir called <name> in the outputs/ directory, `mkdir outputs/my_new_run/`

* Create a outputs dir in there `mkdir outputs/my_new_run/outputs`

* Copy the file `src/config_template.py` to `outputs/<name>/opt.py`

* Edit the `opt.py` file's parameters. Make sure to set TASK to either "MNIST" or "SHAPES"

### MNIST

Make sure TASK is set to "MNIST" in the opt.py file of the output directory.

Then run by passing the opt.py path to the script `python scripts/learn_mnist_dataset.py../outputs/my_new_run/opt.py`

### Shapes shapes shapes

Make sure TASK is set to "SHAPES" in the opt.py file of the output directory.

Then run by passing the opt.py path to the script `python scripts/learn_shapes_dataset.py../outputs/my_new_run/opt.py`

Open shapes/properties/__init__.py to modify what shapes and settings are in play and shit

    * modify the individual classes to change what transformations are available

    * modify the properties list in that file to change which properties are available

## Help on getting theano to use gpu

(this hints for myself on how to use the gpu... dont listen to me I broke my computer)

Put this in a ~/.theanorc (replace cuda with where cuda is)

```
[nvcc]
flags=-D_FORCE_INLINES

[cuda]
root=/usr/loca/cuda-8.0/

[global]
device = gpu
floatX = float32
```
