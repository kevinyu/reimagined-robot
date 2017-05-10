# PUth this in a ~/.theanorc (replace cuda with where cuda is)
# oh this hints for myself on how to use the gpu... dont listen to me I broke my computer

[nvcc]
flags=-D_FORCE_INLINES

[cuda]
root=/usr/loca/cuda-8.0/

[global]
device = gpu
floatX = float32


# DUDE DO THIS TOO

(Okay the point of this is basically to add the src directory to front of PYTHONPATH)

edit your conda environment activation script at 

~/miniconda2/envs/<env-name>/

and add two directories, etc/conda/activate.d
 and etc/conda/deactivate.d
 then touch etc/conda/activate.d/env_vars.sh
 then touch etc/conda/deactivate.d/env_vars.sh

 Then add in to the activate one

 export WHATEVER U WANT

 and deactivate 

 unset WHATEVER YOU WANT

 NOICE!!!!!

and set the right environment variables for most goodest stuff


# Running junk

heres some hints how to run stuff... most of it probably doesnt work

you should proably modify the SAVE_DIR in the config.py before running a new setting or clear out the directory
it saves intermediate parameters

### MNIST (this used to work, but dont know if it does anymore)

you must set SHAPES = False in the config.py

then `python scripts/learn_mnist_dataset.py`

### Shapes shapes shapes

you must set SHAPES = True in the config.py
open shapes/properties/__init__.py to modify what shapes and settings are in play and shit
    - modify the individual classes to change what transformations are available
    - modify the properties list in that file to change which properties are available

then `python scripts/learn_shapes_dataset.py` for relief

