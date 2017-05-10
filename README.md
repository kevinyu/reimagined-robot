# PUth this in a ~/.theanorc (replace cuda with where cuda is)

[nvcc]
flags=-D_FORCE_INLINES

[cuda]
root=/usr/loca/cuda-8.0/

[global]
device = gpu
floatX = float32
# warn_float64=pdb


# DUDE DO THIS TOO

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
