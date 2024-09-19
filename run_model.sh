#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=8000
#SBATCH --qos=standard
#SBATCH --time=3-6:00:00
#SBATCH --job-name=model train


# load modules or conda environments here
module add CUDA/12.0.0

# script to run
which_cats='multipleCats' # 'multipleCatsNoPersons' 'singleCat' 'multipleCats' 
n_classes='all' #'all' #35
n_shuffle=2
n_iterations=10
n_folds=10
batch_size=64

# script to run
python alexnet.py $which_cats $n_classes $n_shuffle $n_iterations $n_folds $batch_size 


