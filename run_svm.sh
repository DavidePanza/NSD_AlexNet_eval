#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000
#SBATCH --qos=standard
#SBATCH --time=4-6:00:00
#SBATCH --job-name=svm


# setting vars 
which_cats='singleCat'  # 'multipleCatsNoPersons' 'multipleCats' 'singleCat'
n_shuffle_svm=10

# run
python svm.py $which_cats $n_shuffle_svm 
