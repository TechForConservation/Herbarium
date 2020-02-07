#!/bin/bash

#SBATCH --job-name=flowering_or_not_densenet_cropped_cont2
#SBATCH --output=flowering_densenet_cropped_cont2.txt
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:v100:1
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --mail-user=tony.odongo@yale.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# using your anaconda environment
module purge
module restore cuda_10
source activate tf_10
env > environment.txt
which python
python /gpfs/loomis/home.grace/teo22/project/Herbarium/CNN_herb_densenet_cropped_cont2.py
