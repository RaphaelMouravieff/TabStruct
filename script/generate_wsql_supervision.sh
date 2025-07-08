#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=v0
#SBATCH --time=88:00:00
#SBATCH --output=train/results/ALL.out

python data/data_processing/create_wsql.py 





  