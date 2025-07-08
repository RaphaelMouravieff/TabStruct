#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=v0
#SBATCH --time=88:00:00
#SBATCH --output=script/results/synthetic_train.out

python data/data_processing/create_train.py \
  --grammar ALL  \
  --n_samples 1000 \
  --numeric_range 0 999 \
  --max_column 8 \
  --min_column 6 \
  --max_row 8 \
  --min_row 6 \
  --output_path data/train_synthetic.json


python data/data_processing/create_train.py \
  --grammar ALL  \
  --n_samples 1000 \
  --numeric_range 0 999 \
  --max_column 8 \
  --min_column 6 \
  --max_row 8 \
  --min_row 6 \
  --output_path data/valid_synthetic.json



python data/data_processing/create_inference.py \
  --grammar COMPO \
  --numeric_range 0 999 \
  --output_path data/COMPO \
  --repetition_rate 0 \
  --missing_values 0
  