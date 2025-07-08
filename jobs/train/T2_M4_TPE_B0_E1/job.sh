#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=v97
#SBATCH --gpus-per-node=1
#SBATCH --time=88:00:00
#SBATCH --partition=hard
#SBATCH --output=/home/raphael.gervillie/TabStructGit/jobs/train/T2_M4_TPE_B0_E1/results/ALL.out

python /home/raphael.gervillie/TabStructGit/run.py \
  --encoding_type T2_M4_TPE_B0_E1 \
  --model_name_or_path facebook/bart-base \
  --config_name microsoft/tapex-base \
  --tokenizer_name facebook/bart-base \
  --do_train \
  --do_eval \
  --train_file "/home/raphael.gervillie/TabStructGit/data/dataset_and_json/train/train_ALL.json" \
  --validation_file "/home/raphael.gervillie/TabStructGit/data/dataset_and_json/train/valid_ALL.json"  \
  --output_dir /home/raphael.gervillie/TabStructGit/models/T2_M4_TPE_B0_E1/ALL \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 8 \
  --learning_rate 3e-5 \
  --eval_steps 5000 \
  --save_steps 5000 \
  --warmup_steps 0 \
  --evaluation_strategy steps \
  --predict_with_generate \
  --num_beams 5 \
  --weight_decay 1e-2 \
  --label_smoothing_factor 0.1 \
  --max_steps 1000000 \
  --max_query_length 41 \
  --max_labels_length 87 \
  --max_target_length 128 \
  --max_source_length 512 \
  --logging_dir "/home/raphael.gervillie/TabStructGit/logs/train/T2_M4_TPE_B0_E1/ALL" \
  --logging_steps 50 \
  --overwrite_output_dir 1 \
  --overwrite_cache 1 \
  --pad_to_max_length 1 \
  --save_strategy steps \
  --save_total_limit 1 \
  --load_best_model_at_end 1 \
  --task train
