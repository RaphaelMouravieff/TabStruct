#!/bin/bash

tasks=("ALL")
mapfile -t all_models < all_models_seg.txt

generate_job() {
  cat <<EOF > train/$2/job.sh
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=v$3
#SBATCH --gpus-per-node=1
#SBATCH --time=88:00:00
#SBATCH --partition=hard
#SBATCH --output=train/$2/results/${tasks[$1]}.out

python ../run.py \\
  --encoding_type $2 \\
  --model_name_or_path facebook/bart-base \\
  --config_name microsoft/tapex-base \\
  --tokenizer_name facebook/bart-base \\
  --do_train \\
  --do_eval \\
  --train_file "../data/dataset_and_json/train/train_${tasks[$1]}.json" \\
  --validation_file "../data/dataset_and_json/train/valid_${tasks[$1]}.json"  \\
  --output_dir ../models/$2/${tasks[$1]} \\
  --per_device_train_batch_size 8 \\
  --gradient_accumulation_steps 1 \\
  --per_device_eval_batch_size 8 \\
  --learning_rate 3e-5 \\
  --eval_steps 5000 \\
  --save_steps 5000 \\
  --warmup_steps 0 \\
  --evaluation_strategy steps \\
  --predict_with_generate \\
  --num_beams 5 \\
  --weight_decay 1e-2 \\
  --label_smoothing_factor 0.1 \\
  --max_steps 1000000 \\
  --max_query_length 41 \\
  --max_labels_length 87 \\
  --max_target_length 128 \\
  --max_source_length 512 \\
  --logging_dir "../logs/train/$2/${tasks[$1]}" \\
  --logging_steps 50 \\
  --overwrite_output_dir 1 \\
  --overwrite_cache 1 \\
  --pad_to_max_length 1 \\
  --save_strategy steps \\
  --save_total_limit 1 \\
  --load_best_model_at_end 1 \\
  --show_tokenization 0 \\
  --is_header 1
EOF
}

for task in "${tasks[@]}"; do
  index=0
  for name in "${all_models[@]}"; do

    mkdir -p train/$name/results
    echo "Generated job for: $name count: $index"
    generate_job $task $name $index

    if [ "$1" == "true" ]; then
      sbatch train/$name/job.sh
    fi
    ((index++))
  done
done

