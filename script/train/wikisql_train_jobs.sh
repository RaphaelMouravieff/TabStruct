#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
echo "BASE_DIR: $BASE_DIR"

tasks=("wikisql")
mapfile -t all_models < "$BASE_DIR/all_models.txt"

generate_job() {
  cat <<EOF > $BASE_DIR/jobs/train/$2/wikisql.sh
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=v$3
#SBATCH --partition=hard
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=88:00:00
#SBATCH --output=$BASE_DIR/jobs/train/$2/results/${tasks[$1]}.out

python $BASE_DIR/run.py \\
  --encoding_type $2 \\
  --model_name_or_path facebook/bart-base \\
  --do_train \\
  --do_eval \\
  --dataset_name $BASE_DIR/data/dataset_and_json/wikisql \\
  --output_dir $BASE_DIR/models/$2/${tasks[$1]} \\
  --config_name microsoft/tapex-base \\
  --tokenizer_name facebook/bart-base \\
  --per_device_train_batch_size 2 \\
  --gradient_accumulation_steps 12 \\
  --per_device_eval_batch_size 4 \\
  --learning_rate 3e-5 \\
  --eval_steps 1000 \\
  --save_steps 4000 \\
  --warmup_steps 1000 \\
  --evaluation_strategy steps \\
  --predict_with_generate \\
  --num_beams 5 \\
  --weight_decay 1e-2 \\
  --label_smoothing_factor 0.1 \\
  --max_steps 20000 \\
  --logging_dir "$BASE_DIR/logs/train/$2/${tasks[$1]}" \\
  --logging_steps 10 \\
  --overwrite_output_dir 1 \\
  --overwrite_cache 1 \\
  --pad_to_max_length 1 \\
  --max_source_length 1024 \\
  --show_tokenization 0 \\
  --task train

EOF
}

for task in "${tasks[@]}"; do
  index=0
  for name in "${all_models[@]}"; do

    mkdir -p $BASE_DIR/jobs/train/$name/results
    echo "Generated job for: $name count: $index"
    generate_job $task $name $index

    if [ "$1" == "true" ]; then
      sbatch $BASE_DIR/jobs/train/$name/wikisql.sh
    fi
    ((index++))


  done
done
