#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
echo "BASE_DIR: $BASE_DIR"

mapfile -t all_models < $BASE_DIR/all_models.txt





generate_job() {
  cat <<EOF > $BASE_DIR/jobs/test/$name/compositional.sh
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=v$index
#SBATCH --gpus-per-node=1
#SBATCH --time=88:00:00
#SBATCH --partition=hard
#SBATCH --output=$BASE_DIR/jobs/test/$name/results/compositional.out

python $BASE_DIR/run.py \\
  --task test_synthetic \\
  --encoding_type $name \\
  --do_eval \\
  --config_name microsoft/tapex-base \\
  --tokenizer_name facebook/bart-base \\
  --dataset_name $BASE_DIR/data/test/compositional/ALL  \\
  --output_dir $BASE_DIR/models/$name/compositional/ALL \\
  --per_device_eval_batch_size 8 \\
  --logging_dir $BASE_DIR/logs/test/compositional/ALL/$name \\
  --logging_steps 50 \\
  --predict_with_generate \\
  --pad_to_max_length 1 \\
  --is_header 1

EOF
}

index=0
for name in "${all_models[@]}"; do
  mkdir -p $BASE_DIR/jobs/test/$name/results
  echo "Compositional job for: $name count: $index"
  generate_job $task $name $index
  if [ "$1" == "true" ]; then
    sbatch $BASE_DIR/jobs/test/$name/compositional.sh
  fi
  ((index++))
done

