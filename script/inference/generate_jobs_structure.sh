# Structure
#!/bin/bash


mapfile -t names < all_models_seg.txt


tasks=("ALL")

generate_job() {
  mkdir -p inference/structure/${tasks[$task]}/results
  cat <<EOF > inference/structure/${tasks[$task]}/job$index.sh
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=v$index
#SBATCH --gpus-per-node=1
#SBATCH --time=88:00:00
#SBATCH --partition=hard
#SBATCH --output=inference/structure/${tasks[$task]}/results/job$index_$2_${tasks[$task]}.out

python ../inference.py \\
  --encoding_type $2 \\
  --do_eval \\
  --task ${tasks[$task]} \\
  --config_name microsoft/tapex-base \\
  --tokenizer_name facebook/bart-base \\
  --dataset_name ../data/dataset_and_json/inference/structure/${tasks[$task]}  \\
  --output_dir ../models/$2/inference_structure/${tasks[$task]} \\
  --per_device_eval_batch_size 8 \\
  --logging_dir "../logs/inference/structure/${tasks[$task]}/$2" \\
  --logging_steps 50 \\
  --predict_with_generate \\
  --pad_to_max_length 1 \\
  --is_header 1

EOF
}

for task in $(seq 0 $((${#tasks[@]} - 1))); do
    index=0
    for name in "${names[@]}"; do
      generate_job $task $name $index
      if [ "$1" == "true" ]; then
        sbatch inference/structure/${tasks[$task]}/job$index.sh
      fi
      ((index++))
    done
done



