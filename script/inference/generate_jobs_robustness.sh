#!/bin/bash
# Robustness

# Define arrays
tasks=("ALL")
repetition_rates=(20 40)
missing_values=(0 0)
mapfile -t names < all_models_seg.txt

# Define a function to generate job.sh files
generate_job() {
  mkdir -p inference/robustness/$task/R${repetition_rates[$2]}_M${missing_values[$2]}/results
  cat <<EOF > inference/robustness/$task/R${repetition_rates[$2]}_M${missing_values[$2]}/job$index.sh
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=v$index
#SBATCH --gpus-per-node=1
#SBATCH --time=88:00:00
#SBATCH --partition=hard
#SBATCH --output=inference/robustness/$task/R${repetition_rates[$2]}_M${missing_values[$2]}/results/$name.out

python ../inference.py \\
  --encoding_type $name \\
  --do_eval \\
  --task $task \\
  --config_name microsoft/tapex-base \\
  --tokenizer_name facebook/bart-base \\
  --dataset_name ../data/dataset_and_json/inference/robustness/R${repetition_rates[$2]}_M${missing_values[$2]}  \\
  --output_dir ../models/$name/inference_robustness/R${repetition_rates[$2]}_M${missing_values[$2]} \\
  --per_device_eval_batch_size 8 \\
  --logging_dir "../logs/inference/robustness/$name/R${repetition_rates[$2]}_M${missing_values[$2]}" \\
  --logging_steps 50 \\
  --predict_with_generate \\
  --pad_to_max_length 1 \\
  --is_header 1

EOF
}

# Generating jobs
for task in "${tasks[@]}"; do
  for i in $(seq 0 $((${#missing_values[@]} - 1))); do
    index=0
    for name in "${names[@]}"; do
      
      echo "Generated job for: $name"
      generate_job $task $i $name $index

      if [ "$1" == "true" ]; then
          sbatch inference/robustness/$task/R${repetition_rates[$i]}_M${missing_values[$i]}/job$index.sh
      fi
      ((index++))
    done 
  done
done



