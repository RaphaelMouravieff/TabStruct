# !/bin/bash
tasks=("WSQL_bart_base")
mapfile -t names < acl_literaturevs.txt

# Define a function to generate job.sh files
generate_job() {
  mkdir -p inference/wikisql/$task/results
  cat <<EOF > inference/wikisql/$task/wsbab$index.sh
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=wsbab$index
#SBATCH --partition=hard
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=88:00:00
#SBATCH --output=inference/wikisql/$task/results/$name.out

python ../inference.py \\
  --encoding_type $name \\
  --do_eval \\
  --task $task \\
  --config_name microsoft/tapex-base \\
  --tokenizer_name facebook/bart-base \\
  --dataset_name ../data/dataset_and_json/train/wikisql  \\
  --output_dir ../models/wikisql/$task/$name \\
  --per_device_eval_batch_size 8 \\
  --logging_dir "../logs/inference/wikisql/$task/$name" \\
  --logging_steps 50 \\
  --predict_with_generate \\
  --pad_to_max_length 1 \\
  --max_source_length 1024 \\

EOF
}

# Generating jobs
for task in "${tasks[@]}"; do
  index=0
  for name in "${names[@]}"; do
    
    echo "Generated job for: $name"
    generate_job $task $i $name $index

    if [ "$1" == "true" ]; then
        sbatch inference/wikisql/$task/wsbab$index.sh
    fi
    ((index++))
  done 
done
