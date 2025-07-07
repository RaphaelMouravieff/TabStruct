#!/bin/bash

# Define arrays
tasks=("ALL")
input_token_structures=("T0" "RIC" "RCC")
mask_sparsity_levels=("M0" "M1" "M2" "M3" "M4" "M5" "M6")
positional_embeddings=("TPE" "CPE")
encoding_structure_biases=("false" "true") 
tabular_structure_embeddings=("NRCE" "RCE")

index=0
# Loop through each combination of the arrays
for task in $(seq 0 $((${#tasks[@]} - 1))); do
  for input_token_structure in $(seq 0 $((${#input_token_structures[@]} - 1))); do
    for mask_sparsity_level in $(seq 0 $((${#mask_sparsity_levels[@]} - 1))); do
      for positional_embedding in $(seq 0 $((${#positional_embeddings[@]} - 1))); do
        for encoding_structure_bias in $(seq 0 $((${#encoding_structure_biases[@]} - 1))); do
          for tabular_structure_embedding in $(seq 0 $((${#tabular_structure_embeddings[@]} - 1))); do

            # Create the name by combining the array elements
            name=${input_token_structures[$input_token_structure]}_${mask_sparsity_levels[$mask_sparsity_level]}_${positional_embeddings[$positional_embedding]}_${encoding_structure_biases[$encoding_structure_bias]}_${tabular_structure_embeddings[$tabular_structure_embedding]}
            
            # Add the second condition: Skip if input_token_structure is "T0" and mask_sparsity_level is in "M6","M5" or "M4"
            if [[ "${input_token_structures[$input_token_structure]}" == "T0" && ( "${mask_sparsity_levels[$mask_sparsity_level]}" == "M6" || "${mask_sparsity_levels[$mask_sparsity_level]}" == "M5" || "${mask_sparsity_levels[$mask_sparsity_level]}" == "M4" ) ]]; then
              continue
            fi

            # Add the second condition: Skip if input_token_structure is "T0" and mask_sparsity_level is in "M6","M5" or "M4"
            if [[ "${input_token_structures[$input_token_structure]}" == "RIC" && ( "${mask_sparsity_levels[$mask_sparsity_level]}" == "M6" || "${mask_sparsity_levels[$mask_sparsity_level]}" == "M5" || "${mask_sparsity_levels[$mask_sparsity_level]}" == "M4" ) ]]; then
              continue
            fi

            echo "$name" >> all_models_acl.txt
        
            echo "Add : $name"
            echo "Counter : $index"

            ((index++))
          done
        done
      done
    done
  done
done

