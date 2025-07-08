

from transformers import (DataCollatorForSeq2Seq, set_seed)

from tabstruct.utils.show import show_example
from tabstruct.utils.create_heatmaps import create_heatmaps

from tabstruct.utils.paths import  find_checkpoint

from tabstruct.models.model_setup import load_config, load_tokenizer, load_model
from tabstruct.data_modules.data_loader import load_inference_heat_map
from tabstruct.data_modules.preprocessing import preprocess_datasets
from tabstruct.metrics.base_metric import compute_metrics
from tabstruct.bin.training import setup_trainer, run_evaluation

from functools import partial

import os 
from datasets import DatasetDict
import json
import torch




def main_inference_synthetic_data(model_args, data_args, training_args, logger):

    if model_args.model_name_or_path is None and model_args.task:
        # Get the current script's absolute path and navigate up two levels
        current_path = os.path.abspath(__file__)
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))

        # Find the best checkpoint for the specified task and encoding type

        checkpoint_path = find_checkpoint(base_path, model_args.encoding_type, model_args.task, logger)
        if checkpoint_path:
            model_args.model_name_or_path = checkpoint_path
        else:
            logger.info("No valid checkpoint found. Check the task and encoding type.")

    set_seed(training_args.seed)

    logger.info(f"training_args : \n{training_args}")
    logger.info(f"data_args : \n{data_args}")
    logger.info(f"model_args : \n{model_args}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()
    logger.info(f"device : {device}")

    config = load_config(data_args, model_args, logger)
    tokenizer = load_tokenizer(model_args, logger)
    model = load_model(model_args, config, logger)
    model.to(device)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        #padding=True if data_args.pad_to_max_length else "longest",
        label_pad_token_id=-100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,)



    compute_metrics_ = partial(compute_metrics, tokenizer=tokenizer, data_args=data_args)
    

    results_inference = {}

    datasets = load_inference_heat_map(data_args, logger)
    datasets.cleanup_cache_files()
    for row in range(4, 12):
        for col in range(4, 12):
            dataset_name = f"test_{row}_{col}"

            dataset_eval = DatasetDict({"validation": datasets[dataset_name]})

            
            show_example(dataset_eval, "validation", logger)

            _, dataset_eval, _ = preprocess_datasets(dataset_eval, tokenizer, data_args, model_args, training_args, logger)
            
            if training_args.do_eval:

                logger.info("*** Evaluate***")
                trainer = setup_trainer(model, training_args, dataset_eval, dataset_eval, tokenizer, data_collator, compute_metrics_)
                run_evaluation(trainer, data_args, dataset_eval)
                history = trainer.state.log_history[0]


            logger.info(f"history : {history}")
            accuracy = history["eval_denotation_accuracy"]
            loss = history["eval_loss"]

            results_inference[str(row)+"_"+str(col)] = {"accuracy" : accuracy, "loss" : loss}

    

    generalization = data_args.dataset_name.split('/')[-2]
    experience_name = data_args.dataset_name.split('/')[-1]
    task = model_args.task

    save_dir = f"../logs/results/{generalization}/{experience_name}"
    filename = model_args.encoding_type
    save_path = os.path.join(os.path.join(save_dir, filename), task)


    logger.info(f'filename : {filename}')
    logger.info(f'save_path : {save_path}')
    # Ensure the save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(f"{save_path}/{filename}.json", 'w') as json_file:
        json.dump(results_inference, json_file)


    create_heatmaps(results_inference, save_path)








