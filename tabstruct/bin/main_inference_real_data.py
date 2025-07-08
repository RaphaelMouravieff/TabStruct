
import torch

from tabstruct.utils.paths import  get_max_checkpoint
from tabstruct.models.model_setup import load_config, load_tokenizer, load_model
from tabstruct.data_modules.data_loader import load_datasets
from tabstruct.data_modules.preprocessing import preprocess_datasets
from tabstruct.metrics.base_metric import compute_metrics
from tabstruct.bin.training import setup_trainer, run_evaluation, run_prediction

from functools import partial

from transformers import (DataCollatorForSeq2Seq, set_seed)
from datasets import DatasetDict

import json
import os


def main_inference_real_data(model_args, data_args, training_args, logger):

 
    set_seed(training_args.seed)

    logger.info(f"training_args : \n{training_args}")
    logger.info(f"data_args : \n{data_args}")
    logger.info(f"model_args : \n{model_args}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()
    logger.info(f"device : {device}")

    tokenizer = load_tokenizer(model_args, logger)

    max_checkpoint, save_path = get_max_checkpoint(model_args, logger)
    logger.info(f"max_checkpoint : {max_checkpoint}")

    file_path_test = f"{save_path}/test_{model_args.encoding_type}.json"
    file_path_validation = f"{save_path}/validation_{model_args.encoding_type}.json"
    file_valid = os.path.exists(file_path_test) and any(key.startswith("checkpoint") for key in json.load(open(file_path_test)))

    if  max_checkpoint is None:
        logger.info('Starting validation..')
        run_validation(tokenizer, model_args, data_args, training_args, logger, device, save_path)

        max_checkpoint, save_path = get_max_checkpoint(model_args, logger)
        logger.info(f"After validation max_checkpoint : {max_checkpoint}")


    if max_checkpoint is not None and not file_valid:
        logger.info('Starting test..')
        run_test(tokenizer, model_args, data_args, training_args, logger, device, save_path, max_checkpoint)

    if file_valid:
        logger.info('Not runing test and validation there is already a test result.')
        
        with open(file_path_validation, 'r') as f:
            data = json.load(f)
        logger.info(f"Results checkoints {model_args.task}:\n {data}")


        with open(file_path_test, 'r') as f:
            data = json.load(f)

        logger.info(f"Best checkoint result test {model_args.task}:\n {data}")



def run_test(tokenizer, model_args, data_args, training_args, logger, device, save_path, max_checkpoint):
    training_args.do_predict = True
    training_args.do_eval = False


    logger.info(training_args.do_eval)

    current_path = os.path.abspath(__file__)
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
    models_path = os.path.join(base_path, 'models', model_args.encoding_type, model_args.task)

    datasets = load_datasets(data_args, model_args, logger)

    dataset_test = DatasetDict({"test": datasets["test"]})
    #show_example(dataset_test, "test", logger)
    _, _, dataset_test = preprocess_datasets(dataset_test, tokenizer, data_args, model_args, training_args, logger)


    logger.info(f'Test on best ckpt : {max_checkpoint}')
    model_ckpt_path = os.path.join(models_path, max_checkpoint)
    logger.info(f'model_ckpt_path : {model_ckpt_path}')

    model_args.model_name_or_path = os.path.join(models_path, max_checkpoint)
    config = load_config(data_args, model_args, logger)
    model = load_model(model_args, config, logger)
    model.to(device)

        

    data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )


    compute_metrics_ = partial(compute_metrics, tokenizer=tokenizer, data_args=data_args)

    trainer = setup_trainer(model, training_args, dataset_test, dataset_test, tokenizer, data_collator, compute_metrics_)
    

    logger.info("*** Evaluate ***")
    trainer = setup_trainer(model, training_args, dataset_test, dataset_test, tokenizer, data_collator, compute_metrics_)
    metrics  = run_prediction(trainer, data_args, dataset_test)


    accuracy = metrics['predict_denotation_accuracy']
    loss = metrics["predict_loss"]
    results_inference = {} 
    results_inference[max_checkpoint] = {"accuracy" : accuracy, "loss" : loss}


    with open(f"{save_path}/test_{model_args.encoding_type}.json", 'w') as json_file:
        json.dump(results_inference, json_file)

    
def run_validation(tokenizer, model_args, data_args, training_args, logger, device, save_path):
    training_args.do_train = False
    # Setup path :
    # get checkpoints : 
    current_path = os.path.abspath(__file__)
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
    models_path = os.path.join(base_path, 'models', model_args.encoding_type, model_args.task)
    checkpoints_path = [path for path in os.listdir(models_path) if path.startswith('checkpoint')]


    datasets = load_datasets(data_args, model_args, logger)
    #datasets['validation'] = datasets['validation'].select(list(range(11)))

    dataset_eval = DatasetDict({"validation": datasets["validation"]})
    
    #show_example(dataset_eval, "validation", logger)
    _, dataset_eval, _ = preprocess_datasets(dataset_eval, tokenizer, data_args, model_args, training_args, logger)

    
    results_inference = {}
    for checkpoint in checkpoints_path:
        logger.info(f'Validation on ckpt : {checkpoint}')
        model_ckpt_path = os.path.join(models_path, checkpoint)
        logger.info(f'model_ckpt_path : {model_ckpt_path}')
        
        model_args.model_name_or_path = os.path.join(models_path, checkpoint)
        config = load_config(data_args, model_args, logger)
        model = load_model(model_args, config, logger)
        model.to(device)

        data_collator = DataCollatorForSeq2Seq(
                data_args.training_type,
                tokenizer,
                model=model,
                label_pad_token_id=-100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
                pad_to_multiple_of=8 if training_args.fp16 else None,
            )
        compute_metrics_ = partial(compute_metrics, tokenizer=tokenizer, data_args=data_args)

        
        logger.info("*** Evaluate ***")
        trainer = setup_trainer(model, training_args, dataset_eval, dataset_eval, tokenizer, data_collator, compute_metrics_)
        run_evaluation(trainer, data_args, dataset_eval)
        history = trainer.state.log_history[0]
            
        logger.info(f"history : {history}")
        accuracy = history["eval_denotation_accuracy"]
        loss = history["eval_loss"]

        results_inference[checkpoint] = {"accuracy" : accuracy, "loss" : loss}

        del model


    


    with open(f"{save_path}/validation_{model_args.encoding_type}.json", 'w') as json_file:
        json.dump(results_inference, json_file)






