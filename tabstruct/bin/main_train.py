

import os



from tabstruct.utils.logger import setup_logger
from tabstruct.utils.show import show_example
from tabstruct.utils.args import  ModelArguments, DataTrainingArguments
from tabstruct.models.model_setup import load_config, load_tokenizer, load_model
from tabstruct.utils.sanity_checks import check_parameters
from tabstruct.data_modules.data_loader import load_datasets
#from transformers import DataCollatorForSeq2Seq
from tabstruct.data_modules.preprocessing import preprocess_datasets
from tabstruct.metrics.base_metric import compute_metrics
from tabstruct.bin.training import setup_trainer, run_training, run_evaluation, run_prediction
from tabstruct.data_modules.data_collator import CustomDataCollatorForSeq2Seq


from functools import partial
import sys



def main_train(model_args, data_args, training_args):



    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    model_args.is_inference = False
    tapas_path = '../models/tapas-base' if "base" in model_args.model_name_or_path else '../models/tapas-large'
    model_args.tapas_path = tapas_path

    logger = setup_logger()

    check_parameters(model_args, data_args, training_args, logger)
    
    datasets = load_datasets(data_args, model_args, logger)

    config = load_config(data_args, model_args, logger)
    tokenizer = load_tokenizer(model_args, logger)
    model = load_model(model_args, config, logger)
    logger.info(datasets)

    if training_args.do_train and data_args.training_type != "pre-training":
        show_example(datasets, "train",logger)

    if data_args.training_type != "pre-training":
        train_dataset, eval_dataset, predict_dataset = preprocess_datasets(datasets, tokenizer, data_args, model_args, training_args, logger)
    else :
        train_dataset = datasets["train"] if training_args.do_train else None
        eval_dataset = datasets["validation"] if training_args.do_eval else None
        predict_dataset = datasets["test"] if training_args.do_predict else None
        logger.info("Preprocessing is not needed for pre-training.")

    data_collator = CustomDataCollatorForSeq2Seq(
        data_args.training_type,
        tokenizer,
        model=model,
        padding=True if data_args.pad_to_max_length else "longest",
        label_pad_token_id=-100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,)

    

    compute_metrics_ = partial(compute_metrics, tokenizer=tokenizer, data_args=data_args)
    trainer = setup_trainer(model, training_args, train_dataset, eval_dataset, tokenizer, data_collator, compute_metrics_)

    if training_args.do_train:
        run_training(trainer, data_args, training_args, logger)

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        run_evaluation(trainer, data_args, eval_dataset)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        run_prediction(trainer, tokenizer, data_args, predict_dataset, training_args)

    return trainer.state.log_history

