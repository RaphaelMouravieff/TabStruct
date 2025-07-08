

from tabstruct.utils.logger import setup_logger
from tabstruct.utils.sanity_checks import check_parameters
from tabstruct.bin.main_inference_real_data import main_inference_real_data
from tabstruct.bin.main_inference_synthetic_data import main_inference_synthetic_data



def main_inference(model_args, data_args, training_args):
    

    logger = setup_logger()

    #assert not (model_args.task and model_args.model_name_or_path), \
    #    "model_name_or_path cannot be specified if task is specified"

    model_args.is_inference = True

    check_parameters(model_args, data_args, training_args, logger)


    if model_args.task == "test_wikisql":
        logger.info(f"Start inference for real datasets")
        main_inference_real_data( model_args, data_args, training_args, logger)

    if model_args.task == "test_synthetic":
        logger.info(f"Start inference for synthetic datasets")
        main_inference_synthetic_data( model_args, data_args, training_args, logger)

