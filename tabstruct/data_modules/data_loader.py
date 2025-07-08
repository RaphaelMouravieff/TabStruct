from datasets import load_dataset, load_from_disk


def load_datasets(data_args, model_args, logger):

    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        if data_args.dataset_name in ["wikitablequestions"] : 
            datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir)
            logger.info(f'Load dataset {data_args.dataset_name}')

        else:
            try:
                logger.info(f"try load from disk ..")
                datasets = load_from_disk(data_args.dataset_name)
                logger.info(f"Load dataset From Disk PATH = {data_args.dataset_name}")
            except:
                pass
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        
        if model_args!="PT":
            datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
        if model_args=="PT":
            datasets = load_from_disk(data_args.train_file)

    logger.info("datasets",datasets)    
    return datasets



def load_inference_heat_map(data_args, logger):

    datasets = load_from_disk(data_args.dataset_name)
        
    logger.info("datasets",datasets)
    
    return datasets