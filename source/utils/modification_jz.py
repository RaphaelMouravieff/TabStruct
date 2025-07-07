


mapping = {"facebook/bart-base":"/gpfswork/rech/kns/uxe25ug/TabStruct/models/bart-base",
            "facebook/bart-large":"/gpfswork/rech/kns/uxe25ug/TabStruct/models/bart-large",
            "microsoft/tapex-base":"/gpfswork/rech/kns/uxe25ug/TabStruct/models/tapex-base",
            "microsoft/tapex-large":"/gpfswork/rech/kns/uxe25ug/TabStruct/models/tapex-large",
            "google/tapas-base": "/gpfswork/rech/kns/uxe25ug/TabStruct/models/tapas-base",
            "google/tapas-large": "/gpfswork/rech/kns/uxe25ug/TabStruct/models/tapas-large"}

def jz_modifs(model_args, data_args, training_args, logger):


    if model_args.model_name_or_path in ["facebook/bart-base","facebook/bart-large","microsoft/tapex-base", "microsoft/tapex-large"]:
        old_name = model_args.model_name_or_path
        model_args.model_name_or_path = mapping[model_args.model_name_or_path]
        logger.info(f"modification of model_args.model_name_or_path :\nfrom {old_name} -> to {model_args.model_name_or_path}")

    if model_args.config_name in ["facebook/bart-base","facebook/bart-large","microsoft/tapex-base", "microsoft/tapex-large"]:
        old_name = model_args.config_name
        model_args.config_name = mapping[model_args.config_name]
        logger.info(f"modification of model_args.config_name :\nfrom {old_name} -> to {model_args.config_name}")

    if model_args.tokenizer_name in ["facebook/bart-base","facebook/bart-large","microsoft/tapex-base", "microsoft/tapex-large"]:
        old_name = model_args.tokenizer_name
        model_args.tokenizer_name = mapping[model_args.tokenizer_name]
        logger.info(f"modification of model_args.tokenizer_name :\nfrom {old_name} -> to {model_args.tokenizer_name}")


    if not model_args.is_inference:
        if model_args.tapas_path in ['google/tapas-base', 'google/tapas-large']:
            old_name = model_args.tapas_path
            model_args.tapas_path = mapping[model_args.tapas_path]
            logger.info(f"modification of model_args.tapas_path :\nfrom {old_name} -> to {model_args.tapas_path}")



    assert model_args.model_name_or_path not in ["facebook/bart-base","facebook/bart-large","microsoft/tapex-base", "microsoft/tapex-large"], \
        "can not load models from huggingface when in JZ"


    if data_args.dataset_name:
        assert data_args.dataset_name not in ["wikitablequestions"], \
            "can not load dataset from huggingface when in JZ"
        
    return model_args, data_args, training_args


