import torch

def check_parameters(model_args, data_args, training_args, logger):

    encoding_types = model_args.encoding_type.split('_')
    input_token_structure, mask_sparsity_level, positional_embedding, encoding_structure_bias, tabular_structure_embedding = encoding_types
        
    model_args.tabular_structure_embedding = tabular_structure_embedding
    model_args.encoding_structure_bias = encoding_structure_bias 
    model_args.positional_embedding = positional_embedding
    model_args.mask_sparsity_level = mask_sparsity_level
    model_args.input_token_structure = input_token_structure

    logger.info(f"Data-args max_source_length : {data_args.max_source_length}")
    logger.info(f"input_token_structure : {model_args.input_token_structure}")
    logger.info(f"mask_sparsity_level : {model_args.mask_sparsity_level}")
    logger.info(f"positional_embedding : {model_args.positional_embedding}")
    logger.info(f"encoding_structure_bias : {model_args.encoding_structure_bias}")
    logger.info(f"tabular_structure_embedding : {model_args.tabular_structure_embedding}")

    logger.info(f"per_device_train_batch_size = {training_args.per_device_train_batch_size}")
    logger.info(f"gradient_accumulation_steps = {training_args.gradient_accumulation_steps}")
    GPUS = torch.cuda.device_count()
    logger.info(f"num_gpus = {GPUS}")
    logger.info(f"batch size = {GPUS * training_args.gradient_accumulation_steps * training_args.per_device_train_batch_size}")

    if training_args.resume_from_checkpoint is not None:
        if model_args.model_name_or_path is None:
            model_args.model_name_or_path = training_args.resume_from_checkpoint
            logger.info(f"model_name_or_path is None and  resume_from_checkpoint is not None setting model_name_or_path = resume_from_checkpoint")

    data_args.is_wtq = any(keyword in data_args.dataset_name for keyword in ["robut", "wikitablequestions", "tiny_wtq", "wikisql"]) if data_args.dataset_name else False
    
    logger.info(f"is real data ? : {data_args.is_wtq}")

    assert model_args.input_token_structure in ["T0", "T1", "T2"], \
        "input_token_structure must be one of: 'T0', 'T1' or 'T2'"

    assert model_args.mask_sparsity_level in [f"M{i}" for i in range(7)], \
        "mask_sparsity_level must be one of: 'M0' to 'M6'"

    assert model_args.tabular_structure_embedding in ["E0", "E1"], \
        "tabular_structure_embedding must be either 'E0' or 'E1'"

    assert model_args.encoding_structure_bias in ["B0", "B1"], \
        "encoding_structure_bias must be either 'B0' or 'B1' "

    assert model_args.positional_embedding in ["TPE", "CPE"], \
        "tabular_structure_embedding must be either 'TPE' or 'CPE'"
    
    if model_args.mask_sparsity_level in ["M4","M5","M6"]:
        assert model_args.input_token_structure == "T2", \
            "input_token_structure must be 'T2' when mask_sparsity_level is 'M4', 'M5', 'M6'"

    assert model_args.config_name is not None, \
        "Specify a config_name; use 'microsoft/tapex-base' or 'microsoft/tapex-large'."

    assert model_args.tokenizer_name is not None, \
        "Specify a tokenizer_name; use 'facebook/bart-base' or 'facebook/bart-large'."

    if model_args.attention_type == "flash":
        raise NotImplementedError("Flash attention is not supported at the moment.")

    assert model_args.attention_type in ["flex", "sdpa"], \
        f"model_args.attention_type must be either 'flex' or 'sdpa', but got {model_args.attention_type}"

    if model_args.attention_type == "flex":
        assert model_args.mask_sparsity_level == "M3", \
            "When attention_type is 'flex', mask_sparsity_level must be 'M3'."
        assert model_args.encoding_structure_bias == "B0", \
            "encoding_structure_bias must be False when using 'flex' attention, as bias is not supported"
        

    if "large" in model_args.model_name_or_path:
        assert "large" in model_args.tapas_path, \
            "When a large model is used, tapas_path should also be a large model"

    if "base" in model_args.model_name_or_path:
        assert "base" in model_args.tapas_path, \
            "When a base model is used, tapas_path should also be a base model"