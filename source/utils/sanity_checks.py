import torch

def check_parameters(model_args, data_args, training_args, logger):

    encoding_types = model_args.encoding_type.split('_')
    if len(encoding_types) == 5:
        input_token_structure, mask_sparsity_level, positional_embedding, encoding_structure_bias, tabular_structure_embedding = encoding_types
        question_in_decoder = False
    if len(encoding_types) == 6:
        input_token_structure, mask_sparsity_level, positional_embedding, encoding_structure_bias, tabular_structure_embedding, question_in_decoder = encoding_types
        question_in_decoder = True
        
    model_args.tabular_structure_embedding = tabular_structure_embedding
    model_args.encoding_structure_bias = encoding_structure_bias 
    model_args.positional_embedding = positional_embedding
    model_args.mask_sparsity_level = mask_sparsity_level
    model_args.input_token_structure = input_token_structure
    #model_args.question_in_decoder = question_in_decoder

    #logger.info(f"Training/evaluation parameters {training_args}")mask_sparsity_level
    logger.info(f"Data-args max_source_length : {data_args.max_source_length}")
    logger.info(f"input_token_structure : {model_args.input_token_structure}")
    logger.info(f"mask_sparsity_level : {model_args.mask_sparsity_level}")
    logger.info(f"positional_embedding : {model_args.positional_embedding}")
    logger.info(f"encoding_structure_bias : {model_args.encoding_structure_bias}")
    logger.info(f"tabular_structure_embedding : {model_args.tabular_structure_embedding}")
    logger.info(f"question_in_decoder : {model_args.question_in_decoder}")


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

    #if data_args.is_wtq or data_args.training_type in ["pre-training","pre-training-tokenize"]:
    #    assert  data_args.max_source_length == 1024, \
    #    "If using wikitablequestions or wikisql, set --max_source_length 1024"

    #elif not data_args.is_wtq:
    #    assert  data_args.max_source_length == 512, \
    #    "If not using wikitablequestions or wikisql, set --max_source_length 512"

    assert model_args.input_token_structure in ["T0", "T1", "T2"], \
        "input_token_structure must be one of: 'T0', 'T1' or 'T2'"

    assert model_args.mask_sparsity_level in [f"M{i}" for i in range(9)], \
        "mask_sparsity_level must be one of: 'M0' to 'M9'"

    assert model_args.tabular_structure_embedding in ["E1", "E0"], \
        "tabular_structure_embedding must be either 'E1' or 'E0'"

    assert model_args.encoding_structure_bias in ["B3", "B2", "B1", "B0"], \
        "encoding_structure_bias must be either 'B1' or 'B0'"

    assert model_args.positional_embedding in ["TPE", "CPE"], \
        "tabular_structure_embedding must be either 'TPE' or 'CPE'"
    
    if data_args.training_type == "pre-training":
        assert model_args.input_token_structure in data_args.dataset_name, \
            f"If tabular_structure_embedding is '{model_args.input_token_structure}', you should use a tokenized dataset for pre-training. (use training_type = 'pre-training-tokenize')"
        

    assert data_args.max_query_length + data_args.max_labels_length ==  data_args.max_target_length, \
        f"max_query_length ({data_args.max_query_length}) + max_labels_length ({data_args.max_labels_length}) ==  max_target_length ({data_args.max_target_length})"


    if model_args.mask_sparsity_level in ["M4","M5","M6"]:
        assert model_args.input_token_structure == "T2", \
            "input_token_structure must be 'T2' when mask_sparsity_level is 'M4', 'M5', 'M6'"


    assert model_args.config_name is not None, \
        "Specify a config_name; use 'microsoft/tapex-base' or 'microsoft/tapex-large'."

    assert model_args.tokenizer_name is not None, \
        "Specify a tokenizer_name; use 'facebook/bart-base' or 'facebook/bart-large'."

    # attention type checks:
    if model_args.attention_type == "flash":
        raise NotImplementedError("Flash attention is not supported at the moment.")

    assert model_args.attention_type in ["flex", "sdpa"], \
        f"model_args.attention_type must be either 'flex' or 'sdpa', but got {model_args.attention_type}"

    if model_args.attention_type == "flex":
        assert model_args.mask_sparsity_level == "M3", \
            "When attention_type is 'flex', mask_sparsity_level must be 'M3'."
        assert model_args.encoding_structure_bias == "B0", \
            "encoding_structure_bias must be False when using 'flex' attention, as bias is not supported"