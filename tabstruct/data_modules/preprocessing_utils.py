import pandas as pd


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

def show_all(example, input_ids, token_type, attention_mask, decoder_input_ids, decoder_attention_mask, labels, tokenizer, logger):
    
        df = pd.DataFrame(example['table']['rows'], columns=example['table']['header'])
        logger.info(f"\nTable:\n{df}")
        logger.info(f"\Question:\n{example['question']}")
        logger.info(f"\Answers:\n{example['answers']}")

        encoded = [tokenizer.decode(i) for i in input_ids]

        X = pd.DataFrame(token_type)
        X = pd.concat([X, pd.Series(encoded)], axis=1)
        X = pd.concat([X, pd.Series(input_ids)], axis=1)
        X = pd.concat([X, pd.Series(attention_mask)], axis=1)
        X = pd.concat([X, pd.Series(labels)], axis=1)
  

        if decoder_input_ids is not None:
            X = pd.concat([X, pd.Series(decoder_input_ids)], axis=1)
            X = pd.concat([X, pd.Series(decoder_attention_mask)], axis=1)
            
        logger.info(f"\nInput:\n{X}")



def flatten(table: dict, query: str) -> str:
    header = "|".join(table['header'])  
    rows = ["|".join(map(str, row)) for row in table['rows']]  
    return f"{query}||{header}||" + "||".join(rows)


def reconstruct(flattened: str) -> tuple:
    parts = flattened.split("||")    
    query = parts[0]  
    header = parts[1].split("|") 
    rows = [row.split("|") for row in parts[2:]] 
    table = {
        "header": header,
        "rows": rows
    }
    return table, query


def are_tables_equal(table1: dict, flattened: str) -> bool:
    table2, _ = reconstruct(flattened)
    if table1["header"] != table2["header"]:
        return False
    if table1["rows"] != table2["rows"]:
        return False
    return True

def pad_sequence_decoder_att_mask(sequence, max_len, logger=None):

    padding_value = 0
    eos_value = 1

    if len(sequence) > max_len:
        if logger is not None:
            logger.info(f"Truncating sequence decoder attention mask from {len(sequence)} to {max_len} tokens.")      
        sequence = sequence[:max_len-1]
        sequence.append(eos_value)

    return sequence + [padding_value] * (max_len - len(sequence))



def get_labels(answers, query, query_length,  data_args, tokenizer, padding, pad_token_id, question_in_decoder, is_inference):

    query_length +=1
    if question_in_decoder is False:
        answers_text = ", ".join(answers) if data_args.is_wtq else answers
        labels = tokenizer(text_target=answers_text, max_length=data_args.max_target_length, padding=padding, truncation=True)

        labels = [(l if l != pad_token_id else -100) for l in labels["input_ids"]]
        decoder_input_ids = None
        decoder_attention_mask = None

    if question_in_decoder is True:
        #if is_training:

        answers = ", ".join(answers) if data_args.is_wtq else answers
        tokenizer.padding_side = "left"
        query_ids = tokenizer(query, max_length=data_args.max_query_length, padding=padding, truncation=True).input_ids[:-1]+[0]
        tokenizer.padding_side = "right"
        answer_ids = tokenizer(answers).input_ids[1:]
        answer_ids = pad_sequence_decoder_att_mask(answer_ids, data_args.max_labels_length, logger=None)


        labels = query_ids + answer_ids
        decoder_input_ids = [2] + labels[:-1]
        labels = [
                -100 if i < data_args.max_query_length else (l if l != pad_token_id else -100)
                for i, l in enumerate(labels)
            ]
        
        if is_inference:
            decoder_input_ids = [2] + query_ids
            decoder_attention_mask = [1] * data_args.max_query_length
            labels = [
                l if l != pad_token_id else -100
                for i, l in enumerate(answer_ids)
            ]



            
    return labels, decoder_input_ids, decoder_attention_mask






