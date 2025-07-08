import pandas as pd


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)



def get_labels(answers, data_args, tokenizer, padding, pad_token_id):

    answers_text = ", ".join(answers) if data_args.is_wtq else answers
    labels = tokenizer(text_target=answers_text, max_length=data_args.max_target_length, padding=padding, truncation=True)

    labels = [(l if l != pad_token_id else -100) for l in labels["input_ids"]]

            
    return labels






