from copy import deepcopy
from wikisql_utils import _TYPE_CONVERTER, retrieve_wikisql_query_answer_tapas
from datasets import load_dataset
from transformers import BartTokenizer
from typing import Any, List

def preprocess_tableqa_function(examples):

    def _convert_table_types(_table):
        """Runs the type converter over the table cells."""
        ret_table = deepcopy(_table)
        types = ret_table["types"]
        ret_table["real_rows"] = ret_table["rows"]
        typed_rows = []
        for row in ret_table["rows"]:
            typed_row = []
            for column, cell_value in enumerate(row):
                typed_row.append(_TYPE_CONVERTER[types[column]](cell_value))
            typed_rows.append(typed_row)
        ret_table["rows"] = typed_rows
        return ret_table

    example_tables = examples["table"]
    example_sqls = examples["sql"]

    answers = []
    for example_sql, example_table in zip(example_sqls, example_tables):
        tapas_table = _convert_table_types(example_table)
        answer_list: List[str] = retrieve_wikisql_query_answer_tapas(tapas_table, example_sql)
        answers.append(answer_list)

    return {"answers" : answers}


if __name__ == "__main__":

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    dataset = load_dataset('wikisql')
    print('Start collecting answers..')
    dataset = dataset.map(preprocess_tableqa_function, batched=True)
    

    output_dir = "../dataset_and_json/train/wikisql"
    print(f"Save dataset to {output_dir}")
    dataset.save_to_disk(output_dir)


