import argparse
import pathlib
import sys
import torch
import os
from datasets import Dataset, DatasetDict


src = pathlib.Path(__file__).parents[1]
sys.path.append(str(src))

from data_processing.sql_dataset.SQLDataset import SQLDataset
from data_processing.sql_dataset.SQLDataset_utils import create_deterministic_transition, get_transition_matrix
import random


def create_dataset(n_rows, n_columns, args, transition_matrix):
    print(f'args.repetition_rate = {args.repetition_rate}')
    # Init SQLDataset

    observations = []

    for _ in range(200):

        if args.grammar not in ["ALL", "COMPO"]:
            dataset = SQLDataset(
                grammar=args.grammar,
                table_name="w",
                batch_size=1,
                max_nonterminals=4,
                return_row_data=True,
                n_samples=1,
                numeric_range=args.numeric_range,
                n_columns=n_columns,
                n_rows=n_rows,
                repetition_rate=args.repetition_rate,
                missing_values=args.missing_values,
                transition_matrix=transition_matrix, )
            
        if args.grammar == "ALL":
            
            grammar = random.choice(["IN", "LIMIT", "SELECT" ,"WHERE", "CONDI1", "CONDI2", "CONDI3", 'NESTEDSELECT'])
            dataset = SQLDataset(
                grammar=grammar,
                table_name="w",
                batch_size=1,
                max_nonterminals=4,
                return_row_data=True,
                n_samples=1,
                numeric_range=args.numeric_range,
                n_columns=n_columns,
                n_rows=n_rows,
                repetition_rate=args.repetition_rate,
                missing_values=args.missing_values,
                transition_matrix=transition_matrix, ) 

        if args.grammar == "COMPO":
            
            grammar = random.choice(["IN_compo", "SELECT_compo", "CONDI1_compo", "CONDI2_compo"])
            dataset = SQLDataset(
                grammar=grammar,
                table_name="w",
                batch_size=1,
                max_nonterminals=4,
                return_row_data=True,
                n_samples=1,
                numeric_range=args.numeric_range,
                n_columns=n_columns,
                n_rows=n_rows,
                repetition_rate=args.repetition_rate,
                missing_values=args.missing_values,
                transition_matrix=transition_matrix, ) 

        dataloader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=None)

        for _, batch in dataloader:
            observation = {
                "table": {
                    "rows": batch['tables'][0].values.tolist(),
                    "header": batch['tables'][0].columns.tolist()
                },
                "question": batch['queries'][0],
                "answers": batch['answers'][0]
            }
            print(observation)
            print('\n'*3)
            observations.append(observation)

    return observations

def main(args: argparse.Namespace) -> None:


    vocab_size = (args.numeric_range[1] - args.numeric_range[0])+1
    deterministic_transition =  create_deterministic_transition(vocab_size)
    print("Deterministic Transition Matrix:")
    print(deterministic_transition)
    transition_matrix = get_transition_matrix(args.S, vocab_size, deterministic_transition)

    all_datasets = {}
    
    min_ = 4
    max_ = 13 
    if args.debug == True:
        max_ = 5
    for row in range(min_, max_):
        for col in range(min_, max_):
            print(f'Start Creating dataset for row={row} and col={col}')
            observations = create_dataset(row, col, args, transition_matrix)
            dataset_name = f"test_{row}_{col}"
            dataset = Dataset.from_dict({
                "table": [obs["table"] for obs in observations],
                "question": [obs["question"] for obs in observations],
                "answers": [obs["answers"] for obs in observations]
            })
            all_datasets[dataset_name] = dataset
    
    dataset_dict = DatasetDict(all_datasets)
    dataset_dict.save_to_disk(args.output_path)

    print(f"Saved datasets to {args.output_path}")


if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument("--grammar", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--numeric_range", type=int, nargs=2, default=(0, 999))
    parser.add_argument("--repetition_rate", type=float, default=0.0)
    parser.add_argument("--missing_values", type=int, choices=[0, 1])
    parser.add_argument("--S", type=float, default=None)
    parser.add_argument("--debug", default=0, type=int, choices=[0, 1])




    args = parser.parse_args()

    # Print the parsed arguments to check if everything is correct
    print(args)

    main(args)