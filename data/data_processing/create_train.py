import argparse
import pathlib
import sys
import torch
import json
import random 


# Define the path for the src
src = pathlib.Path(__file__).parents[1]
sys.path.append(str(src))

from data_processing.sql_dataset.SQLDataset import SQLDataset
from data_processing.sql_dataset.SQLDataset_utils import create_deterministic_transition, get_transition_matrix


def main(args: argparse.Namespace) -> None:


    observations = []

    transition_matrix = None
    if args.S is not None:
        vocab_size = (args.numeric_range[1] - args.numeric_range[0])+1
        deterministic_transition =  create_deterministic_transition(vocab_size)
        print("Deterministic Transition Matrix:")
        print(deterministic_transition)
        transition_matrix = get_transition_matrix(args.S, vocab_size, deterministic_transition)
    
    for counter in range(args.n_samples):

        n_columns = random.randint(args.min_column, args.max_column)
        n_rows = random.randint(args.min_row, args.max_row)
        print(f"Step : {counter}/{args.n_samples}, n_column={n_columns}, n_rows={n_rows}")


        if args.grammar != "ALL":
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
                transition_matrix=transition_matrix,
            )
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
                transition_matrix=transition_matrix,
            )



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




    # Save to JSON file
    with open(args.output_path, "w") as json_file:
        json.dump(observations, json_file, indent=4)

    print(f"Saved observations to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--grammar", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--n_samples", type=int)
    parser.add_argument("--numeric_range", type=int, nargs=2, default=(0, 999))
    parser.add_argument("--max_column", type=int)
    parser.add_argument("--max_row", type=int)
    parser.add_argument("--min_column", type=int)
    parser.add_argument("--min_row", type=int)
    parser.add_argument("--S", type=float, default=None)

    args = parser.parse_args()

    # Print the parsed arguments to check if everything is correct
    print(args)

    main(args)