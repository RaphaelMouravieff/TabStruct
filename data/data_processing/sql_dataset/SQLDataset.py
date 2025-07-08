import pathlib
from typing import Tuple, List, Optional, Union, Type, Dict
from argparse import ArgumentParser, Namespace
from torch.utils.data.dataset import IterableDataset

from data_processing.tools.MatrixGenerator.MatrixGenerator import MatrixGenerator
from data_processing.tools.SQLFuzzer.SQLFuzzer import SQLFuzzer
from data_processing.tools.SQLFuzzer.grammars import grammars, Grammar
from data_processing.tools.SQLExecutor.SQLExecutor import SQLExecutor

import numpy as np

class SQLDataset(IterableDataset) :
 
    def __init__(
            self,
            grammar: str,
            table_name : str,
            start_symbol : str = "<Query>",
            n_samples :  int = 10000,
            batch_size : int = 1,
            n_columns: int = 8,
            n_rows: int = 8,
            numeric_range: Tuple[int, int] = (0, 999),
            columns_names: Optional[List[str]] = None,
            auto_column_permute: bool = True,
            max_nonterminals: int = 3,
            random_seed: int = 42,
            repetition_rate: float = 0.0, 
            missing_values: bool = False,
            transition_matrix: np.ndarray = None,
            return_row_data : bool = False,

    ) :
        """

        :param grammar:
        :param table_name:
        :param n_columns:
        :param n_rows:
        :param numeric_range:
        :param columns_names:
        :param auto_column_permute:
        :param max_nonterminals:
        :param start_symbol:
        :param random_seed:
        """
        super().__init__()

        


        assert n_samples > 0, f"Number of samples should be greater than zero."
        self.n_samples = n_samples

        # init Matrix Generator
        self.matrix_generator = MatrixGenerator(
            n_columns=n_columns,
            n_rows=n_rows,
            numeric_range=numeric_range,
            columns_names=columns_names,
            dataset_size=1,
            auto_column_permute=auto_column_permute,
            random_seed=random_seed,
            repetition_rate=repetition_rate,
            missing_values=missing_values,
            transition_matrix=transition_matrix,
        )

        # init SQLFuzzer
        self.SQL_query_generator = SQLFuzzer(
            grammar = grammars[grammar],
            fields=self.matrix_generator.columns_names,
            max_nonterminals=max_nonterminals,
            start_symbol=start_symbol
        )

        # init SQLExecutor
        self.SQL_executor = SQLExecutor(
            table_name=table_name
        )

        self.return_row_data = return_row_data

        self.use_transition_matrix = 1 if transition_matrix is not None else 0

    def __len__(self):
        return self.n_samples


    def __iter__(self):

        for _ in range(self.n_samples) :

            batch = {
                "tables" : [],
                "queries" : [],
                "answers" : []
            }

            # generating matrix
            if self.use_transition_matrix:
                print("Generating matrices using the transition matrix")
                matrix = self.matrix_generator.generate_matrix_transi()
            else:
                print("Generating random matrices without the transition matrix")
                matrix = self.matrix_generator.generate_matrix()
            batch['tables'].append(matrix)

            # generating sql
            query = self.SQL_query_generator.generate_sql_query(matrix)
            batch['queries'].append(query)

            # processing sql results
            results = self.SQL_executor.process(matrix, query)
            batch['answers'].append(results)
            encoded_input = None

        if self.return_row_data :
            yield encoded_input, batch
        else :
            yield encoded_input

    @staticmethod
    def add_data_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """
        Add data module specific args. To be overrided
        :param parent_parser: main parser
        :return: main  parser updated
        """

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('SQLQA Dataset')


        group.add_argument("--grammar", type = str, default = "grammar_WHERE", choices = grammars.keys())
        group.add_argument("--table_name", type = str, default = "w")
        group.add_argument("--start_symbol", type = str, default = "<Query>")
        group.add_argument("--n_columns", type = int, default = 8)
        group.add_argument("--n_rows", type = int, default = 8)
        group.add_argument("--numeric_range", nargs = "+",  type = int, default = (0, 999))
        group.add_argument("--columns_names", nargs = "+",  type = str, default = None)
        group.add_argument("--no_auto_column_permute", action="store_false",  default = True)
        group.add_argument("--max_nonterminals", type = int, default = 4)

        return parser