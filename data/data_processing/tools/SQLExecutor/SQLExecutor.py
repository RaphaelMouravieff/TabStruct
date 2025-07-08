import pathlib
from typing import Dict, Union, Any, Tuple, List
from fuzzingbook.GrammarFuzzer import GrammarFuzzer
from argparse import ArgumentParser
import sqlite3
import pandas as pd


class SQLExecutor :
    """
    Object allowing sql query execution on pandas dataframes and post-processing results
    """

    def __init__(
            self,
            table_name : str,
    ) :
        """

        :param table_name:
        """

        self.table_name = table_name

        # init sqlite server
        self.conn = sqlite3.connect(':memory:')  # Use ':memory:' to create a database in RAM

    def post_process(self, result_df) :

        # convert to list
        result = result_df.values.tolist()

        # flatten
        result = [ str(x[0]) for x in result ]

        # convert to string
        result = ", ".join(result)

        return result

    def process(self, input_table : pd.DataFrame, query : str) -> str :
        """

        :param input_table:
        :param query:
        :return:
        """

        # passing dataframe to sql
        input_table.to_sql(self.table_name, self.conn, index=False, if_exists='replace')

        # executing sql
        result_df = pd.read_sql_query(query, self.conn)

        # post process results
        result = self.post_process(result_df)

        return result

    def process_batch(self, samples : List[Dict]) -> None :
        """
        Update every sample in a batch with SQL results
        :param samples: List of input sample
        """

        for sample in samples :
            sample['answers'] = self.process(sample['tables'], sample['queries'])

    @staticmethod
    def add_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """

        :param parent_parser:
        :return:
        """

        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--grammar", type=pathlib.Path)
        parser.add_argument("--fields", type=str, nargs = "+",  default=None)
        parser.add_argument("--max_nonterminals", type=int, default=5)
        parser.add_argument("--start_symbol", type=str, default= '<Query>')


        return parser