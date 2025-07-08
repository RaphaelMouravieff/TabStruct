import json
import pathlib
from typing import List, Optional, Callable
import pandas as pd
from fuzzingbook.GrammarFuzzer import GrammarFuzzer
from argparse import ArgumentParser
from data_processing.tools.SQLFuzzer.grammars import dummy_grammar



class SQLFuzzer :

    def __init__(
            self,
            grammar : Callable,
            fields : List[str],
            max_nonterminals : int = 10,
            start_symbol : str = '<Query>'
    ) :

        # defining grammar for derivation
        self.grammar = grammar

        if fields is None :
            raise ValueError(f"Using SQLFuzzer as standalone requires to define the fields of the associated table")

        self.fields = fields

        assert max_nonterminals > 0, f"Depth of generation should be greater than zero."
        self.max_nonterminals = max_nonterminals

        self.start_symbole = start_symbol

        # init grammar based fuzzer
        self.fuzzer = GrammarFuzzer(
            grammar = dummy_grammar(),
            start_symbol = start_symbol,
            max_nonterminals = self.max_nonterminals
        )

    def generate_sql_queries_batched(self, n_exemple) :

        return [
            self.fuzzer.fuzz() for _ in range(n_exemple)
        ]

    def generate_sql_query(self, table : Optional[pd.DataFrame] = None) -> str :

        if table is not None :
            # generate new grammar based on input table
            new_grammar = self.grammar(table)

            self.fuzzer.grammar = new_grammar

        return self.fuzzer.fuzz()


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