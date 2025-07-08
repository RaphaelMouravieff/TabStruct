from typing import Tuple, List, Optional
import random
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle
import os



class MatrixGenerator :
    """
    Class generating matrixces filled with random number as pandas DataFrames
    """

    def __init__(
            self,
            n_columns : int = 8,
            n_rows : int = 8,
            numeric_range : Tuple[int, int] = (0, 999),
            columns_names : Optional[List[str]] = None,
            dataset_size : int = 1000,
            auto_column_permute : bool = True,
            random_seed : int  = 42,
            repetition_rate: float = 0.0, 
            missing_values: bool = False,
            transition_matrix: np.ndarray = None):

        """
        :param n_columns: number of columns
        :param n_rows:  number of rows
        :param numeric_range: range of values adopted within the dataFrame
        :param columns_names: names of each columns, default is C0, C1, ..., Cn
        :param dataset_size:
        """

        assert n_columns > 0, f"The number of columns should be strictly positive"
        self.n_columns = n_columns

        assert n_rows > 0, f"The number of rows should be strictly positive"
        self.n_rows = n_rows

        self.numeric_range = numeric_range

        self.columns_names = [ f"C{i}" for i in range(14) ]

        assert dataset_size > 0, f"Dataset should be at least of size one"
        self.dataset_size = dataset_size

        self.auto_column_permute = auto_column_permute

        self.repetition_rate = repetition_rate
        self.missing_values = missing_values

        self.transition_matrix = transition_matrix

    def generate_matrix(self) -> pd.DataFrame:
        """
        Generate a matrix filled with random numbers, with control over repetition of vocabulary.
        :param R: Repetition parameter (0 to 100) where 0 means no repetition and 100 means full repetition.
        :return: DataFrame representing the out matrix
        """
        # Create the vocabulary
        R = self.repetition_rate / 100
        M = self.missing_values
        vocabulary = [str(i) for i in range(self.numeric_range[0], self.numeric_range[1] + 1)]

        # Decide on the repeated word
        repeated_word = str(random.randint(self.numeric_range[0], self.numeric_range[1])) if M == False else ""

        # Initialize the matrix
        matrix = []
        for _ in range(self.n_rows):
            row = []
            for _ in range(self.n_columns):
                if random.random() < R:
                    row.append(repeated_word)
                else:
                    row.append(random.choice(vocabulary))
            matrix.append(row)

        # Permute columns if required
        if self.auto_column_permute:
            random.shuffle(self.columns_names)

        # Create and return the DataFrame
        return pd.DataFrame(matrix, columns=self.columns_names[:self.n_columns])

    
    def generate_matrix_transi(self) -> pd.DataFrame:
        """
        Generate a matrix using a transition matrix to control transitions between vocabulary words.
        :return: DataFrame representing the matrix generated using transition probabilities
        """
        # Create the vocabulary
        vocabulary = [str(i) for i in range(self.numeric_range[0], self.numeric_range[1] + 1)]
        vocab_size = len(vocabulary)
        
        # Initialize matrix
        matrix = []
        
        for _ in range(self.n_rows):
            row = []
            # Start with a random word from the vocabulary
            current_word = random.choice(vocabulary)
            row.append(current_word)
            
            # Use the transition matrix for subsequent words
            for _ in range(self.n_columns - 1):
                current_word_idx = vocabulary.index(current_word)
                # Sample the next word based on the transition matrix probabilities
                next_word_idx = np.random.choice(range(vocab_size), p=self.transition_matrix[current_word_idx])
                next_word = vocabulary[next_word_idx]
                row.append(next_word)
                current_word = next_word
            
            matrix.append(row)
        
        # Permute columns if required
        if self.auto_column_permute:
            random.shuffle(self.columns_names)
        
        # Create and return the DataFrame
        return pd.DataFrame(matrix, columns=self.columns_names[:self.n_columns])

    @staticmethod
    def add_specific_args(parent_parser: ArgumentParser) -> ArgumentParser :
        """
        Add matrixGenerator specific args
        :param parent_parser: main parser
        :return: updated parser
        """

        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--n_columns", type=int, default=8)
        parser.add_argument("--n_rows", type=int, default=8)
        parser.add_argument("--random_seed", type=int, default=42)
        parser.add_argument("--dataset_size", type=int, default=10000)
        parser.add_argument("--numeric_range", nargs = "+", type=int, default=[0, 999])
        parser.add_argument("--columns_names", nargs = "+", type=str, default=None)
        parser.add_argument("--no_auto_column_permute", action='store_false', default=True)

        return parser
