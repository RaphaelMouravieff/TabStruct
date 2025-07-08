

import os
import pickle
import numpy as np

def create_deterministic_transition(vocab_size, deterministic_transition_dir='../dataset_and_json/deterministic_transition/'):
    """
    Creates a deterministic transition matrix or loads an existing one if it already exists.
    
    :param vocab_size: The size of the vocabulary (number of states).
    :param deterministic_transition_dir: Directory where transition matrices are stored.
    :return: The deterministic transition matrix.
    """
    
    # Create the directory if it does not exist
    if not os.path.exists(deterministic_transition_dir):
        os.makedirs(deterministic_transition_dir)
    
    # Build the expected file name based on the vocab_size
    expected_file_name = f"deterministic_transition_vocab_size_{vocab_size}.pkl"
    expected_file_path = os.path.join(deterministic_transition_dir, expected_file_name)
    
    # Check if the file exists
    if os.path.exists(expected_file_path):
        print(f"Found existing deterministic transition matrix: {expected_file_name}")
        # Load the deterministic transition matrix from the file
        with open(expected_file_path, 'rb') as f:
            deterministic_transition = pickle.load(f)
    else:
        # Generate a new random deterministic transition matrix
        deterministic_transition = np.zeros((vocab_size, vocab_size))
        for i in range(vocab_size):
            # Select a random state (excluding the current state)
            random_state = np.random.choice([x for x in range(vocab_size) if x != i])
            deterministic_transition[i][random_state] = 1
        
        # Save the new deterministic transition matrix
        with open(expected_file_path, 'wb') as f:
            pickle.dump(deterministic_transition, f)
        print(f"Created and saved new deterministic transition matrix: {expected_file_name}")
    
    return deterministic_transition

def get_transition_matrix(S, vocab_size, deterministic_transition):
    """
    Creates the full transition matrix by blending the deterministic transitions 
    with uniform transitions, based on the similarity parameter S.
    """
    # Create uniform transition matrix
    if S is not None:
        uniform_transition = np.ones((vocab_size, vocab_size)) / vocab_size

        if S == 1:
            # For maximum similarity, use the deterministic transition matrix
            transition_matrix = deterministic_transition
        else:
            # For S < 1, blend deterministic and uniform transitions
            transition_matrix = S * deterministic_transition + (1 - S) * uniform_transition
        
        return transition_matrix
    else : 
        return None