import os.path
import pandas as pd
import numpy as np
import random
import torch
import ast
from torch.utils.data import Dataset
from tqdm import tqdm

from scripts.puzzle_utils import Puzzle
from scripts.glove import GloVe 

class ConnectionsDataset(Dataset):

    def __init__(self, glove: GloVe, puzzle: Puzzle):
        super().__init__()
        self.glove = glove
        self.puzzle = puzzle

        if not os.path.isfile("data/connections_training.npz"):
            self.build_dataset()
        
        self.data = np.load("data/connections_training.npz", allow_pickle = True)
    
    def build_dataset(self):
        '''
        Function for building a dataset for training the neural solver for Connections.
        In short, for each id up to 500, create a correct and an incorrect example.
        Incorrect example is created by taking a correct example and substituting a
        random number between 1 and 3 words in the example with remaining words.
        '''
        dataset = pd.DataFrame(columns = ["x", "y"])

        # 500 is arbitrary number so that we can still test on a decently sized portion of the data later
        for id in tqdm(range(500), desc = "creatiing csv"):
            all_words, _ = self.puzzle.get_puzzle_by_id(id)

            correct_group = self.puzzle.get_one_group(id)
           
            incorrect_group = self.puzzle.get_one_group(id)
            nr_changes = random.randint(1, 3)
            indices_to_change = list(range(4))
            for _ in range(nr_changes):
                substitute_word = random.sample(set(all_words) - set(incorrect_group), 1)[0]
                idx = random.sample(indices_to_change, 1)[0]
                indices_to_change.remove(idx)
                incorrect_group[idx] = substitute_word

            x_right = self.glove.embed_puzzle_words(correct_group).flatten()
            x_wrong = self.glove.embed_puzzle_words(incorrect_group).flatten()

            if x_right.size < 1200 or x_wrong.size < 1200:
                continue

            dataset.loc[2 * id] = [x_right, 1]
            dataset.loc[2 * id + 1] = [x_wrong, 0]

        np.savez("data/connections_training.npz", x = np.array(dataset["x"].tolist()), y = dataset["y"].values, allow_pickle = True)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = torch.from_numpy(self.data["x"][idx])
        y = torch.as_tensor(self.data["y"][idx])
           
        return x, y
