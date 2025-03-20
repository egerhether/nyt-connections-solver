import torch
import random 
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.solver import Solver
from scripts.glove import GloVe
from scripts.puzzle_utils import Puzzle
from scripts.neural.predictor import Predictor
from scripts.neural.evaluator import NeuralEvaluator
from scripts.neural.dataset import ConnectionsDataset

# the general idea here is to recreate the real scenario of playing the game
# namely: 
# (during training) the neural net will receive a concat vector of embeddings of words in a group
# and its aim is to predict whether it is a group or not
# a special dataset is created for that purpose.
# then (at inference) [find alg for choosing initial groups] 
# and once a group has been determined, submit it by using puzzle.check_if_group
# like a real person playing the game. if the submitted group is wrong, lose a "life"
# and try again.
        
class NeuralSolver(Solver):

    def __init__(self, puzzle: Puzzle, evaluator: NeuralEvaluator, dataset: ConnectionsDataset, epochs: int, glove: GloVe):

        self.glove = glove
        self.predictor = Predictor() # part of the solver responsible for finding groups
        self.evaluator = evaluator # part of the solver responsible for evaluating found groups
        self.puzzle = puzzle # puzzle for testing its hypotheses before final prediction
        self.dataset = dataset # dataset for training 
        self.epochs = epochs

    def solve_puzzle(self, words: list, puzzle_id: int):
        '''
        Simulates a game of NYT Connections being played by two actors: predictor and evaluator.
        Predictor proposes groups to create, evaluator judges predictor's propositions 
        determining if the group is to be submitted or not. The two actors have 4 lives, just like
        a player of NYT Connections.
        '''
        
        lives = 4 # lives for our play simulation
        self.clusters = np.ones(16) * -1
        current_group_id = 0

        while lives != 0:
           
            group_attmpt = self.predictor.find_group(words) # TODO: this has to take AND output list of strings
            embedded_group = self.embed_words(group_attmpt)
            is_group = self.evaluator(embedded_group)

            if is_group < 0.5: 
               ## not a group -> find another
               continue
               
            is_sol = self.puzzle.check_if_group(puzzle_id, group_attmpt)
            if not is_sol:
                lives -= 1
                continue

            group_word_idxs = self.get_word_idxs(group_attmpt)
            self.clusters[group_word_idxs] = current_group_id
            current_group_id += 1

            # remove the group words for next finding steps
            words = [word for word in words if word not in group_attmpt]

        groups_found = len(np.unique(self.clusters)) 
        if groups_found != 4:
            # if we didn't find a solution 
            idxs_remaining = np.where(self.clusters == -1)
            idxs_to_distribute = [i % (5 - groups_found) for i in range((groups_found - 1) * 4)]
            random.shuffle(idxs_to_distribute)
            print(idxs_to_distribute) # delete later            
            self.clusters[idxs_remaining] = idxs_to_distribute


        self.clusters = self.clusters.tolist()


    def train_evaluator(self):
        '''
        Performs training of the evaluator network.
        '''
        
        train_data = DataLoader(self.dataset, batch_size = 16, shuffle = True)
        loss_module = nn.BCELoss()
        optimizer = Adam(self.evaluator.parameters(), lr = 1e-3)

        epoch = 0

        for epoch in tqdm(range(self.epochs), desc = f"Training evaluator network"):

            for x, y in train_data:
                optimizer.zero_grad()            
                out = self.evaluator(x)
                out = out.squeeze(1)
                loss = loss_module(out, y.float())
                loss.backward() 
                optimizer.step()


    def embed_words(self, words: list):
        
        embeddings = self.glove.embed_puzzle_words(words)

        return embeddings.flatten()
