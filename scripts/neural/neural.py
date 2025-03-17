import torch
import torch.nn as nn
from torch.optim import Adam

from scripts.solver import Solver

# the general idea here is to recreate the real scenario of playing the game
# namely: 
# (during training) the neural net will receive a concat vector of embeddings of words in a group
# and its aim is to predict whether it is a group or not
# a special dataset is created for that purpose.
# then (at inference) [find alg for choosing initial groups] 
# and once a group has been determined, submit it by using puzzle.check_if_group
# like a real person playing the game. if the submitted group is wrong, lose a "life"
# and try again.
        
        
