from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import numpy as np

from scripts.solver import Solver
from scripts.glove import GloVe

class KMeansSolver(Solver):

    def __init__(self, glove: GloVe):
        super().__init__()
        self.kmeans = KMeans(n_clusters = 4)
        self.glove = glove

    def solve_puzzle(self, words):
        '''
        Solves the puzzle using fixed size k-means algorithm.
        Args:
            words (list): list of words of the puzzle
        '''
        embeddings = self.glove.embed_puzzle_words(words)
        self.kmeans.fit(embeddings)

        centers = self.kmeans.cluster_centers_
        centers = centers.reshape(-1, 1, embeddings.shape[-1]).repeat(4, 1).reshape(-1, embeddings.shape[-1])

        distance_matrix = cdist(embeddings, centers)
        self.clusters = linear_sum_assignment(distance_matrix)[1] // 4 