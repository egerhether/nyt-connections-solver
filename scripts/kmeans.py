from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import numpy as np

from solver import Solver

class KMeansSolver(Solver):

    def __init__(self):
        
        self.kmeans = KMeans(n_clusters = 4)

    def solve_puzzle(self, embeddings):
        n_clusters = int(np.ceil(len(embeddings) / 4))
        self.kmeans.fit(embeddings)

        centers = self.kmeans.cluster_centers_
        centers = centers.reshape(-1, 1, embeddings.shape[-1]).repeat(4, 1).reshape(-1, embeddings.shape[-1])

        distance_matrix = cdist(embeddings, centers)
        self.clusters = linear_sum_assignment(distance_matrix)[1] // 4
        
    
    def build_solution(self, words):

        attempted_sol = [set() for _ in range(4)]

        for idx, cluster_id in enumerate(self.clusters):
            attempted_sol[cluster_id].add(words[idx])

        return attempted_sol