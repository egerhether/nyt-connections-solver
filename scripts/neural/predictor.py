from scripts.glove import GloVe
import numpy as np

class Predictor:

    def __init__(self, glove: GloVe):
        self.glove = glove

    def find_group(self, words: list):
        '''
        Finds a candidate group from a list of words by optimizing total in-group
        similarity. 
        Args:
            words (list): list of words to find a group in
        Returns:
            group_attempt (list): list of words making up a group
            iterations (int): number of iterations taken to find the group
        '''
        embeddings = self.glove.embed_puzzle_words(words)
        norms = np.linalg.norm(embeddings, axis = 1, keepdims = True) 
        norm_embeddings = embeddings / norms
        sim_matrix = np.dot(norm_embeddings, norm_embeddings.T) 
        
        # for a random word find group that maximises similarity
        group_idxs, iterations = self.optimize_similarity(sim_matrix)
        
        group_attempt = [words[idx] for idx in group_idxs] 
        
        return group_attempt, iterations 


    def optimize_similarity(self, sim_matrix: np.ndarray):
        '''
        Performs an iterative algorithm maximising similarity within group.
        Args:
            sim_matrix (np.ndarray): similarity matrix of all words
        Returns:
            current_words (np.ndarray): indexes of candidate group
            iter (int): number of iterations taken to converge
        '''
        indexes = range(16)
        initial_words = np.random.choice(indexes, 4, replace = False)
        tol = 1e-5
        iter = 0
        delta = 1

        while delta > tol:
            initial_sim = sim_matrix[np.ix_(initial_words, initial_words)]
            old_size = np.linalg.norm(initial_sim)
            
            # find the word with lowest total similarirt with all other words
            # in the current group
            group_similarities = [np.sum(initial_sim[i]) for i in range(4)]
            word_to_remove = np.argmin(group_similarities)
            
            # find a candidate new word as highest similarity with remaining words
            current_words = np.delete(initial_words, word_to_remove)
            total_similarities = [np.sum(sim_matrix[np.ix_(np.array([i]), current_words)]) for i in range(16)]
            word_to_add = np.argmax(total_similarities)

            #include new word
            current_words = np.append(current_words, word_to_add) 
            
            current_sim = sim_matrix[np.ix_(current_words, current_words)] 
            current_size = np.linalg.norm(current_sim)
            delta = np.abs(current_size - old_size)
            iter += 1
            initial_words = current_words

        return current_words, iter + 1 
        
    
