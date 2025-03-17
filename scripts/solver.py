import random 

class Solver:

    def __init__(self):
        pass
    
    def solve_puzzle(self, words: list):
        '''
        Default implementation of solving the puzzle. Creates a list of indices
        of clusters each word is assigned to.
        Args:
            words (list): list of words of the puzzle. Not used in default implementation. 
        '''
        self.clusters = [i % 4 for i in range(16)]
        random.shuffle(self.clusters)

    def build_solution(self, words: list):
        '''
        Default implementation of building the solution for the puzzle. Uses the 
        clusters attribute to create a list of sets of 4 words.
        Returns:
            attemped_sol (list): list of sets of 4 words represeting the solution.
        '''
        
        attempted_sol = [set() for _ in range(4)]
        for idx, cluster_id in enumerate(self.clusters):
            attempted_sol[cluster_id].add(words[idx])

        return attempted_sol