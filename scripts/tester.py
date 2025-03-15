from scripts.solver import Solver
from scripts.puzzle_utils import Puzzle
from tqdm import tqdm

class MethodTester:

    def __init__(self, solver: Solver):
        
        self.solver = solver
        self.puzzle = Puzzle()

    def test_solver(self):
        '''
        Performs a test of the solver by attempting to solve all puzzles.
        '''
        
        results = {"is_sol": [], "num_corr": []}

        for id in tqdm(range(638)):
            try: 
                random_words, id = self.puzzle.get_puzzle_by_id(id)
                self.solver.solve_puzzle(random_words)
                attempted_sol = self.solver.build_solution(random_words)
                is_sol, num_corr = self.puzzle.check_solution(attempted_sol, id)
                results["is_sol"].append(is_sol)
                results["num_corr"].append(num_corr)
            except:
                pass       

        return results