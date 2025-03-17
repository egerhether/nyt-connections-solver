import json
import random
import re

class Puzzle:

    def __init__(self, filepath = "NYT-Connections-Answers/connections.json"):

        with open(filepath) as file:
            self.puzzles = json.load(file)
            file.close()

    def get_random_puzzle(self):
        '''
        Returns:
          words (list): a randomised list of words all belonging to a random puzzle from the dataset
          id (int): id of the puzzle chosen, for checking purposes later
        '''

        puzzle_id = random.randint(1, 638)

        return self.get_puzzle_by_id(puzzle_id)

    def check_solution(self, words: list, id: int):
        '''
        Args:
            words (list): a list of sets of 4 strings containing attempted solution
            id (int): id of the puzzle of the attempted solution
        Returns:
            is_solution (bool): true if it is a solution
            num_correct (int): number of total correct words in the best case interpretation of the attempted solution
        '''
        puzzle = self.puzzles[id]['answers']
        is_solution = False
        num_correct = 0
        
        # keeps track of groups already checked
        indices = [0, 1, 2, 3]

        for level in puzzle:
            group = set(level['members'])

            longest_intersection = 0
            for i in indices:
                intersection = group.intersection(words[i])

                if len(intersection) > longest_intersection:
                    longest_intersection = len(intersection)
                    indices.remove(i)
                
            num_correct += longest_intersection

        is_solution = (num_correct == 16)
              
        return is_solution, num_correct
    
    def get_puzzle_by_id(self, puzzle_id: int):
        '''
        Args:
            puzzle_id (int): id of the puzzle to return
        Returns:
            new_words (list): words of the puzzle in random order
            puzzle_id (int): id of the puzzle returned 
        '''

        puzzle = self.puzzles[puzzle_id]['answers']

        words = []
        for level in puzzle:
            group = level['members']
            words.extend(group)

        random.shuffle(words)

        new_words = [re.sub(r"\s+", "", word, flags=re.UNICODE) for word in words]
        return new_words, puzzle_id

    def check_if_group(self, puzzle_id: int, group: list):
        '''
        Args:
            puzzle_id (int): id of the puzzle to check
            group (list): list of words to check if they are part of the solution
        Returns:
            is_sol (bool): boolean checking if group matches one of the solution groups 
        '''

        puzzle = self.puzzles[puzzle_id]['answers']
        for level in puzzle:
            if group == level['members']:
                return True

        return False
