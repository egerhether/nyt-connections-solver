import csv
import numpy as np
from tqdm import tqdm

class GloVe:

    def __init__(self, filepath):

        with open(filepath) as embeddings:
            
            self.vocab = {}
            for line in tqdm(embeddings, "Creating GloVe lookup map"):
                line = line.split()
                self.vocab[line[0]] = np.array(line[1:], dtype = np.float32)

            embeddings.close()

        
    def get_embedding(self, word):
        '''
        Args:
            word (str): word we want the embedding of
        Returns:
            embedding (ndarray): embedding of the word
        '''
        return self.vocab[word]
    
    def embed_puzzle_words(self, words):

        embeddings = [self.vocab[word.lower()] for word in words]
        embeddings = np.asarray(embeddings)

        return embeddings