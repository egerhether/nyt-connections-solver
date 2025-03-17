import numpy as np
from tqdm import tqdm

# TODO: improve generation of embeddings for complex phrases
# sum the composite embeddings ???
class GloVe:

    def __init__(self, filepath: str):

        with open(filepath) as embeddings:
            
            self.vocab = {}
            for line in tqdm(embeddings, "Creating GloVe lookup map"):
                line = line.split()
                self.vocab[line[0]] = np.array(line[1:], dtype = np.float32)

            embeddings.close()

        
    def get_embedding(self, word: str):
        '''
        Args:
            word (str): word we want the embedding of
        Returns:
            embedding (ndarray): embedding of the word
        '''
        return self.vocab[word]
    
    def embed_puzzle_words(self, words: list):
        '''
        Args:
            words (list): list of words to return the embeddings of
        Returns:
            embeddings (np.ndarray): numpy array of all the embeddings
        '''
        embeddings = [self.vocab[word.lower()] for word in words]
        embeddings = np.asarray(embeddings)

        return embeddings
