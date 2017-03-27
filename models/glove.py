import numpy as np


class Glove(object):
    def __init__(self, str_table):
        embeddings_index = {}
        dim = None
        for line in str_table.split('\n'):
            line = line.strip()

            if line == "":
                continue

            # if (len(embeddings_index) > 500):
            #    break

            values = line.split()

            word = values[0]
            if dim is None:
                dim = len(values) - 1

            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

        self._embedding_index = embeddings_index
        self.dim = dim

        print "Glove index load dim " + str(dim) + " and " + str(len(embeddings_index)) + " words "

    def reduce_dim(self, dim):
        self.dim = dim

    def get_embedding_as_list(self, word):
        embedding = self.get_embedding(word)
        return embedding.tolist()

    def get_embedding(self, word):
        if not word in self._embedding_index:
            return np.array([0.0] * self.dim)

        return self._embedding_index[word][0:self.dim]
