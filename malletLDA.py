from LDA import LDA, get_file
from gensim.models.wrappers.ldamallet import LdaMallet
import csv

# path of mallet file, place in root directory
mallet_path = './mallet-2.0.8/bin/mallet'


class MalletLDA(LDA):
    def __init__(self):
        super().__init__()
        self.name = "mallet"

    def create_model(self, topics=20, workers=2):
        return LdaMallet(mallet_path, corpus=self.corpus, num_topics=topics, id2word=self.words,
                         workers=workers)

    def load_lda_model(self):
        return get_file("lda-data/mallet-model")


if __name__ == '__main__':
    model = MalletLDA()
    model.run_neural_embedding(11, "50_7")
