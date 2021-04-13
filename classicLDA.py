from LDA import LDA, get_file
from gensim.models.ldamulticore import LdaMulticore
import csv
import os
import time
from multiprocessing import Pool


class ClassicLDA(LDA):
    def __init__(self, *args):
        if len(args) > 1: super().__init__(args)
        else: super().__init__()
        self.name = "classic"
        self.model = None

    # use model as field for methods designed for classic LDA model
    def set_model(self, model):
        self.model = model

    def create_model(self, topics=20, workers=1):
        return LdaMulticore(corpus=self.corpus, num_topics=topics, id2word=self.words, workers=workers)

    def load_lda_model(self):
        return get_file("lda-data/lda-model")

    def compute_top_topics(self):
        return self.model.top_topics(corpus=self.corpus, texts=self.lemmatized, dictionary=self.words, coherence='c_v',
                                     processes=2, topn=10)


if __name__ == '__main__':
    model = ClassicLDA()
    model.run_multiple_increasing_topics(8, "rapid", limit=65, start=20)
    # model.run_neural_embedding(11, "50_7_20")
