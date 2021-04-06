from LDA import LDA, get_file
from gensim.models.ldamulticore import LdaMulticore
import csv
import os


class ClassicLDA(LDA):
    def __init__(self):
        super().__init__()
        self.name = "classic"
        self.model = None

    # use model as field for methods designed for classic LDA model
    def set_model(self, model):
        self.model = model

    def create_model(self, topics=20, workers=2):
        return LdaMulticore(corpus=self.corpus, num_topics=topics, id2word=self.words, workers=workers)

    def load_lda_model(self):
        return get_file("lda-data/lda-model")

    def compute_top_topics(self):
        return self.model.top_topics(corpus=self.corpus, texts=self.lemmatized, dictionary=self.words, coherence='c_v',
                                     processes=2, topn=10)


if __name__ == '__main__':
    model = ClassicLDA()
    # model.run_multiple(11, "50_7")
    model.run_neural_embedding(11, "50_7")
