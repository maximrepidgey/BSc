import sys

from LDA import LDA, get_file
from gensim.models.ldamodel import LdaModel
import csv
import os
import time
from multiprocessing import Pool


class ClassicLDA(LDA):
    def __init__(self, path, *args):
        if len(args) > 1:
            super().__init__(path, args)
        else:
            super().__init__(path)
        self.name = "classic"
        self.model = None
        self.__eta = None

    # use model as field for methods designed for classic LDA model
    def set_model(self, model):
        self.model = model

    def set_eta(self, value):
        self.__eta = value

    def create_model(self, topics=20, workers=1):
        return LdaModel(corpus=self.corpus, num_topics=topics, id2word=self.words, eta=self.__eta)

    def load_lda_model(self):
        return get_file("lda-data/classic")

    def compute_top_topics(self):
        return self.model.top_topics(corpus=self.corpus, texts=self.lemmatized, dictionary=self.words, coherence='c_v',
                                     processes=2, topn=10)


if __name__ == '__main__':
    from NLP import nlp
    from pprint import pprint

    nlp("59_3", 30, passages=True, albert=True)

    classic = ClassicLDA("test/classic-test/")
    # classic.set_eta(5)
    model = classic.create_model_and_save(3)
    classic.df_topic_sent_keywords_print()
    pprint(model.show_topics())

    sys.exit(0)
    nlp("31_6", 30, passages=True, albert=True)
    classic = ClassicLDA("test/classic-test/")
    model = classic.create_model_and_save(3)
    model = classic.load_lda_model()
    print(classic.passages)
    print(len(classic.passages))
    # sys.exit(0)
    tmp = model[classic.corpus]
    for i, row in enumerate(tmp):
        print(i, row)  # i is doc_num and row is a list of tuples (topic_num, distribution) ...
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        print(row)
        for j, (topic_num, prop_topic) in enumerate(row):
            print(j)
            print(topic_num)
            print(prop_topic)
        break

    # pprint(model.top_topics(classic.corpus, classic.lemmatized, classic.words, topn=10))
    classic.df_topic_sent_keywords_print()
    pprint(model.show_topics())
