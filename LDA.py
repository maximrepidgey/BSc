import pickle
import gensim
from gensim.models import CoherenceModel
from pprint import pprint
import pandas as pd
import os

import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
from multiprocessing import Pool
os.environ.update({'MALLET_HOME': r'./mallet-2.0.8/bin/mallet'})
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# path of mallet file, place in root directory
mallet_path = './mallet-2.0.8/bin/mallet'


def init_lemmatized():
    return get_file("lemmatized")


def init_corpus():
    return get_file("corpus")


def init_words():
    return get_file("words")


def init_lda_model():
    return get_file("lda-model")


def init_mallet_model():
    return get_file("mallet-model")


def get_file(filename):
    infile = open(filename, 'rb')
    tmp = pickle.load(infile)
    infile.close()
    return tmp


class Mallet:
    def __init__(self):
        self.lemmatized = init_lemmatized()
        self.corpus = init_corpus()
        self.words = init_words()

    def run_multiple_mallet_and_print(self, limit=20, start=2, step=2):
        model_list, coherence_values = self.compute_coherence_values(limit, start, step)
        # save values
        out_lemmatized = open('values_mallet', 'wb')
        pickle.dump({"model": model_list, "values": coherence_values}, out_lemmatized)
        out_lemmatized.close()
        # Show graph
        x = range(start, limit, step)
        plt.plot(x, coherence_values)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.legend("coherence_values", loc='best')
        plt.show()

    def load_multiple_mallet(self):
        return get_file("values_mallet")

    def compute_coherence_values(self, limit=20, start=2, step=3):
        """
        Compute c_v coherence for various number of topics

        Parameters:
        ----------
        dictionary : Gensim dictionary
        corpus : Gensim corpus
        texts : List of input texts
        limit : Max num of topics

        Returns:
        -------
        model_list : List of LDA topic models
        coherence_values : Coherence values corresponding to the LDA model with respective number of topics
        """
        coherence_values = []
        model_list = []
        for num_topics in range(start, limit, step):
            model = self.create_mallet_model(num_topics)
            model_list.append(model)

            coherence_model = self.compute_coherence(model)
            coherence_values.append(coherence_model.get_coherence())
            print("done with {} topics".format(num_topics))

        return model_list, coherence_values

    def create_mallet_model(self, topics=20):
        return gensim.models.wrappers.LdaMallet(mallet_path, corpus=self.corpus, num_topics=topics, id2word=self.words)

    def create_mallet_model_and_save(self, topics):
        model = self.create_mallet_model(topics)
        out_lda_filename = open('mallet-model', 'wb')
        pickle.dump(model, out_lda_filename)
        out_lda_filename.close()
        return model

    def compute_coherence(self, model):
        return CoherenceModel(model, texts=self.lemmatized, dictionary=self.words, coherence='c_v')

    def format_topics_sentences(self):
        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        ldamodel = init_lda_model()
        corpus = self.corpus
        for i, row in enumerate(ldamodel[corpus]):
            row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
            # row = sorted(row, key=lambda x: (x[1]), reverse=True) # old line
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(
                        pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]),
                        ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # Add original text to the end of the output
        texts = self.words
        contents = pd.Series(texts)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return sent_topics_df

    def df_topic_sent_keywords_print(self, new=0):
        if new == 1:
            df_topic_sents_keywords = self.format_topics_sentences()
        else:
            topics = open('topic-document', 'rb')
            df_topic_sents_keywords = pickle.load(topics)
            topics.close()

        # Format
        df_dominant_topic = df_topic_sents_keywords.reset_index()
        df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
        print("print topics")
        print(df_dominant_topic.head(10))

    def visualize(self, model):
        pyLDAvis.enable_notebook()
        vis = pyLDAvis.gensim.prepare(model, self.corpus, self.words)
        vis

    def normalize_output_topics(self):
        topics = best_model.show_topics(formatted=False)
        num_terms = len(topics[0][1])
        print("topic_id", end=',')
        for x in range(num_terms):
            if x == num_terms-1:
                print("term{}".format(x), end='\n')
            else:
                print("term{}".format(x), end=',')
        for i in range(len(topics)):
            print(topics[i][0], end=', ')
            terms = topics[i][1]
            for k in range(num_terms):
                if k == num_terms-1:
                    print(terms[k][0], sep='', end='\n')
                else:
                    print(terms[k][0], sep='', end=',')


if __name__ == "__main__":
    test_model = Mallet()

    mallet_values = test_model.load_multiple_mallet()
    best_score_pos = mallet_values['values'].index(max(mallet_values['values']))
    best_model = mallet_values['model'][best_score_pos]
    pprint(best_model.show_topics(formatted=False))
    test_model.normalize_output_topics()
    # exit(0)
    # test_model.run_multiple_mallet_and_print(limit=16, start=1, step=1)
    # test_model.df_topic_sent_keywords_print()
