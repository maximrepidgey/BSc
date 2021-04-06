import pickle
import gensim
from gensim.models import CoherenceModel
from gensim.models.ldamulticore import LdaMulticore
from pprint import pprint
import pandas as pd
import os
import time

import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
from multiprocessing import Pool
import csv

import warnings


os.environ.update({'MALLET_HOME': r'./mallet-2.0.8/bin/mallet'})
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# path of mallet file, place in root directory
mallet_path = './mallet-2.0.8/bin/mallet'


def init_lemmatized():
    return get_file("lda-data/lemmatized")


def init_corpus():
    return get_file("lda-data/corpus")


def init_words():
    return get_file("lda-data/words")


def load_lda_model():
    return get_file("lda-data/lda-model")


def load_mallet_model():
    return get_file("lda-data/mallet-model")


def load_topics():
    return get_file("lda-data/topic-document")


def get_file(filename):
    infile = open(filename, 'rb')
    tmp = pickle.load(infile)
    infile.close()
    return tmp


def load_multiple_mallet():
    return get_file("lda-data/values_mallet")


def normalize_output_topics(best):
    topics = best.show_topics(formatted=False, num_topics=best.get_topics().shape[0])  # all topics
    num_terms = len(topics[0][1])
    output = [["topic_id"]]
    for x in range(num_terms):
        if x == num_terms - 1:
            tmp = "term{}".format(x)
        else:
            tmp = "term{}".format(x)
        output[0].append(tmp)
    for i in range(len(topics)):
        # set topics in order, it is important for neural embedding
        line = [i]
        terms = topics[i][1]
        for k in range(num_terms):
            if k == num_terms - 1:
                line.append(str(terms[k][0]))
            else:
                line.append(str(terms[k][0]))
        output.append(line)
    return output


def prepare_data_for_labelling():
    mallet_values = load_multiple_mallet()
    best_score_pos = mallet_values['values'].index(max(mallet_values['values']))
    best_model = mallet_values['model'][best_score_pos]
    # pprint(best_model.show_topics(formatted=False))
    output_csv = normalize_output_topics(best_model)
    return output_csv


def plot_multiple_models(start, limit, step, coherence_values, name=None, show=False):
    # Show graph
    x = range(start, limit, step)
    plt.plot(x, coherence_values, linestyle='--', marker="o")
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend("coherence_values", loc='best')
    # max point
    max_value = max(coherence_values)
    xpos = coherence_values.index(max_value)
    x_max = x[xpos]
    # zip joins x and y coordinates in pairs
    for xs, ys in zip(x, coherence_values):
        label = "{}".format(xs)
        if xs == x_max:  # annotate the max value in red
            plt.annotate(x_max, (x_max, max_value), fontsize=9, color='red', textcoords="offset points",
                         xytext=(0, 5), ha='center')
            continue
        plt.annotate(label,  # this is the text
                     (xs, ys),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 8),  # distance from text to points (x,y)
                     ha='center',  # horizontal alignment can be left, right or center
                     fontsize=8)
    if name is not None:
        name += '.png'
        plt.savefig(name)
    if show: plt.show()
    plt.close('all')


class LDA:
    def __init__(self):
        self.lemmatized = init_lemmatized()
        self.corpus = init_corpus()
        self.words = init_words()

    def run_multiple_mallet_and_print(self, name, limit=10, start=1, step=2, path=None):
        model_list, coherence_values = self.compute_coherence_values(limit, start, step)
        # save values
        out_lemmatized = open('lda-data/values_mallet', 'wb')
        pickle.dump({"model": model_list, "values": coherence_values}, out_lemmatized)
        out_lemmatized.close()
        # save coherence score for each model
        x = range(start, limit, step)
        if path is not None:
            out_data = open(path, 'wb')
            pickle.dump({'model': list(x), 'values': coherence_values}, out_data)
            out_data.close()

        plot_multiple_models(start, limit, step, coherence_values, name=name)

    def compute_coherence_values(self, limit=20, start=1, step=3):
        coherence_values = []
        model_list = []
        for num_topics in range(start, limit, step):
            model = self.create_mallet_model(num_topics)
            model_list.append(model)

            coherence_model = self.compute_coherence(model)
            coherence_values.append(coherence_model.get_coherence())
            print("done with {} topics".format(num_topics))

        return model_list, coherence_values

    def create_mallet_model(self, topics=20, workers=2):
        return gensim.models.wrappers.LdaMallet(mallet_path, corpus=self.corpus, num_topics=topics, id2word=self.words,
                                                workers=workers)

    def create_mallet_model_and_save(self, topics):
        model = self.create_mallet_model(topics)
        out_lda_filename = open('lda-data/mallet-model', 'wb')
        pickle.dump(model, out_lda_filename)
        out_lda_filename.close()
        return model

    def create_lda_model(self, topics, workers=2):
        return LdaMulticore(corpus=self.corpus, num_topics=topics, id2word=self.words, workers=workers)

    def compute_coherence(self, model, workers=2, topics=None):
        return CoherenceModel(model, topics=topics, texts=self.lemmatized, dictionary=self.words, coherence='c_v', topn=10, processes=workers)

    def format_topics_sentences(self):
        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        ldamodel = load_lda_model()
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
            topics = open('lda-data/topic-document', 'rb')
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

    def compute_top_topics(self, model):
        return model.top_topics(corpus=self.corpus, texts=self.lemmatized, dictionary=self.words, coherence='c_v',
                                processes=2, topn=10)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    lda = LDA()
    # lda.run_multiple_mallet_and_print(limit=16, start=1, step=1)
    # lda.df_topic_sent_keywords_print()
    topics = 106
    tmp_value = load_multiple_mallet()
    tmp_value = tmp_value['values']
    print(len(tmp_value))
    models_num = range(1, 40, 2)
    print(len(list(models_num)))
    exit(0)

    start_time = time.time()
    mallet_model = lda.create_lda_model(topics)
    print("--- %s seconds --- for mallet" % (time.time() - start_time))

    start_time = time.time()
    lda_model = lda.create_lda_model(topics)
    print("--- %s seconds --- for lda parallelized" % (time.time() - start_time))

    print(lda.compute_coherence(mallet_model).get_coherence(), end=", ")
    print(lda.compute_coherence(lda_model).get_coherence())
    top_topics = lda.compute_top_topics(lda_model)
    print(len(top_topics), end=', ')
    print(lda.compute_coherence(None, topics=top_topics).get_coherence())

