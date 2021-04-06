import pickle
from gensim.models import CoherenceModel
from pprint import pprint
import pandas as pd
import os
import time

from abc import abstractmethod

import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
from multiprocessing import Pool
import csv


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


def load_mallet_model():
    return get_file("lda-data/mallet-model")


def load_topics():
    return get_file("lda-data/topic-document")


def get_file(filename):
    infile = open(filename, 'rb')
    tmp = pickle.load(infile)
    infile.close()
    return tmp


def normalize_output_topics_csv(best):
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


# return list of list of terms
def get_formatted_topics(model):
    topics = model.show_topics(formatted=False, num_topics=model.get_topics().shape[0])  # all topics
    formatted_topics = []
    for topic in topics:
        terms = []
        for term in topic[1]:
            terms.append(term[0])
        formatted_topics.append(terms)
    return formatted_topics


class LDA:
    def __init__(self):
        self.lemmatized = init_lemmatized()
        self.corpus = init_corpus()
        self.words = init_words()
        self.name = ""

    def get_name(self):
        return self.name

    @abstractmethod
    def load_lda_model(self):
        pass

    @abstractmethod
    def create_model(self, topics=20, workers=2):
        pass

    def load_multiple_lda(self):
        return get_file("lda-data/values_" + self.get_name())

    def run_multiple_lda_and_print(self, name=None, limit=10, start=1, step=2, path=None):
        model_list, coherence_values = self.compute_coherence_values(limit, start, step)
        # save values
        out_lemmatized = open('lda-data/values_' + self.get_name(), 'wb')
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
            model = self.create_model(num_topics)
            model_list.append(model)

            coherence_model = self.compute_coherence(model)
            coherence_values.append(coherence_model.get_coherence())
            print("done with {} topics".format(num_topics))

        return model_list, coherence_values

    # runs multiple LDA model fot k topics for n times
    def run_multiple(self, n, folder, step=1, limit=16, start=1):
        path = "test/" + self.get_name() + "-test/" + folder + "/"
        if not os.path.exists(path): os.makedirs(path)

        for x in range(1, n):
            self.run_multiple_lda_and_print(folder + "img" + str(x), start=start, step=step, limit=limit,
                                            path=path + "data-" + str(x))
            output_csv = self.prepare_data_for_labelling()
            # in order to find number of topics for best model compute number of rows in labels.csv
            # this file goes to Neural embedding
            with open(path + "labels-" + str(x) + ".csv", "w") as fb:
                writer = csv.writer(fb)
                writer.writerows(output_csv)

    #  n-1 should be equal to number of labels.csv files
    def run_neural_embedding(self, n, folder):
        path = "test/" + self.get_name() + "-test/" + folder

        os.chdir("NETL-Automatic-Topic-Labelling/model_run")
        for x in range(1, n):
            cand_out = "./../../" + path + "/output_candidates"+str(x)
            unsup_out = "./../../" + path + "/output_unsupervised"+str(x)
            sup_out = "./../../" + path + "/output_supervised"+str(x)
            label_file_name = "./../../" + path+"/labels-"+str(x)+".csv"
            os.system(
                "python get_labels.py -cg -us -s -d " + label_file_name + " -ocg " + cand_out + " -ouns " + unsup_out + " -osup " + sup_out)
        os.chdir("./../..")

    def create_model_and_save(self, topics, workers=2):
        tmp = self.create_model(topics, workers)
        out = open("lda-data/" + self.get_name(), "wb")
        pickle.dump(tmp, out)
        out.close()
        return tmp

    def compute_coherence(self, model, workers=2, topics=None):
        return CoherenceModel(model, topics=topics, texts=self.lemmatized, dictionary=self.words, coherence='c_v',
                              topn=10, processes=workers)

    def prepare_data_for_labelling(self):
        mallet_values = self.load_multiple_lda()
        best_score_pos = mallet_values['values'].index(max(mallet_values['values']))
        best_model = mallet_values['model'][best_score_pos]
        # pprint(best_model.show_topics(formatted=False))
        output_csv = normalize_output_topics_csv(best_model)
        return output_csv

    def format_topics_sentences(self):
        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        ldamodel = self.load_lda_model()
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


if __name__ == "__main__":
    lda = LDA()
    lda.df_topic_sent_keywords_print(1)
