from LDA import LDA, get_file, normalize_output_topics_csv
from gensim.models.wrappers.ldamallet import LdaMallet
import csv
import time
import sys
import os
from statistics import stdev, variance
import scipy.stats as st

from NLP import nlp
import matplotlib.pyplot as plt


# path of mallet file, place in root directory
mallet_path = './mallet-2.0.8/bin/mallet'


class MalletLDA(LDA):
    def __init__(self):
        super().__init__()
        self.name = "mallet"

    def create_model(self, topics=20, workers=1):
        return LdaMallet(mallet_path, corpus=self.corpus, num_topics=topics, id2word=self.words,
                         workers=workers)

    def load_lda_model(self):
        return get_file("lda-data/mallet-model")


if __name__ == '__main__':
    query = "32_1"
    query_stop = "32_2"
    n = 20
    orig_num_topics_start = 3
    orig_num_topics_end = 7
    n_docs = [40]
    # n_docs = [50]
    iterations = [85]
    tmp = 0
    for doc in n_docs:
        nlp(query, query_stop, doc)
        mallet = MalletLDA()
        mallet.run_multiple_increasing_topics(21, "test/mallet-test/"+query+"/docs_{}/".format(doc), limit=iterations[tmp])
        tmp += 1

    sys.exit(0)
    # run mallet model with fix topic for n times, in order to get fluctuation
    # can be used for clusters
    models, scores = mallet.run_multiple_fix_topic(orig_num_topics_start, orig_num_topics_end, n)

    num_topic = orig_num_topics_start
    dir_name = "test/mallet-multiple/"+query+"/docs_20/"
    # analyse the result
    with open(dir_name+"statistics.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["num_topics", "mean", "max", "min", "stdev", "variance", "CI_95", "Ci_99"])
        for el in scores:
            x = range(0, n)
            print(str(num_topic) + " topics")
            # plot values
            plt.plot(x, el, linestyle='--', marker="o")
            plt.xlabel("iteration")
            plt.ylabel("Coherence score")
            plt.title("LDA models with {} topics".format(num_topic))
            plt.savefig(dir_name+"img" + str(num_topic) + ".png")
            # plt.show()
            plt.close("all")
            # compute statistics
            mean_value = sum(el)/len(el)
            CI_95 = st.t.interval(alpha=0.95, df=len(el) - 1, loc=mean_value, scale=st.sem(el))
            CI_99 = st.t.interval(alpha=0.99, df=len(el) - 1, loc=mean_value, scale=st.sem(el))
            writer.writerow([num_topic, mean_value, max(el), min(el), stdev(el), variance(el), CI_95, CI_99])
            num_topic += 1

    num_topic = orig_num_topics_start
    for el_model in models:
        num_model = 1
        for el in el_model:
            out = normalize_output_topics_csv(el)
            dir_name_labels = dir_name+"/labels/topics_"+str(num_topic)
            if not os.path.exists(dir_name_labels): os.makedirs(dir_name_labels)
            with open(dir_name_labels+"/label-"+str(num_model), "w") as label:
                wr = csv.writer(label)
                wr.writerows(out)
            num_model += 1
        num_topic += 1
