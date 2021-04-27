from LDA import LDA, get_file, normalize_output_topics_csv
from gensim.models.wrappers.ldamallet import LdaMallet
import csv
import time
import sys
import os
from multiprocessing import Pool
from NLP import nlp
from dataAnalysis import compare_results
import matplotlib.pyplot as plt


# path of mallet file, place in root directory
mallet_path = './mallet-2.0.8/bin/mallet'


class MalletLDA(LDA):
    def __init__(self, path, *args):
        if len(args) > 1:
            super().__init__(path, args)
        else:
            super().__init__(path)
        self.name = "mallet"
        self.__alpha = 50
        self.__threshold = 0.0
        self.__optimize_interval = 0

    def set_alpha(self, value):
        self.__alpha = value

    def set_threshold(self, value):
        self.__threshold = value

    def set_interval(self, value):
        self.__optimize_interval = value

    def create_model(self, topics=20, workers=1):
        return LdaMallet(mallet_path, corpus=self.corpus, num_topics=topics, id2word=self.words,
                         workers=workers, alpha=self.__alpha, topic_threshold=self.__threshold, optimize_interval=self.__optimize_interval)

    def load_lda_model(self):
        return get_file("lda-data/mallet-model")


query = "61_1"
docs_num = 10
path = "test/rapid/" + query + "/docs_{}/".format(docs_num)

if __name__ == '__main__':
    # n = 20
    orig_num_topics_start = 3
    orig_num_topics_end = 7
    alphas = [0, 1, 10, 20, 40, 60, 80, 100]
    threshold = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.4]
    tmp = 0
    queries = ["32_1", "50_1", "59_2", "75_1"]
    stops = ["32_2", "50_2", "59_3", "75_2"]
    nlp(query, docs_num, True)
    mallet = MalletLDA(path)
    mallet.run_multiple_increasing_topics(6, limit=14)
    # with Pool(len(alphas)) as p:
    #     p.map(simulation.increasing_topics_set_alpha, alphas)


    sys.exit(0)
    # run mallet model with fix topic for n times, in order to get fluctuation
    # can be used for clusters

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
