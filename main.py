from NLP import nlp
from LDA import get_file, normalize_output_topics_csv
from malletLDA import MalletLDA
from dataAnalysis import compare_results
from classicLDA import ClassicLDA
from multiprocessing import Pool
from statistics import mean
import pickle
import time
from math import ceil
import sys
import csv
import os

dir_name = "test/mallet-test/"  # dir_name must be of format <path>/
queries = [{'query': '50_1', 'docs': 50},
           {'query': '32_1', 'docs': 50},
           {'query': '50_7', 'docs': 50},
           {'query': '59_2', 'docs': 50},
           {'query': '75_1', 'docs': 50}]


def run(input):
    query = input['query']
    docs = input['docs']
    path = dir_name + query + "/docs_{}/".format(docs)
    data, id2word, data_lemmatized, corpus, passages = nlp(query, docs)
    test_model = MalletLDA(path, data_lemmatized, corpus, id2word, passages)
    test_model.run_multiple_increasing_topics(21, path, limit=76)
    compare_results(path, 21)  # generate results.csv


def read_data():
    data = get_file("test/mallet-test/61_8/fix_2-docs_30/data-1")
    models = data['model']
    print(len(models))
    models_netl= [models[0], models[13], models[17], models[19], models[22], models[26], models[27], models[38], models[49]]
    x = 1
    for mo in models_netl:
        res = normalize_output_topics_csv(mo)
        with open("test/mallet-test/59_3/fix_3-docs_30/other/labels-{}.csv".format(x), "w") as fb:
            writer = csv.writer(fb)
            writer.writerows(res)
        x+=1

# run mallet model with increasing number of topics, from 1 to fix limit
# runs for different number of retrieved documents
def increasing_topics(docs):
    path = dir_name + query + "/docs_{}/".format(docs)
    data, id2word, data_lemmatized, corpus, passages = nlp(query, docs, True)
    test_model = MalletLDA(path, data_lemmatized, corpus, id2word, passages)
    test_model.run_multiple_increasing_topics(21, limit=31)
    compare_results(path, 21)  # generate results.csv


def increasing_topics_set_alpha(a):
    path = "test/mallet-test/"+query+"/short-alpha-{}-docs_{}/".format(a, docs_num)
    mallet = MalletLDA(path)
    mallet.set_alpha(a)
    # mallet.run_multiple_increasing_topics(11, limit=58)
    mallet.run_multiple_increasing_topics(21, limit=14)
    compare_results(path, 21)  # generate results.csv


alpha = 100


def increasing_topics_set_alpha_q(q):
    path = "test/mallet-test/" + q + "/alpha_{}-docs_{}/".format(alpha, docs_num)
    data, id2word, data_lemmatized, corpus, passages = nlp(q, docs_num)
    mallet = MalletLDA(path, data_lemmatized, corpus, id2word, passages)
    mallet.set_alpha(alpha)
    mallet.run_multiple_increasing_topics(21, limit=58)
    compare_results(path, 21)  # generate results.csv


# runs multiple mallet model for fix number of topics
def fix_topics(topic):
    path = "test/mallet-test/"+query+"/fix_{}-docs_{}/".format(topic, docs_num)
    mallet = MalletLDA(path)
    mallet.run_multiple_fix_topic(topic, limit=51)
    compare_results(path, 2)  # generate results.csv


# run multiple mallet model for fix number of topics and increase alpha
# number of retrieved documents is fixed
def fix_topics_set_alpha(topic):
    for a in alphas:
        path = "test/fix/mallet/"+query+"/fix_{}-alpha_{}-docs_{}/".format(topic, a, docs_num)
        mallet = MalletLDA(path)
        mallet.set_alpha(a)
        mallet.run_multiple_fix_topic(topic, limit=31)
        compare_results(path, 2)  # generate results.csv


def var_topic_threshold(th):
    path = "test/mallet-test/"+query+"/threshold-{}-docs_{}/".format(th, docs_num)
    mallet = MalletLDA(path)
    mallet.set_threshold(th)
    mallet.run_multiple_increasing_topics(21, path, limit=76)
    compare_results(path, 21)  # generate results.csv


def fix_topic_interval_optimization(topic):
    for opt in optimizations:
        path = "test/fix/mallet/" + query + "/fix_{}-inter_{}-docs_{}/".format(topic, opt, docs_num)
        mallet = MalletLDA(path)
        mallet.set_interval(opt)
        mallet.run_multiple_fix_topic(topic, limit=31)
        compare_results(path, 2)  # generate results.csv


def fix_topics_eta(topic):
    for eta in etas:
        path = "test/fix/classic/" + query + "/fix_{}-eta_{}-docs_{}/".format(topic, eta, docs_num)
        model = ClassicLDA(path)
        model.set_eta(eta)
        model.run_multiple_fix_topic(topic, limit=31)
        compare_results(path, 2)  # generate results.csv


def query_distribution(query):
    data, id2word, data_lemmatized, corpus, passages = nlp(query, docs_num, True)
    final_data = []
    q_num = query[0:2]
    q_part = query[3:]
    for t in topics:
        path = "test/dist/mallet/" + q_num + "/" + q_part + "/fix_{}-docs_{}/".format(t, docs_num)
        classic = MalletLDA(path, data_lemmatized, corpus, id2word, passages)
        classic.run_multiple_fix_topic(t, 21)
        # find the best model
        values = get_file(path + "data-1")
        best_score_pos = values['values'].index(max(values['values']))
        best_model = values['model'][best_score_pos]
        with open(path + "best", "wb") as f:
            pickle.dump(best_model, f)

        final_data.append(classic.df_topic_sent_keywords_print())

    # create a list with max value for each topic number
    max_per_topic = [max(row[2:]) for row in final_data]
    max_per_topic.append(mean(max_per_topic))  # average per this query

    final_data.append(max_per_topic)
    with open("test/dist/mallet/" + q_num + "/" + q_part + "/results-docs_{}.csv".format(docs_num), "w") as f:
        writer = csv.writer(f)
        writer.writerows(final_data)


queries_list_done = ["32_1", "32_3", "32_6", "50_1", "50_5", "50_7", "59_2", "59_5", "75_1", "75_3", "75_8"]
query_main = "67_{}"
queries_list = [query_main.format(x) for x in range(1, 11)]

query = "59_3"
docs_num = 30
documents = [10, 20, 30, 40, 50]  # possible number of retrieved documents
alphas = [0, 1, 10, 20, 40, 60, 80, 100]
topics1 = [1, 2, 3, 4]
topics2 = [5, 6, 7, 8]
topics = topics1 + topics2

etas = [0.0001, 0.001, 0.01, 0.1, 1, 10]

optimizations = [1, 10, 20, 50, 100, 200]
fast_query = ["75_1", "75_3", "75_8", "50_7", "50_5"]
query_albert = ["37_{}".format(i) for i in range(1, 9)]


def rename(path):
    for d in os.listdir(path):
        # not .DS_Store
        if d[0] != ".":
            os.rename(path + d, path + d[3:])


if __name__ == "__main__":
    # query_distribution("32_7")
    # sys.exit(0)
    # nlp(query, docs_num, True)
    print(query_albert)
    parameters = query_albert
    with Pool(4) as p:
        p.map(query_distribution, parameters)
