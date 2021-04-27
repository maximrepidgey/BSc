from NLP import nlp
from LDA import get_file, normalize_output_topics_csv
from malletLDA import MalletLDA
from dataAnalysis import compare_results
from classicLDA import ClassicLDA
from multiprocessing import Pool
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
    data, id2word, data_lemmatized, corpus = nlp(query, docs)
    test_model = MalletLDA(path, data_lemmatized, corpus, id2word)
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
    path = dir_name+query+"/docs_{}/".format(docs)
    data, id2word, data_lemmatized, corpus = nlp(query, docs, True)
    test_model = MalletLDA(path, data_lemmatized, corpus, id2word)
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
    path = "test/mallet-test/"+q+"/alpha_{}-docs_{}/".format(alpha, docs_num)
    data, id2word, data_lemmatized, corpus = nlp(q, docs_num)
    mallet = MalletLDA(path, data_lemmatized, corpus, id2word)
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


queries_list_done = ["32_1", "32_3", "32_6", "50_1", "50_5", "50_7", "59_2", "59_5", "75_1", "75_3", "75_8"]
query_main = "67_{}"
queries_list = [query_main.format(x) for x in range(1, 11)]


query = "59_3"
docs_num = 30
documents = [10, 20, 30, 40, 50]  # possible number of retrieved documents
alphas = [0, 1, 10, 20, 40, 60, 80, 100]
topics1 = [1, 2, 3, 4]
topics2 = [5, 6, 7, 8]
topics = topics1+topics2

etas = [0.0001, 0.001, 0.01, 0.1, 1, 10]

optimizations = [1, 10, 20, 50, 100, 200]
fast_query = ["75_1", "75_3", "75_8", "50_7", "50_5"]


if __name__ == "__main__":
    nlp(query, docs_num, True)
    parameters = topics
    with Pool(len(parameters)) as p:
        p.map(fix_topics_set_alpha, parameters)

