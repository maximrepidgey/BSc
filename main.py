from NLP import nlp
import LDA
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
queries = [{'query': '50_1', 'query_stop': '50_2', 'docs': 50},
           {'query': '32_1', 'query_stop': '32_2', 'docs': 50},
           {'query': '50_7', 'query_stop': '50_8', 'docs': 50},
           {'query': '59_2', 'query_stop': '59_3', 'docs': 50},
           {'query': '75_1', 'query_stop': '75_2', 'docs': 50}]


def run(input):
    query = input['query']
    docs = input['docs']
    path = dir_name + query + "/docs_{}/".format(docs)
    data, id2word, data_lemmatized, corpus = nlp(query, input['query_stop'], docs)
    test_model = MalletLDA(path, data_lemmatized, corpus, id2word)
    test_model.run_multiple_increasing_topics(21, path, limit=76)
    compare_results(path, 21)  # generate results.csv


def read_data():
    data = LDA.get_file("./test-run/50_9/docs_30/models-score")
    best_score_pos = data['values'].index(max(data['values']))
    best_score = data['model'][best_score_pos]
    print(best_score)
    print(data)


# generate results for running increasing number of topics
def run_full_query(docs):
    path = dir_name + query + "/docs_{}/".format(docs)
    data, id2word, data_lemmatized, corpus = nlp(query, query_stop, docs)
    test_model = MalletLDA(path, data_lemmatized, corpus, id2word)
    test_model.run_multiple_increasing_topics(21, limit=81)
    compare_results(path, 21)  # generate results.csv


query = "59_5"
query_stop = "59_6"
documents = [10, 20, 30, 40, 50]  # possible number of retrieved documents


if __name__ == "__main__":
    with Pool(len(documents)) as p:
        p.map(run_full_query, documents)
