from collections import Counter
from LDA import get_file
import csv
import pickle
import os
import numpy as np
import scipy.stats as st
from statistics import stdev, variance


# compare results of multiple LDA model with different number of topics
# produce a file with all statistical data about models
# path should be of form <path>/
def compare_results(path, docs_num, start=1):
    # Function returns N largest elements
    def nmaxelements(list1, el):
        final_list = []
        for i in range(0, el):
            max1 = 0

            for j in range(len(list1)):
                if list1[j] > max1:
                    max1 = list1[j]

            list1.remove(max1)
            final_list.append(max1)

        return final_list

    # 15 columns
    rows = [["num_model", "best", "top-2", "top-3", "top-4", "mean", "min", "max", "score-2", "score-3", "score-4", "stdev", "variance", "CI-95", "CI-99"]]
    for x in range(start, docs_num):
        row = [x]
        out = open(path + "data-{}".format(x), "rb")
        data = pickle.load(out)
        out.close()
        values = data['values']
        # best
        best_score_pos = values.index(max(values))
        best_model = data['model'][best_score_pos]
        row.append(best_model)
        # top-3
        copy_values = values[:]
        top_3_values = nmaxelements(copy_values, 4)
        second_top = values.index(top_3_values[1])  # second highest
        third_top = values.index(top_3_values[2])  # third highest
        fourth_top = values.index(top_3_values[3])  # fourth highest
        row.append(data['model'][second_top])
        row.append(data['model'][third_top])
        row.append(data['model'][fourth_top])
        mean_value = sum(values)/len(values)
        row.append(mean_value)  # mean
        row.append(min(values))  # min
        row.append(max(values))  # max
        row.append(top_3_values[1])  # top-2 value
        row.append(top_3_values[2])  # top-3 value
        row.append(top_3_values[3])  # top-4 value
        row.append(stdev(values))  #stdev
        row.append(variance(values))  # variance
        # confidence interval
        row.append(st.t.interval(alpha=0.95, df=len(values) - 1, loc=mean_value, scale=st.sem(values)))
        row.append(st.t.interval(alpha=0.99, df=len(values) - 1, loc=mean_value, scale=st.sem(values)))
        rows.append(row)

    with open(path + "results.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


# analyse results from results.csv file generated by previous method
# reports the models distribution for top-1, top-3 and top-4 score
def analyse_top_best_results(file):
    # todo print also other (stdev, var, CIs) data from results.csv
    with open(file, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        best = []
        best_score = []
        second = []
        second_score = []
        third = []
        third_score = []
        fourth = []
        fourth_score = []
        for row in reader:
            best.append(row[1])
            best_score.append(row[7])
            second.append(row[2])
            second_score.append(row[8])
            third.append(row[3])
            third_score.append(row[9])
            third.append(row[4])
            third_score.append(row[10])
            fourth.append(row[4])
            fourth_score.append(row[11])
        #  top-1 values
        print("top-1 values: ", Counter(best))
        list_int = list(map(float, best_score))
        print("top-1: ", end="")
        print("mean %.5f" % (sum(list_int) / len(list_int)), end=", \t")
        print("stdev %.5f" % stdev(list_int), end=", \t")
        print("variance %.5f" % variance(list_int))
        # top-3 values
        top_3 = best + second + third
        top_3_score = best_score + second_score + third_score
        print("top-3 values: ", Counter(top_3))
        list_int = list(map(float, top_3_score))
        print("top-3: ", end="")
        print("mean %.5f" % (sum(list_int) / len(list_int)), end=", \t")
        print("stdev %.5f" % stdev(list_int), end=", \t")
        print("variance %.5f" % variance(list_int))
        # top-4 values
        top_4 = top_3 + fourth
        top_4_score = top_3_score + fourth_score
        print("top-4 values: ", Counter(top_4))
        list_int = list(map(float, top_4_score))
        print("top-4: ", end="")
        print("mean %.5f" % (sum(list_int)/len(list_int)), end=", \t")
        print("stdev %.5f" % stdev(list_int), end=", \t")
        print("variance %.5f" % variance(list_int), end="\n\n")


# runs analyse_top_best_results for the entire folder
def analyse_top_best_results_folder(path, file):
    folders = []
    for root in os.walk(path):
        folders.append(root[0])
    del folders[0]  # remove first element, since it is the root folder
    folders.sort()
    for folder in folders:
        print("----------- {} ----------".format(folder))
        analyse_top_best_results(folder+"/"+file)


# get data from all results.csv files for all number of documents retrieved
def get_results_file_intra_comparison(path, file_name):
    folders = []
    for root in os.walk(path):
        folders.append(root[0])
    del folders[0]  # remove first element, since it is the root folder
    folders.sort()
    print(folders)
    means = []
    stdevs = []
    variances = []
    for x in range(0, len(folders)):
        data_path = folders[x] + "/" + file_name
        mean_values = []
        stdev_values = []
        var_values = []
        with open(data_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                mean_values.append(row[5])
                stdev_values.append(row[11])
                var_values.append(row[12])
        means.append(mean_values)
        stdevs.append(stdev_values)
        variances.append(var_values)
    return means, stdevs, variances, folders


# computes the difference between models for various number of topics
def compare_intra_model(path, file_name):
    means, stdevs, variances, folders = get_results_file_intra_comparison(path, file_name)
    for x in range(0, len(means)):
        # comvert from string to float
        mean_values_float = list(map(float, means[x]))
        stdev_values_float = list(map(float, stdevs[x]))
        var_values_float = list(map(float, variances[x]))
        print("-----------------  {}  ----------------".format(folders[x]))
        print("MEAN values")
        print("mean %.5f" % (sum(mean_values_float) / len(mean_values_float)), end=", \t")
        print("stdev %.5f" % stdev(mean_values_float), end=", \t")
        print("variance %.5f" % variance(mean_values_float))

        print("STDEV values")
        print("mean %.5f" % (sum(stdev_values_float) / len(stdev_values_float)), end=", \t")
        print("stdev %.5f" % stdev(stdev_values_float), end=", \t")
        print("variance %.5f" % variance(stdev_values_float))

        print("VARIANCE values")
        print("mean %.5f" % (sum(var_values_float) / len(var_values_float)), end=", \t")
        print("stdev %.5f" % stdev(var_values_float), end=", \t")
        print("variance %.5f" % variance(var_values_float), end="\n\n")


# todo create method for inter model comparison, i.e., between queries
if __name__ == '__main__':
    query = "75_1"
    query = "50_7"
    query = "59_2"
    query = "32_1"
    query = "50_5"
    query = "75_8"
    query = "75_3"
    query = "32_6"
    query = "50_5"


    generate = False
    alphas = [0, 10, 20, 40, 60, 80, 100]

    for a in alphas:
        compare_results("test/mallet-test/{}/alpha-{}-docs_30/".format(query, a), 21)

    if generate:
        compare_results("test/mallet-test/{}/alpha-100-docs_30/".format(query), 11)
        compare_results("test/mallet-test/{}/docs_40/".format(query), 21)
        analyse_top_best_results("test/mallet-test/{}/docs_40/results.csv".format(query))

    analyse_top_best_results_folder("test/mallet-test/{}/".format(query), "results.csv")
    # compare_intra_model("test/mallet-test/{}".format(query), "results.csv")
    compare_intra_model("test/mallet-test/{}/".format(query), "results.csv")

