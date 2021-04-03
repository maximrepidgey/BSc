from NLP import nlp
from LDA import Mallet

import sys
import csv
import os

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

documents = [10, 20, 30, 50, 100]  # possible retrieved documents
dir_name = "./test-run/"


def run(first, last):
    last += 1
    query = str(init_query)
    query += '_{}'
    for x in range(first, last):
        for n in documents:
            print("run LDA for {} documents for query {}".format(n, query.format(x)))
            if x == 10:
                query_stop = str(init_query + 1)
                query_stop += '_1'
                nlp(query.format(x), query_stop, n)
            else:
                nlp(query.format(x), query.format(x + 1), n)

            file_name = query.format(x) + "/docs_{}".format(n)
            full_file_path = dir_name + file_name
            if not os.path.exists(full_file_path):
                os.makedirs(full_file_path)

            fig_name = full_file_path + "/score"
            labels_file_name = full_file_path+"/labels.csv"
            # todo define a good number of topics
            if n == 10:
                topic_num = 15
                step_num = 1
            elif n == 20:
                topic_num = 37
                step_num = 2
            elif n == 30:
                topic_num = 49
                step_num = 2
            elif n == 50:
                topic_num = 75
                step_num = 3
            elif n == 100:
                topic_num = 146
                step_num = 5
            else:
                topic_num = 50
                step_num = 1

            test_model = Mallet()
            test_model.run_multiple_mallet_and_print(fig_name, limit=topic_num, start=1, step=step_num)
            best_score_model, output_csv = test_model.prepare_data_for_labelling()
            #  TODO write into file, solution: compute number of rows in labels.csv
            print(best_score_model)
            # this file goes to Neural embedding
            with open(labels_file_name, "w") as fb:
                writer = csv.writer(fb)
                writer.writerows(output_csv)
            # python get_labels.py -cg -us -s -d <data_file> -ocg <candidates_output> -ouns <unsupervised_output> -osup <supervised_output>
            os.chdir("NETL-Automatic-Topic-Labelling--master/model_run")
            cand_out = "./../../test-run/" + file_name + "/output_candidates"
            unsup_out = "./../../test-run/" + file_name + "/output_unsupervised"
            sup_out = "./../../test-run/" + file_name + "/output_supervised"
            label_file_name = "./../."+labels_file_name
            os.system(
                "python get_labels.py -cg -us -s -d " + label_file_name + " -ocg " + cand_out + " -ouns " + unsup_out + " -osup " + sup_out)
            os.chdir("./../..")
            # sys.exit(0)  # stop for test purpose


def test():
    query = str(init_query)
    query += '_{}'
    print(query.format(2 + 1))


if __name__ == "__main__":
    init_query = 50
    run(1, 2)
