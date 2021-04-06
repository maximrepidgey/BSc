from NLP import nlp
import LDA

import sys
import csv
import os

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

documents = [10, 20, 30, 50, 100]  # possible number of retrieved documents
dir_name = "test-run/"  # dir_name must be of format <path>/


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
            if not os.path.exists(full_file_path): os.makedirs(full_file_path)

            fig_name = full_file_path + "/score"
            labels_file_name = full_file_path+"/labels.csv"
            models_score_path = full_file_path+"/models-score"
            # todo define a good number of topics
            if n == 10:
                topic_num = 16
                step_num = 1
            elif n == 20:  # 39 topics
                topic_num = 40
                step_num = 2
            elif n == 30:  # 51 topic
                topic_num = 52
                step_num = 2
            elif n == 50:  # 76 topic
                topic_num = 77
                step_num = 3
            elif n == 100:  # 146 topic
                topic_num = 147
                step_num = 5
            else:
                topic_num = 50
                step_num = 1

            test_model = LDA.LDA()
            test_model.run_multiple_mallet_and_print(fig_name, limit=topic_num, start=1, step=step_num, path=models_score_path)
            output_csv = LDA.prepare_data_for_labelling()
            # in order to find number of topics for best model compute number of rows in labels.csv
            # this file goes to Neural embedding
            with open(labels_file_name, "w") as fb:
                writer = csv.writer(fb)
                writer.writerows(output_csv)

            # python get_labels.py -cg -us -s -d <data_file> -ocg <candidates_output> -ouns <unsupervised_output> -osup <supervised_output>
            os.chdir("NETL-Automatic-Topic-Labelling--master/model_run")
            cand_out = "./../../" + full_file_path + "/output_candidates"
            unsup_out = "./../../" + full_file_path + "/output_unsupervised"
            sup_out = "./../../" + full_file_path + "/output_supervised"
            label_file_name = "./../../"+labels_file_name
            os.system(
                "python get_labels.py -cg -us -s -d " + label_file_name + " -ocg " + cand_out + " -ouns " + unsup_out + " -osup " + sup_out)
            os.chdir("./../..")
            # sys.exit(0)  # stop for test purpose


def test():
    i = 0
    for x in range(1, 147, 5):
        i += 1
        print(x, end=", ")
    print(i)


def read_data():
    data = LDA.get_file("./test-run/50_9/docs_30/models-score")
    best_score_pos = data['values'].index(max(data['values']))
    best_score = data['model'][best_score_pos]
    print(best_score)
    print(data)


if __name__ == "__main__":
    init_query = 50
    run(10, 10)
