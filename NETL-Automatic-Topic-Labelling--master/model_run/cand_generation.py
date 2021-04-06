"""
Author:         Shraey Bhatia
Date:           October 2016
File:           candidate_gen.py

Updated by:     Sihwa Park
Date:           January 7, 2019
Fix:            Updated to work with Gensim 3.6.0 and Python 3.6.5

This file generates label candidates and save the output in a file. It uses both
doc2vec and word2vec models and normalise them to unit vector. There are a couple of
pickle files namely doc2vec_indices and word2vec_indices  which restrict the search of
word2vec and doc2vec labels. These pickle files are in support_files.
"""

import os
import gensim
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from gensim.models import Word2Vec
import math
from collections import defaultdict
from numpy import exp, log, dot, zeros, outer, random, dtype, float32 as REAL,\
    double, uint32, seterr, array, uint8, vstack, fromstring, sqrt, newaxis,\
    ndarray, empty, sum, prod, ones, ascontiguousarray
from gensim import utils, matutils
import re
import pickle
import multiprocessing as mp
from multiprocessing import Pool
import argparse

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("num_cand_labels")
parser.add_argument("doc2vecmodel")
parser.add_argument("word2vecmodel")
parser.add_argument("data")
parser.add_argument("outputfile_candidates")
parser.add_argument("doc2vec_indices")

parser.add_argument("word2vec_indices")
args = parser.parse_args()

"""
Pickle file needed to run the code. These file have the indices of doc2vec which 
have length of label(wiki title less than 5). The word2vec file has indices of the
file which were used in to create phrases in word2vec model. The indices are taken from 
trained doc2vec and word2vec models. Additionally there is some bit of preprocessing involved 
of removing brackets from some candidate labels. To get more insight into it refer to the paper.
"""

with open(args.doc2vec_indices, 'rb') as m:
    d_indices = pickle.load(m)
with open(args.word2vec_indices, 'rb') as n:
    w_indices = pickle.load(n)

# Models loaded
model1 = Doc2Vec.load(args.doc2vecmodel)
model2 = Word2Vec.load(args.word2vecmodel)
print("models loaded")

# Loading the data file
topics = pd.read_csv(args.data)
try:
    new_frame = topics.drop('domain', 1)
    topic_list = new_frame.set_index('topic_id').T.to_dict('list')
except:
    topic_list = topics.set_index('topic_id').T.to_dict('list')
print("Data Gathered")


w_indices = list(set(w_indices))
d_indices = list(set(d_indices))

# Models normalised in unit vector from the indices given above in pickle files.
model1.syn0norm = (model1.wv.vectors / sqrt((model1.wv.vectors ** 2).sum(-1))[..., newaxis]).astype(REAL)
model1.docvecs.vectors_docs_norm = (model1.docvecs.vectors_docs / sqrt((model1.docvecs.vectors_docs ** 2).sum(-1))[..., newaxis]).astype(REAL)[d_indices]
print("doc2vec normalized")

model2.syn0norm = (model2.wv.vectors / sqrt((model2.wv.vectors ** 2).sum(-1))[..., newaxis]).astype(REAL)
model3 = model2.syn0norm[w_indices]
print("word2vec normalized")


# This method is mainly used to remove brackets from the candidate labels.
def get_word(word):
    if type(word) != str:
        return word
    inst = re.search(r"_\(([A-Za-z0-9_]+)\)", word)
    if inst is None:
        return word
    else:
        word = re.sub(r'_\(.+\)', '', word)
        return word


def get_labels(topic_num):
    valdoc2vec = 0.0
    valword2vec = 0.0
    cnt = 0
    store_indices = []

    print("Processing Topic number " + str(topic_num))

    for item in topic_list[topic_num]:
        try:
            # The word2vec value of topic word from doc2vec trained model
            tempdoc2vec = model1.syn0norm[model1.wv.vocab[item].index]
        except:
            pass
        else:
            meandoc2vec = matutils.unitvec(tempdoc2vec).astype(
                REAL)    # Getting the unit vector
            # The dot product of all labels in doc2vec with the unit vector of topic word
            distsdoc2vec = dot(model1.docvecs.vectors_docs_norm, meandoc2vec)
            valdoc2vec = valdoc2vec + distsdoc2vec

        try:
            # The word2vec value of topic word from word2vec trained model
            tempword2vec = model2.syn0norm[model2.wv.vocab[item].index]
        except:
            pass
        else:
            meanword2vec = matutils.unitvec(
                tempword2vec).astype(REAL)  # Unit vector

            # The dot prodiuct of all possible labels in word2vec vocab with the unit vector of topic word
            distsword2vec = dot(model3, meanword2vec)

            """
            This next section of code checks if the topic word is also a potential label in trained word2vec model. If that is the case, it is
            important the dot product of label with that topic word is not taken into account.Hence we make that zero and further down the code
            also exclude it in taking average of that label over all topic words.

            """

            if model2.wv.vocab[item].index in w_indices:

                i_val = w_indices.index(model2.wv.vocab[item].index)
                store_indices.append(i_val)
                distsword2vec[i_val] = 0.0
            valword2vec = valword2vec + distsword2vec

    print("Topic " + str(topic_num) + " (Progress 1/10): item iteration done")

    # Give the average vector over all topic words
    avgdoc2vec = valdoc2vec / float(len(topic_list[topic_num]))
    # Average of word2vec vector over all topic words
    avgword2vec = valword2vec / float(len(topic_list[topic_num]))

    print("Topic " + str(topic_num) + " (Progress 2/10): average vector done")

    # argsort and get top 100 doc2vec label indices
    bestdoc2vec = matutils.argsort(avgdoc2vec, topn=100, reverse=True)
    resultdoc2vec = []
    # Get the doc2vec labels from indices
    for elem in bestdoc2vec:
        ind = d_indices[elem]
        temp = model1.docvecs.index_to_doctag(ind)
        resultdoc2vec.append((temp, float(avgdoc2vec[elem])))

    print("Topic " + str(topic_num) + " (Progress 3/10): getting the doc2vec labels done")

    # This modifies the average word2vec vector for cases in which the word2vec label was same as topic word.
    for element in store_indices:
        avgword2vec[element] = (
            avgword2vec[element] * len(topic_list[topic_num])) / (float(len(topic_list[topic_num]) - 1))

    print("Topic " + str(topic_num) + " (Progress 4/10): modifying the average word2vec vector done")

    # argsort and get top 100 word2vec label indices
    bestword2vec = matutils.argsort(avgword2vec, topn=100, reverse=True)
    # Get the word2vec labels from indices
    resultword2vec = []
    for element in bestword2vec:
        ind = w_indices[element]
        temp = model2.wv.index2word[ind]
        resultword2vec.append((temp, float(avgword2vec[element])))

    print("Topic " + str(topic_num) + " (Progress 5/10): getting the word2vec labels from indices done")

    # Get the combined set of both doc2vec labels and word2vec labels
    comb_labels = list(
        set([i[0] for i in resultdoc2vec] + [i[0] for i in resultword2vec]))
    newlist_doc2vec = []
    newlist_word2vec = []

    print("Topic " + str(topic_num) + " (Progress 6/10): getting the combined set of both doc2vec labels and word2vec labels done")
    # time consuming number 6

    # Get indices from combined labels
    for elem in comb_labels:
        try:

            newlist_doc2vec.append(d_indices.index(
                model1.docvecs.doctags[elem].offset))
            temp = get_word(elem)
            newlist_word2vec.append(w_indices.index(model2.wv.vocab[temp].index))

        except:
            pass
    newlist_doc2vec = list(set(newlist_doc2vec))
    newlist_word2vec = list(set(newlist_word2vec))

    print("Topic " + str(topic_num) + " (Progress 7/10): getting indices from combined labels done")

    # Finally again get the labels from indices. We searched for the score from both doctvec and word2vec models
    resultlist_doc2vecnew = [(model1.docvecs.index_to_doctag(
        d_indices[elem]), float(avgdoc2vec[elem])) for elem in newlist_doc2vec]
    resultlist_word2vecnew = [(model2.wv.index2word[w_indices[elem]], float(
        avgword2vec[elem])) for elem in newlist_word2vec]

    print("Topic " + str(topic_num) + " (Progress 8/10): again getting the labels from indices done")

    # Finally get the combined score with the label. The label used will be of doc2vec not of word2vec.
    new_score = []
    for item in resultlist_word2vecnew:
        k, v = item
        for elem in resultlist_doc2vecnew:
            k2, v2 = elem
            k3 = get_word(k2)
            if k == k3:
                v3 = v + v2
                new_score.append((k2, v3))

    print("Topic " + str(topic_num) + " (Progress 9/10): getting the combined score with the label done")

    new_score = sorted(new_score, key=lambda x: x[1], reverse=True)

    print("Topic " + str(topic_num) + " (Progress 10/10): sorting score done")
    return new_score[:(int(args.num_cand_labels))]


# pool = mp.Pool(processes=2)
# result = pool.map(get_labels, range(0, len(topic_list)))
# fix to avoid out of memory
result = []
for i in range(0, len(topic_list)):
    result.append(get_labels(i))

# The output file for candidates.
g = open(args.outputfile_candidates, 'w')
for i, elem in enumerate(result):
    val = ""
    for item in elem:  # todo add number of topic when writing to file
        val = val + " " + item[0]
    g.write(val + "\n")
g.close()

print("Candidate labels written to " + args.outputfile_candidates)
print("\n")
