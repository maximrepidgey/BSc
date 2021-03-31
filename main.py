import re
import numpy as np
import pandas as pd
from pprint import pprint
import pickle

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from LDA import Mallet

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

# Enable logging for gensim - optional
import logging
import csv

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def nlp():
    # NLTK Stop words
    # 5. Prepare Stopwords
    from nltk.corpus import stopwords

    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

    def sent_to_words(sentences):
        for sentence in sentences:
            yield gensim.utils.simple_preprocess(str(sentence), deacc=True)  # deacc=True removes punctuations

    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(textss):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in textss]

    def make_bigrams(textss):
        return [bigram_mod[doc] for doc in textss]

    def make_trigrams(textss):
        return [trigram_mod[bigram_mod[doc]] for doc in textss]

    def lemmatization(textss, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in textss:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    # Import Newsgroups Data
    # df = pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')
    # df.head()

    with open('test-data/cast/dev_manual_bm25.tsv', 'r', newline='\n') as query_passages:
        rd = csv.reader(query_passages, delimiter=' ', quotechar='"')
        first = next(rd)  # query id
        passages_id = []
        for row in rd:
            if row[0] == '32_8':
                break
            if row[0] == '32_7':
                passages_id.append(row[2])

    # retrieve all relative passages to query id
    with open('test-data/cast/dev_manual_bm25_passages.tsv', 'r', newline='\n') as passages:
        rd = csv.reader(passages, quotechar='"', delimiter='\t')
        passages_list = []
        for row in rd:
            if row[0] in passages_id:
                passages_list.append(row[1])

    # 7. Remove emails and newline characters
    # Convert to list
    # data = df.content.values.tolist()
    data = passages_list
    # Remove Emails
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]
    # Remove distracting single quotes
    data = [re.sub("\'", "", sent) for sent in data]
    out_words = open('data', 'wb')  # save in order to avoid recomputing
    pickle.dump(data, out_words)
    out_words.close()

    # 8. Tokenize words and Clean-up text
    data_words = list(sent_to_words(data))

    # 9. Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # 10. Remove Stopwords, Make Bigrams and Lemmatize
    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)
    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)
    # Initialize spacy 'en_core_web_sm' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en_core_web_sm
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    print("nlp")
    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams,
                                    allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])  # time comsuming here
    print("lemmatizing")
    print(data_lemmatized[:1])

    # 11. Create the Dictionary and Corpus needed for Topic Modeling
    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)
    out_words = open('words', 'wb')  # save in order to avoid recomputing
    pickle.dump(id2word, out_words)
    out_words.close()
    # Create Corpus
    texts = data_lemmatized
    out_lemmatized = open('lemmatized', 'wb')
    pickle.dump(data_lemmatized, out_lemmatized)
    out_lemmatized.close()
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    out_corpus = open('corpus', 'wb')
    pickle.dump(corpus, out_corpus)
    out_corpus.close()
    print("11 ended")

    # 12. Build LDA model
    # lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
    #                                             id2word=id2word,
    #                                             num_topics=20,
    #                                             random_state=100,
    #                                             update_every=1,
    #                                             chunksize=100,
    #                                             passes=10,
    #                                             alpha='auto',
    #                                             per_word_topics=True)
    # print("model built")
    # # save model to file
    # out_lda_filename = open('lda-model', 'wb')
    # pickle.dump(lda_model, out_lda_filename)
    # out_lda_filename.close()

    # 13. View the topics in LDA model
    # Print the Keyword in the 10 topics
    # pprint(lda_model.print_topics())
    # doc_lda = lda_model[corpus]

    # TODO make work this
    # 15. Visualize the topics
    # pyLDAvis.enable_notebook()
    # vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    # vis

    # 16. Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
    # mallet_path = './mallet-2.0.8/bin/mallet'
    # ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=5, id2word=id2word)
    #
    # # Show Topics
    # pprint(ldamallet.show_topics(formatted=False))


if __name__ == "__main__":
    nlp()
    print("run multiple LDAs")
    test_model = Mallet()
    test_model.run_multiple_mallet_and_print(limit=20, start=1, step=1)
    print("finish")
