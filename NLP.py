import re
from pprint import pprint
import pickle

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

from nltk.corpus import stopwords
import spacy  # spacy for lemmatization
import sys
# Enable logging for gensim - optional
import logging
import csv
import warnings

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

warnings.filterwarnings("ignore", category=DeprecationWarning)


def nlp(query_id="", n_documents=20, albert=False, passages=False, own_data=None):
    # NLTK Stop words
    # 5. Prepare Stopwords
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

    if own_data is None:
        if albert:
            ranked_queries_path = "test-data/cast/albert/albert_manual_qld.tsv"
            passages_path = "test-data/cast/albert/qld_manual_passages.tsv"
        else:
            ranked_queries_path = "test-data/cast/dev_manual_bm25.tsv"
            passages_path = "test-data/cast/dev_manual_bm25_passages.tsv"

        # import data
        with open(ranked_queries_path, 'r', newline='\n') as query_passages:
            rd = csv.reader(query_passages, delimiter=' ', quotechar='"')
            passages_id = []
            rank = []
            # collect all documents corresponding to that query
            for row in rd:
                if row[0] == query_id:
                    passages_id.append(row[2])
                    rank.append(int(float(row[3])))

        sorted_zipped = sorted(zip(rank, passages_id))
        sorted_passages = [element for _, element in sorted_zipped]
        passages_id = sorted_passages[:n_documents]  # top n documents

        passages_list = []
        queries_seen = []
        with open("test-data/ivan-data.tsv", "r", newline='\n') as f:
            reader = csv.reader(f, delimiter=' ', quotechar='"')

            for row in reader:
                if row[0] == query_id and row[2] in passages_id:
                    queries_seen.append(row[2])
                    passages_list.append((row[2], row[3]))
                    # passages_list.append({'id': row[2], 'relevance': row[3]})

        queries_unseen = list(set(passages_id) - set(queries_seen))
        for q in queries_unseen:
            passages_list.append((q, 0))
            # passages_list.append({'id': q, 'relevance': 0})

        with open(ranked_queries_path, 'r', newline='\n') as query_passages:
            rd = csv.reader(query_passages, delimiter=' ', quotechar='"')
            # sort basing on machine score
            rank = []
            items = []
            for row in rd:
                if row[0] == query_id:
                    for item in passages_list:
                        if row[2] == item[0]:
                            rank.append(int(float(row[3])))
                            items.append(item)

        sorted_zipped = sorted(zip(rank, items))
        passages_list = [element for _, element in sorted_zipped]

        with open("lda-data/passages", "wb") as f:
            pickle.dump(passages_list, f)

        if passages:
            print(passages_id)
            return

        # retrieve all relative passages to query id
        with open(passages_path, 'r', newline='\n') as passages:
            rd = csv.reader(passages, quotechar='"', delimiter='\t')
            passages_list = []
            for row in rd:
                if row[0] in passages_id:
                    passages_list.append(row[1])
    else:
        passages_list = own_data

    data = passages_list  # set data, must be a list
    # data = df.content.values.tolist()

    # 7. Remove emails and newline characters
    # Remove Emails
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]
    # Remove distracting single quotes
    data = [re.sub("\'", "", sent) for sent in data]
    out_words = open('lda-data/data', 'wb')  # save in order to avoid recomputing
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
    out_words = open('lda-data/words', 'wb')  # save in order to avoid recomputing
    pickle.dump(id2word, out_words)
    out_words.close()
    # Create Corpus
    texts = data_lemmatized
    out_lemmatized = open('lda-data/lemmatized', 'wb')
    pickle.dump(data_lemmatized, out_lemmatized)
    out_lemmatized.close()
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    out_corpus = open('lda-data/corpus', 'wb')
    pickle.dump(corpus, out_corpus)
    out_corpus.close()
    print("11 ended")

    return data, id2word, data_lemmatized, corpus, passages_list


if __name__ == '__main__':
    nlp("68_6", 40, passages=True, albert=True)
