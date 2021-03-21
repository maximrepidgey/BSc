import pickle
import gensim
import matplotlib.pyplot as plt
from gensim.models import CoherenceModel
from pprint import pprint


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        mallet_path = './mallet-2.0.8/bin/mallet'
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=dictionary)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        print("done with {}".format(num_topics))

    return model_list, coherence_values


def init_lemmatized():
    infile = open("lemmatized", 'rb')
    tmp = pickle.load(infile)
    infile.close()
    return tmp


def init_corpus():
    infile = open("corpus", 'rb')
    tmp = pickle.load(infile)
    infile.close()
    return tmp


def init_words():
    infile = open("words", 'rb')
    tmp = pickle.load(infile)
    infile.close()
    return tmp


class Mallet:
    def __init__(self):
        self.lemmatized = init_lemmatized()
        self.corpus = init_corpus()
        self.words = init_words()

    def run_multiple_and_print(self):
        limit = 36
        start = 20
        step = 2
        model_list, coherence_values = compute_coherence_values(dictionary=self.words, corpus=self.corpus,
                                                                texts=self.lemmatized, start=start, limit=limit,
                                                                step=step)
        out_lemmatized = open('values_mallet', 'wb')
        pickle.dump({"model": model_list, "values": coherence_values}, out_lemmatized)
        out_lemmatized.close()
        # Show graph
        x = range(start, limit, step)
        plt.plot(x, coherence_values)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.legend("coherence_values", loc='best')
        plt.show()

    def play_with_values(self):
        infile = open("lda-model", 'rb')
        lda_model = pickle.load(infile)
        infile.close()
        print("lda topics")
        pprint(lda_model.print_topics())

        infile = open("coherence-model-lda", 'rb')
        coherence_model_lda = pickle.load(infile)
        infile.close()

        # coherence_model_lda = CoherenceModel(model=lda_model, texts=lemmatized, dictionary=id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()

        mallet_path = './mallet-2.0.8/bin/mallet'
        infile = open("mallet-model", 'rb')
        ldamallet = pickle.load(infile)
        infile.close()
        # ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=20, id2word=id2word)

        # Show Topics
        print("mallet topics")
        pprint(ldamallet.show_topics(formatted=False))
        coherence_model_lda_mallet = CoherenceModel(model=ldamallet, texts=self.lemmatized, dictionary=self.words,
                                                    coherence='c_v')
        coherence_lda_mallet = coherence_model_lda_mallet.get_coherence()
        print('\nCoherence Score lda classic: ', coherence_lda)
        print('\nCoherence Score mallet: ', coherence_lda_mallet)


if __name__ == "__main__":
    test_model = Mallet()
    test_model.run_multiple_and_print()
