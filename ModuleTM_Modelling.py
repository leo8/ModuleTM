import numpy as np
import pandas as pd
import gensim
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel


def list_lemma_to_ngrams(list_lemmas, n):

    ngram_list = []
    for list in list_lemmas:
        ngram = []
        for i in range(len(list)-n):
            str = ''
            for j in range(n):
                str+= f'{list[i+j]}-'
            ngram.append(str[:-1])

        ngram_list.append(ngram)

        return ngram_list


def model_LDA_tfidf(list_lemma, num_topics=5, passes=500, workers=4, printer='False'):
    """
    Modèle de LDA avec tfidf

    Par défaut, le nombre de topics est égal à 5, et le modèle tourne avec 500 passes et 4 workers. Ces paramètres peuvent être modifiés lors de l'appel de la fonction. On définit aussi un paramètre printer afin de pouvoir choisir d'imprimer les topics dans la console.
    """

    #Création d'un dictionnaire gensim
    array_lemma = np.array(list_lemma)
    dictionary = gensim.corpora.Dictionary(array_lemma)

    #Création d'un "bag of words" avec la fonction doc2bow
    bow_corpus = [dictionary.doc2bow(doc) for doc in array_lemma]

    #Création d'un corpus tfidf à partir du bag of words
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]


    #phase d'apprentissage du modèle de LDA
    lda_model = gensim.models.LdaMulticore(corpus_tfidf, id2word=dictionary, num_topics=num_topics, passes=passes, workers=workers)

    if printer=='True':
        pprint(lda_model.print_topics())

    return lda_model


def model_LDA_gensim(list_lemma, num_topics=5, passes=500, iterations=500, printer='False'):
    """
    Modèle de LDA avec gensim

        Par défaut, le nombre de topics est égal à 5, et le modèle tourne avec 500 passes et 500 itérations. Ces paramètres peuvent être modifiés lors de l'appel de la fonction. On définit aussi un paramètre printer afin de pouvoir choisir d'imprimer les topics dans la console.
    """

    #Création d'un dictionnaire gensim
    array_lemma = np.array(list_lemma)
    dictionary = gensim.corpora.Dictionary(array_lemma)

    #Création d'un "bag of words" avec la fonction doc2bow
    bow_corpus = [dictionary.doc2bow(doc) for doc in array_lemma]

    #phase d'apprentissage du modèle de LDA
    lda_model = gensim.models.ldamodel.LdaModel(corpus=bow_corpus, id2word=dictionary,num_topics=num_topics, random_state=100, update_every=1, chunksize=10, passes=passes, alpha='symmetric', iterations=iterations, per_word_topics=True)

    if printer == 'True':
        pprint(lda_model.print_topics())

    return lda_model


def best_number_of_topics(list_lemma, max_num=25, passes=100, model_LDA='tfidf'):
    """
    Création d'un graphique représentant la cohérence du modèle obtenu en fonction du nombre de topics choisi
    """
    list_coherence = []
    list_topic_number = []

    #On choisit la méthode de LDA à utiliser : tfidf ou gensim

    if model_LDA == 'tfidf':

        #On calcule la cohérence du modèle obtenu avec les différents nombre du topic, et on crée une liste qui retient les scores de cohérence
        for i in range(max_num):
            model = model_LDA_tfidf(list_lemma, num_topics=i+1, passes=passes)
            list_coherence.append(model_coherence(model, list_lemma))
            list_topic_number.append(i+1)


        #On crée un graphique avec le nombre de topics choisi en abscisse et la cohérence du modèle obtenu en ordonnée
        x = np.array(list_topic_number)
        y = np.array(list_coherence)
        plt.plot(x, y)
        plt.title("Cohérence du modèle en fonction du nombre de topics pour {} passes".format(passes))
        plt.xlabel("Nombre de topics")
        plt.ylabel("Score de cohérence du modèle obtenu")
        plt.show()

    elif model_LDA == 'gensim':

        #On effectue la même opération avec le modèle gensim
        for i in range(max_num):
            model = model_LDA_gensim(list_lemma, num_topics=i+1, passes=passes)
            list_coherence.append(model_coherence(model, list_lemma))
            list_topic_number.append(i+1)

        x = np.array(list_topic_number)
        y = np.array(list_coherence)
        plt.plot(x, y)
        plt.title("Cohérence du modèle en fonction du nombre de topics pour {} passes".format(passes))
        plt.xlabel("Nombre de topics")
        plt.ylabel("Score de cohérence du modèle obtenu")
        plt.show()
        plt.show()

    else:
        print('Le modèle LDA choisi n\'est pas reconnu')


def model_coherence(lda_model, list_lemma):
    """
    Calcul de la cohérence du modèle LDA
    """

    #Création d'un dictionnaire gensim
    array_lemma = np.array(list_lemma)
    dic_gensim = gensim.corpora.Dictionary(array_lemma)

    #Calcule le score de cohérence du modèle LDA
    coherence_model_lda = CoherenceModel(model=lda_model, texts=list_lemma, dictionary=dic_gensim, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()

    return coherence_lda


def save_model(model, path):

    model.save(path)

    return True


def load_model(path):

    model = gensim.models.LdaMulticore.load(path)

    return model