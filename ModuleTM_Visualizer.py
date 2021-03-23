import numpy as np
import pandas as pd
import gensim
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from pprintpp import pprint
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import Counter
from wordcloud import WordCloud
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook
import pyLDAvis.gensim


def topics_score_per_doc(lda_model, list_lemma):
    """
    Fonction permettant de tester la pertinence de nos topics pour chacun des documents
    """
    #Création d'un dictionnaire gensim
    array_lemma = np.array(list_lemma)
    dictionary = gensim.corpora.Dictionary(array_lemma)

    #Création d'un "bag of words" avec la fonction doc2bow
    bow_corpus = [dictionary.doc2bow(doc) for doc in array_lemma]

    for i in range(len(list_lemma)):
        print("\nFor document {}".format(i+1))
        for index, score in sorted(lda_model[bow_corpus[0]], key=lambda tup: -1*tup[1]):
            print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))


def model_to_dataframe(lda_model):
    """
    Création d'un dataframe avec les topics de notre modèle LDA
    """
    #Initialisation du DataFrame de sortie
    df = pd.DataFrame()

    topics = lda_model.show_topics(formatted=False)
    for topic, keywords in topics:
        df = df.append(pd.Series([int(topic), keywords]), ignore_index=True)

    df.columns = ['Topic_No', 'Keywords']

    return df


def dominant_topic_per_doc(lda_model, list_lemma):
    """
    Création d'un dataframe avec le topic dominant pour chaque document et sa contribution au document en pourcentage
    """

    #Création d'un dictionnaire gensim
    array_lemma = np.array(list_lemma)
    dictionary = gensim.corpora.Dictionary(array_lemma)

    #Création d'un "bag of words" avec la fonction doc2bow
    bow_corpus = [dictionary.doc2bow(doc) for doc in array_lemma]

    #Initialisation du DataFrame de sortie
    df = pd.DataFrame()

    #On récupère les topics principaux pour chaque document
    for i, row_list in enumerate(lda_model[bow_corpus]):
        row = row_list[0] if lda_model.per_word_topics else row_list
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        #On récupère la contribution en pourcentage du topic et les keywords du topic pour chaque document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:
                wp = lda_model.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                df = df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    #On ajoute la liste des lèmmes dans la dernière colonne du DataFrame
    contents = pd.Series(list_lemma)
    df = pd.concat([df, contents], axis=1)

    df = df.reset_index()
    df.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

    return df


def dataframe_displayer(df):
    """
    Fonction d'affichage des dataframes
    """

    #On paramètre les options d'affichage du module pandas
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)

    print(df)


def wordcloud(lda_model):
    """
    Création d'un nuage de mots pour chaque topic
    La taille des mots dépend de leur poids relatif au sein de chaque topic
    """
    #Sélection de différentes couleurs
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

    #Création d'un nuage de mots
    cloud = WordCloud(background_color='white', width=2500, height=1800, max_words=10, colormap='tab10', color_func=lambda *args, **kwargs: cols[i], prefer_horizontal=1.0)

    topics = lda_model.show_topics(formatted=False)

    num_topics = lda_model.num_topics

    if num_topics%2 == 0:
        r = 2
        n = int(num_topics/2)
    elif num_topics%3 == 0:
        r = 3
        n = int(num_topics/3)
    else:
        r = 1
        n = num_topics

    #Création d'une figure avec matplotlib
    fig, axes = plt.subplots(n, r, figsize=(10,10), sharex=True, sharey=True)

    #Visualisation des topics sous forme de nuage de mots
    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i+1), fontdict=dict(size=16))
        plt.gca().axis('off')

    #Ajustement des paramètres et affichage de la figure
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()


def document_distribution_per_topic(df_dominant_topic):
    """
    Tester fonctionnement --> Commentaire
    """
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'


    fig, axes = plt.subplots(2,2,figsize=(16,14), dpi=160, sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        df_dominant_topic_sub = df_dominant_topic.loc[df_dominant_topic.Dominant_Topic == i, :]
        doc_lens = [len(d) for d in df_dominant_topic.Text]
        ax.hist(doc_lens, bins = 1000, color=cols[i])
        ax.tick_params(axis='y', labelcolor=cols[i], color=cols[i])
        sns.kdeplot(doc_lens, color="black", shade=False, ax=ax.twinx())
        ax.set(xlim=(0, 1000), xlabel='Document Word Count')
        ax.set_ylabel('Number of Documents', color=cols[i])
        ax.set_title('Topic: '+str(i), fontdict=dict(size=16, color=cols[i]))

    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    plt.xticks(np.linspace(0,1000,9))
    fig.suptitle('Distribution of Document Word Counts by Dominant Topic', fontsize=22)
    plt.show()


def word_count_topic_keywords(lda_model, final_list):
    """
    Création d'un graphique pour visualiser le nombre d'occurences et le poids relatif des mots principaux pour chaque topic
    """

    #Création d'un DataFrame avec les mots, les topics, le poids relatif des mots et leur nombre d'occurences
    topics = lda_model.show_topics(formatted=False)
    data_flat = [w for w_list in final_list for w in w_list]
    counter = Counter(data_flat)
    out = []
    for i, topic in topics:
        for word, weight in topic:
            out.append([word, i , weight, counter[word]])
    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])

    num_topics = lda_model.num_topics

    if num_topics%2 == 0:
        r = 2
        n = int(num_topics/2)
    elif num_topics%3 == 0:
        r = 3
        n = int(num_topics/3)
    else:
        r = 1
        n = num_topics

    #Visualisation des données sous la forme de graphiques avec le module matplotlib
    fig, axes = plt.subplots(n, r, figsize=(16,10), sharey=True, dpi=160)
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    for i, ax in enumerate(axes.flatten()):
        ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
        ax_twin = ax.twinx()
        ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
        ax.set_ylabel('Word Count', color=cols[i])
        ax_twin.set_ylim(0, 0.030); ax.set_ylim(0, )
        ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
        ax.tick_params(axis='y', left=False)
        ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
        ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')
    fig.tight_layout(w_pad=2)
    fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)
    plt.show()


def Clustering_Chart(lda_model, list_lemma):
    """
    Création d'un graphique de topic clusters avec le module bokeh
    """

    #Création d'un dictionnaire gensim
    array_lemma = np.array(list_lemma)
    dictionary = gensim.corpora.Dictionary(array_lemma)

    #Création d'un "bag of words" avec la fonction doc2bow
    bow_corpus = [dictionary.doc2bow(doc) for doc in array_lemma]

    #Collecte des poids de chaque topic
    topic_weights = []
    for i, row_list in enumerate(lda_model[bow_corpus]):
        topic_weights.append([w for i, w in row_list])

    #Création d'un array avec les poids des topics
    arr = pd.DataFrame(topic_weights).fillna(0).values
    arr = arr[np.amax(arr, axis=1) > 0.35]

    #Nombre de topics dominant pour chaque document du corpus
    topic_num = np.argmax(arr, axis=1)

    #Création d'un modèle TSNE
    tsne_model = TSNE(n_components=2, verbose=1,    random_state=0, angle=.99, init='pca')
    tsne_lda = tsne_model.fit_transform(arr)

    #Visualisation des topic clusters avec le module Bokeh
    n_topics = lda_model.num_topics
    mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
    plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics), plot_width=900, plot_height=700)
    plot.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num])
    show(plot)


def LDA_vis(lda_model, list_lemma):
    """
    Visualisation des données sur LDAvis
    """

    #Création d'un dictionnaire gensim
    array_lemma = np.array(list_lemma)
    dictionary = gensim.corpora.Dictionary(array_lemma)

    #Création d'un "bag of words" avec la fonction doc2bow
    bow_corpus = [dictionary.doc2bow(doc) for doc in array_lemma]

    #préparation et sauvegarde des données au format html
    visualisation = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dictionary)
    pyLDAvis.save_html(visualisation, 'LDA_Visualization.html')

    #visualisation des données
    pyLDAvis.show(visualisation)


def load_model(path):

    model = gensim.models.LdaMulticore.load(path)
    return model