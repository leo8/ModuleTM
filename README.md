# ModuleTM
Ce module propose des outils pour l'analyse de Topic Modeling sur des textes en français et la visualisation des résultats.

Il contient des fonctions permettant:


Le preprocessing de données textuelles :
- Tokenisation des textes, gestion des problèmes de tirets et de mots composés;
- Lemmatisation des tokens et leur éventuelle transformation en n-grams;
- La gestion des entités nommées.


L'échantillonnage des liste de lemmes obtenues :
- Création d'échantillons pour compenser les différences de longueur de texte;
- Création de sets d'échantillons pour teste le caractère persistant des résultats après différents échantillonnages aléatoires.


L'analyse de Topic Modeling avec la Latent Dirichlet Allocation : 
- Le choix d'un nombre de topics optimal pour un corpus donné;
- L'implémentation de la LDA avec deux modèles différents, gensim et tfidf;
- L'analyse de la pertinence et de la cohérence des modèles de LDA obtenus.
 

La visualisation des résultats à l'aide des outils suivants :
- Dataframes;
- Graphiques;
- Wordclouds;
- T-SNE Clustering;
- LDAvis.


------------------------------------------------------------------------------


This module aims at providing tools for Topic Modeling analysis applied to French texts and results visualization.

It contains functions that enable:


Textual Data Preprocessing :
- Text tokenizing, handling of hyphen problems and compound words;
- Tokens lemmatizing, transformation into n-grams;
- Suppression of entities .


Lemmas sampling :
- Sampling of lemmas lists to balance texts' lenghts differences;
- Creation of samples' set to test the results after different random samplings.


Topic Modeling analysis with Latent Dirichlet Allocation :
- Choice of an optimal number of topics for a given corpus;
- LDA implementation with two different models, gensim and tfidf;
- Analysis of models' relevance and coherence.
 

Results visualizing with further tools :
- Dataframes;
- Charts;
- Wordclouds;
- T-SNE Clustering;
- LDAvis.

