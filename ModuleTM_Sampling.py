import numpy as np


def sample(list_lemmas):
    """
    Echantillonage de notre liste de lèmmes permettant de compenser les différences de taille de texte au sein du corpus
    """
    sampled_list_lemmas = []

    #On prend la taille du plus petit document du corpus pour fixer la longueur de nos échantillons
    l= []
    for e in list_lemmas:
        l.append(len(e))
    size = min(l)

    #Pour chaque texte, on génère un entier aléatoire entre 0 et la différence entre la longueur du texte et la taille de l'échantillon
    #On utilise cet entier comme un indice à partir duquel on prélève un échantillon
    sampled_list_lemmas = []
    for e in list_lemmas:
        if len(e) == size:
            sample = e
        else:
            i = np.random.randint(0, len(e)-size)
            sample = e[i:i+size]
        sampled_list_lemmas.append(sample)

    return sampled_list_lemmas



def sample_set(list_lemmas, cv=5):
    """
    Création d'un set d'échantillons aléatoires pour tester l'échantillonnage et voir si une tendance se dessine malgré le tirage au sort des échantillons
    """
    sample_set = []
    for i in range(cv):
        sample_set.append(sample(list_lemmas))

    return sample_set

def save_sample(sample_set, path):
    """
    Permet d'enregistrer la liste de lemmas dans un fichier .txt
    """
    with open(path, "w", encoding='utf8') as file:
        file.write(str(sample_set))


def read_sample(path):
    """
    Permet de lire les dossiers d'un fichier de lemmas précédemment sauvegardé et de les stocker dans une liste
    """
    with open(path, "r", encoding='utf8') as file:
        list_lemmas = eval(file.readline())

        return list_lemmas
