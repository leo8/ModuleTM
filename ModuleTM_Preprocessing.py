import io
import os
import re
import spacy

#Liste de StopWords ou mots-outils
StopWords = ["un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix", "à", "à demi", "à peine", "à peu près", "absolument", "actuellement", "ainsi", "alors", "apparemment", "approximativement", "après", "après-demain", "assez", "assurément", "au", "aucun", "aucunement", "aucuns", "aujourd'", "aujourd'hui", "auparavant", "aussi", "aussitôt", "autant", "autre", "autrefois", "autrement", "avant", "avant-hier", "avec", "avoir", "beaucoup", "bien", "bientôt", "bon", "c'", "ça", "car", "carrément", "ce", "cela", "celui", "cependant", "certainement", "certes", "ces", "ceux", "chaque", "ci", "comme", "comment", "complètement", "constamment", "contre", "d'", "d'abord", "dans", "davantage", "de", "début", "dedans", "dehors", "déjà", "demain", "depuis", "derechef", "des", "désormais", "deux", "devrait", "diablement", "divinement", "doit", "donc", "dont", "dorénavant", "dos", "droite", "drôlement", "du", "elle", "elles", "en", "en vérité", "encore", "enfin", "ensuite", "entièrement", "entre", "entre-temps", "environ", "essai", "essentiel", "essentiellement", "est", "et", "étaient", "état", "été", "étions", "être", "eu", "extrêmement", "fait", "faites", "femme", "fille", "fois", "font", "force", "grand", "grandement", "guère", "habituellement", "haut", "hier", "homme", "hors", "hui", "ici", "il", "ils", "immédiatement", "infiniment", "insuffisamment", "jadis", "jamais", "je", "joliment", "jusqu", "la", "là", "le", "les", "leur", "leurs",  "lequel", "lol", "longtemps", "lors", "lui", "ma", "maintenant", "mais", "man", "mdr", "même", "mec", "merde", "mes", "mme", "moins", "mon", "mot", "naguère ",  "naturellement", "nécessaire", "ne", "ni", "nommés", "non", "notre", "nous", "nouveaux", "nullement", "ou", "où", "ouai", "ouais", "oui", "par", "parce", "parfois", "parole", "pas", "pas mal", "passablement", "pendant", "personne", "personnes", "petit", "peu", "peut", "peut-être", "pièce", "plupart", "plus", "plutôt", "point", "possible", "pour", "pourquoi", "précisément", "premier", "premièrement", "presque", "probablement", "prou", "puis", "quand", "quasi", "quasiment", "que", "quel", "quelle", "quelles", "quelque", "quelquefois", "quels", "qui", "quoi", "quotidiennement", "rien", "rudement", "s'", "sa", "sans", "sans doute", "ses", "seulement", "si", "sien", "simplement", "sitôt", "soit", "son", "sont", "soudain", "sous", "souvent", "soyez", "subitement", "suffisamment", "sur", "t'", "ta", "tandis", "tant", "tantôt", "tard", "tellement", "tellement", "tels", "terriblement", "tes", "ton", "tôt", "totalement", "toujours", "tous", "tout", "tout à fait", "toutefois", "très", "trois", "trop", "tu", "un", "une", "valeur", "vers", "voie", "voient", "volontiers", "vont", "votre", "vous", "vraiment", "vraisemblablement"]


def get_text(path):
    """
    Retourne le texte d'un fichier a partir du chemin d'acces
    """
    file = io.open(path, 'r', encoding='utf8')
    text = file.read()

    return text


def get_corpus(path):
    """
    Récupère une liste avec pour éléments les différents textes du corpus
    """
    list_path = []
    list_text = []

    for filename in os.listdir(path):
        list_path.append(f'{path}/{filename}')

    for e in list_path:
        list_text.append(get_text(e))

    return list_text


def text_cleaner(text):
    """
    Nettoie le document en gérant les problèmes de tirets et de mots composés
    """
    list_words = re.findall("[\w]+-[\w]?[^t]-[\w]+|[\w]+-\n\w+|[\w]+-[^je|tu|il|elle|nous|vous|ce|le|moi][\w]+|[\w]+-[^ev][\w]{3,}|[\w]+|[.,?!:;()…]", text)


    clean_text = ' '.join(list_words)
    clean_text = clean_text.replace("…", ".")
    clean_text = clean_text.lower()

    return clean_text


def entities_cleaner(doc, lemma):
    """
    Retire les entités nommées repérés par spacy de la liste de lèmmes
    """
    list_entities = []

    for i in range(len(doc.ents)):
        if doc.ents[i].text.lower() not in list_entities:
            list_entities.append(doc.ents[i].text.lower())

    tokenized_entities = []
    for e in list_entities:
        for k in e.split(' '):
            if k not in tokenized_entities:
                tokenized_entities.append(k)

    new_lemma = [e for e in lemma if e not in tokenized_entities]

    return new_lemma


def lemmatizer(clean_text, model='fr_core_news_lg', entities_clean=True):
    """
    Retour une liste de lemmes nettoyée
    """

    #Liste des POS ou parties du discours que l'on souhaite retirer
    bad_pos = ['DET', 'ADV', 'PUNCT', 'PRON', 'AUX', 'SCONJ', 'ADP', 'CCONJ']

    nlp = spacy.load(model)
    doc = nlp(clean_text)

    #On récupère les lèmmes en retirant les POS qui ne nous intéressent pas, ainsi que les Stopwords et les mots de moins de 3 lettres
    lemma = [token.lemma_ for token in doc if token.pos_ not in bad_pos and len(token.text) > 2 and token.lemma_ not in StopWords]

    #On applique la fonction qui permet de nettoyer les entités nommées
    if entities_clean:
        lemma = entities_cleaner(doc, lemma)

    return lemma


def corpus_lemmatizer(path):
    """
    Prend un chemin d'accès vers un dossier qui contient tous les fichiers .txt
    Retourne une liste de lemmas prête à servir pour le Topic Modeling
    """
    list_text = get_corpus(path)
    list_clean_text = [text_cleaner(e) for e in list_text]
    list_lemma = [lemmatizer(e) for e in list_clean_text]

    return list_lemma


def save_lemmas(list_lemmas, path):
    """
    Permet d'enregistrer la liste de lemmas dans un fichier .txt
    """
    with open(path, "w", encoding='utf8') as file:
        file.write(str(list_lemmas))


def read_lemmas(path):
    """
    Permet de lire les dossiers d'un fichier de lemmas précédemment sauvegardé et de les stocker dans une liste
    """
    with open(path, "r", encoding='utf8') as file:
        list_lemmas = eval(file.readline())

        return list_lemmas