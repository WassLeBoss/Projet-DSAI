"""
Text preprocessing utilities shared across encoders.

Usage:
    from src.data.preprocessing import clean_text, tokenize_lemmatize
"""

import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


def download_nltk_resources() -> None:
    """Télécharge les ressources NLTK si nécessaires."""
    nltk.download("wordnet",   quiet=True)
    nltk.download("stopwords", quiet=True)


def clean_text(text: str, lemmatize: bool = True) -> str:
    """
    Nettoie un texte brut : minuscules, suppression stopwords, lemmatisation optionnelle.

    Args:
        text      : texte brut
        lemmatize : si True, applique la lemmatisation WordNet

    Retourne :
        Texte nettoyé sous forme de string (tokens réunis par espace)
    """
    download_nltk_resources()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    tokens = text.lower().split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words] if lemmatize \
             else [t for t in tokens if t not in stop_words]

    return " ".join(tokens)


def tokenize_lemmatize(text: str) -> list[str]:
    """
    Tokenise et lemmatise un texte. Retourne une liste de tokens.
    Utilisé principalement par l'encodeur Word2Vec.
    """
    download_nltk_resources()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    tokens = text.lower().split()
    return [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]


def fix_sentence_spacing(text: str) -> str:
    """Ajoute un espace après les points manquants (ex: 'word.Word' → 'word. Word')."""
    return re.sub(r"\.(?=[A-Z])", ". ", text)
