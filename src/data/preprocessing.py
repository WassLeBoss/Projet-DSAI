"""Text preprocessing utilities shared across encoders."""

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def download_nltk_resources() -> None:
    """Télécharge les ressources NLTK si nécessaires."""
    nltk.download("wordnet", quiet=True)
    nltk.download("stopwords", quiet=True)


def clean_text(text: str, lemmatize: bool = True) -> str:
    """Nettoie un texte brut : minuscules, suppression stopwords, lemmatisation optionnelle."""
    download_nltk_resources()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    tokens = text.lower().split()
    tokens = (
        [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
        if lemmatize
        else [t for t in tokens if t not in stop_words]
    )

    return " ".join(tokens)


def tokenize_lemmatize(text: str) -> list[str]:
    """Tokenise et lemmatise un texte. Retourne une liste de tokens."""
    download_nltk_resources()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    tokens = text.lower().split()
    return [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]


def fix_sentence_spacing(text: str) -> str:
    """Ajoute un espace après les points manquants (ex: 'word.Word' → 'word. Word')."""
    return re.sub(r"\.(?=[A-Z])", ". ", text)
