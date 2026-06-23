"""Lexiques, stopwords et fonctions utilitaires bas-niveau pour les features."""

import re

# Stopwords
STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "it",
    "in",
    "of",
    "to",
    "and",
    "or",
    "but",
    "for",
    "with",
    "on",
    "at",
    "by",
    "this",
    "that",
    "as",
    "be",
    "was",
    "are",
    "were",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "not",
    "no",
    "so",
    "if",
    "from",
    "about",
    "which",
    "who",
    "they",
    "them",
    "their",
    "its",
    "my",
    "your",
    "we",
    "our",
    "i",
    "he",
    "she",
    "his",
    "her",
    "you",
    "me",
    "us",
    "will",
    "would",
    "can",
    "could",
    "should",
    "may",
    "might",
    "than",
    "then",
    "when",
    "what",
    "how",
    "there",
    "here",
    "just",
    "also",
    "more",
    "up",
    "out",
    "into",
    "all",
    "any",
    "some",
    "these",
    "those",
    "been",
    "being",
    "am",
    "each",
    "other",
    "after",
    "before",
    "because",
    "while",
    "although",
    "however",
    "therefore",
    "thus",
    "hence",
    "both",
    "either",
    "neither",
    "whether",
    "since",
    "such",
    "even",
    "only",
    "still",
    "already",
    "very",
    "quite",
    "rather",
    "much",
}

# Lexiques catégoriels
HEDGES = {
    "might",
    "maybe",
    "perhaps",
    "possibly",
    "probably",
    "seem",
    "seems",
    "appeared",
    "likely",
    "unlikely",
    "could",
    "may",
    "suggest",
    "suggests",
    "think",
    "believe",
    "suspect",
    "approximately",
    "roughly",
    "generally",
    "often",
    "sometimes",
}

POSITIVE_WORDS = {
    "good",
    "great",
    "best",
    "better",
    "positive",
    "agree",
    "right",
    "correct",
    "true",
    "helpful",
    "useful",
    "benefit",
    "advantage",
    "support",
    "love",
    "excellent",
    "perfect",
    "wonderful",
    "nice",
}

NEGATIVE_WORDS = {
    "bad",
    "worst",
    "wrong",
    "negative",
    "disagree",
    "false",
    "harmful",
    "danger",
    "problem",
    "issue",
    "fail",
    "failure",
    "terrible",
    "horrible",
    "awful",
    "hate",
    "poor",
    "weak",
}

FIRST_PERSON_SING = {"i", "me", "my", "myself", "mine"}
FIRST_PERSON_PLUR = {"we", "us", "our", "ourselves", "ours"}
SECOND_PERSON = {"you", "your", "yourself", "yours", "yourselves"}
NUMBERED_WORDS = {"first", "second", "third", "fourth", "fifth", "firstly", "secondly"}

# Lexique psycholinguistique VAD-C
VAD_C_LEXICON: dict[str, dict[str, float]] = {}


# Fonctions utilitaires


def tokenize(text: str) -> list[str]:
    """Tokenise un texte en minuscules (uniquement mots alphabétiques)."""
    return re.findall(r"\b[a-z]+\b", text.lower())


def content_words(tokens: list[str]) -> list[str]:
    """Filtre les stopwords pour ne garder que les mots pleins."""
    return [t for t in tokens if t not in STOPWORDS]


def count_syllables(word: str) -> int:
    """Estimation basique du nombre de syllabes (pour Flesch-Kincaid)."""
    word = word.lower()
    if len(word) <= 3:
        return 1
    word = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", word)
    word = re.sub(r"^y", "", word)
    syllables = len(re.findall(r"[aeiouy]{1,2}", word))
    return max(1, syllables)
