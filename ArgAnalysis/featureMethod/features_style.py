import re
import numpy as np
from collections import Counter
from features_utils import (
    tokenize, content_words, count_syllables, 
    HEDGES, POSITIVE_WORDS, NEGATIVE_WORDS, 
    FIRST_PERSON_SING, FIRST_PERSON_PLUR, SECOND_PERSON, 
    NUMBERED_WORDS, VAD_C_LEXICON
)

def style_features(text):
    tokens = tokenize(text)
    n = len(tokens) or 1
    sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    n_sents = len(sents) or 1
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]

    unique = len(set(tokens))

    # Comptages absolus
    n_hedges = sum(1 for t in tokens if t in HEDGES)
    n_pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
    n_neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    n_1sg = sum(1 for t in tokens if t in FIRST_PERSON_SING)
    n_1pl = sum(1 for t in tokens if t in FIRST_PERSON_PLUR)
    n_2p = sum(1 for t in tokens if t in SECOND_PERSON)
    n_definite = sum(1 for t in tokens if t == "the")
    n_indefinite = sum(1 for t in tokens if t in {"a", "an"})
    n_num_words = sum(1 for t in tokens if t in NUMBERED_WORDS)

    # Liens et spécificités
    n_links = len(re.findall(r"https?://\S+", text))
    n_com_links = len(re.findall(r"https?://\S*\.com\S*", text))
    n_edu_links = len(re.findall(r"https?://\S*\.edu\S*", text))
    n_pdf_links = len(re.findall(r"https?://\S*\.pdf\S*", text, re.IGNORECASE))
    
    n_examples = len(re.findall(r"\bfor example\b|\bfor instance\b|\be\.g\.", text, re.IGNORECASE))
    n_questions = text.count("?")
    n_quotes = len(re.findall(r"^>", text, re.MULTILINE))

    # Richesse lexicale
    type_token = unique / n
    freq = Counter(tokens)
    total = sum(freq.values())
    entropy = -sum((c / total) * np.log2(c / total) for c in freq.values() if c > 0)

    # Flesch-Kincaid Grade Level
    total_syllables = sum(count_syllables(t) for t in tokens)
    flesch_kincaid = 0.39 * (n / n_sents) + 11.8 * (total_syllables / n) - 15.59

    # Markdown
    n_bold = len(re.findall(r"\*\*.*?\*\*", text))
    n_italic = len(re.findall(r"\*[^*]+\*|_[^_]+_", text))
    n_bullet = len(re.findall(r"^\s*[-*]\s", text, re.MULTILINE))

    # Scores Psycholinguistiques (VAD-C moyenné sur les mots pleins)
    c_words = content_words(tokens)
    vad_c_scores = {"valence": [], "arousal": [], "dominance": [], "concreteness": []}
    for w in c_words:
        if w in VAD_C_LEXICON:
            for dim in vad_c_scores:
                vad_c_scores[dim].append(VAD_C_LEXICON[w][dim])
                
    avg_valence = np.mean(vad_c_scores["valence"]) if vad_c_scores["valence"] else 0.0
    avg_arousal = np.mean(vad_c_scores["arousal"]) if vad_c_scores["arousal"] else 0.0
    avg_dominance = np.mean(vad_c_scores["dominance"]) if vad_c_scores["dominance"] else 0.0
    avg_concreteness = np.mean(vad_c_scores["concreteness"]) if vad_c_scores["concreteness"] else 0.0

    return {
        # Longueur
        "n_words": n,
        "n_sentences": n_sents,
        "n_paragraphs": len(paras),

        # Mots (Catégories)
        "n_definite": n_definite,
        "n_indefinite": n_indefinite,
        "n_positive": n_pos,
        "n_2p": n_2p,
        "n_links": n_links,
        "n_negative": n_neg,
        "n_hedges": n_hedges,
        "n_1sg": n_1sg,
        "n_1pl": n_1pl,
        "n_com_links": n_com_links,
        "n_pdf_links": n_pdf_links,
        "n_edu_links": n_edu_links,
        "n_examples": n_examples,
        "n_questions": n_questions,
        "n_quotes": n_quotes,

        # Fractions (Normalisation par la longueur)
        "frac_links": n_links / n,
        "frac_com_links": n_com_links / n,
        "frac_definite": n_definite / n,
        "frac_positive": n_pos / n,
        "frac_questions": n_questions / n,
        "frac_italics": n_italic / n,

        # Scores lexicaux
        "arousal": avg_arousal,
        "valence": avg_valence,
        "dominance": avg_dominance,
        "concreteness": avg_concreteness,

        # Argument global
        "word_entropy": entropy,
        "type_token": type_token,
        "flesch_kincaid": flesch_kincaid,

        # Markdown
        "n_italics": n_italic,
        "n_bullet_list": n_bullet,
        "n_bolds": n_bold,
        "n_numbered_words": n_num_words,
    }