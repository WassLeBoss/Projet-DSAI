"""
Features d'interplay entre une réponse et le post original (OP).

Calcule le chevauchement lexical (tous mots, stopwords, mots pleins)
entre le texte de réponse et le texte OP.

Usage:
    from src.features.interplay import interplay_features
    feats = interplay_features(reply_text, op_text)
"""

from src.features.utils import tokenize, content_words, STOPWORDS


def overlap_features(reply_tokens: list[str], op_tokens: list[str], suffix: str) -> dict[str, float]:
    """
    Calcule les métriques de chevauchement entre deux listes de tokens.

    Args:
        reply_tokens : tokens de la réponse
        op_tokens    : tokens du post original
        suffix       : suffixe pour nommer les features (ex: 'all', 'stop', 'content')

    Retourne :
        Dictionnaire de 4 métriques préfixées par suffix
    """
    A = set(reply_tokens)
    O = set(op_tokens)
    inter = A & O
    union = A | O

    n_common   = len(inter)
    reply_frac = n_common / len(A)     if A     else 0.0
    op_frac    = n_common / len(O)     if O     else 0.0
    jaccard    = n_common / len(union) if union else 0.0

    return {
        f"n_common_{suffix}":   n_common,
        f"reply_frac_{suffix}": reply_frac,
        f"op_frac_{suffix}":    op_frac,
        f"jaccard_{suffix}":    jaccard,
    }


def interplay_features(reply_text: str, op_text: str) -> dict[str, float]:
    """
    Calcule toutes les features d'interplay entre une réponse et l'OP.

    Couvre trois niveaux :
      - 'all'     : tous les tokens
      - 'stop'    : uniquement les stopwords
      - 'content' : uniquement les mots pleins (sans stopwords)

    Retourne :
        Dictionnaire de 12 features (4 métriques × 3 niveaux)
    """
    reply_tok = tokenize(reply_text)
    op_tok    = tokenize(op_text)

    reply_stop = [t for t in reply_tok if t in STOPWORDS]
    op_stop    = [t for t in op_tok    if t in STOPWORDS]

    reply_cont = content_words(reply_tok)
    op_cont    = content_words(op_tok)

    feats = {}
    feats.update(overlap_features(reply_tok,  op_tok,   "all"))
    feats.update(overlap_features(reply_stop, op_stop,  "stop"))
    feats.update(overlap_features(reply_cont, op_cont,  "content"))

    return feats
