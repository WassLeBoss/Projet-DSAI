"""Features d'interplay entre une réponse et le post original (OP)."""

from src.features.utils import STOPWORDS, content_words, tokenize


def overlap_features(
    reply_tokens: list[str], op_tokens: list[str], suffix: str
) -> dict[str, float]:
    """Calcule les métriques de chevauchement entre deux listes de tokens."""
    A = set(reply_tokens)
    O = set(op_tokens)
    inter = A & O
    union = A | O

    n_common = len(inter)
    reply_frac = n_common / len(A) if A else 0.0
    op_frac = n_common / len(O) if O else 0.0
    jaccard = n_common / len(union) if union else 0.0

    return {
        f"n_common_{suffix}": n_common,
        f"reply_frac_{suffix}": reply_frac,
        f"op_frac_{suffix}": op_frac,
        f"jaccard_{suffix}": jaccard,
    }


def interplay_features(reply_text: str, op_text: str) -> dict[str, float]:
    """Calcule toutes les features d'interplay entre une réponse et l'OP."""
    reply_tok = tokenize(reply_text)
    op_tok = tokenize(op_text)

    reply_stop = [t for t in reply_tok if t in STOPWORDS]
    op_stop = [t for t in op_tok if t in STOPWORDS]

    reply_cont = content_words(reply_tok)
    op_cont = content_words(op_tok)

    feats = {}
    feats.update(overlap_features(reply_tok, op_tok, "all"))
    feats.update(overlap_features(reply_stop, op_stop, "stop"))
    feats.update(overlap_features(reply_cont, op_cont, "content"))

    return feats
