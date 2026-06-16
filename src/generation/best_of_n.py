"""
Best-of-N : genere N arguments et selectionne le plus convaincant
selon le modele SVM de l'Axe 1.

Principe :
    Pour un OP donne :
    1. Generer N arguments candidats via le LLM (ArgumentGenerator)
    2. Scorer chaque candidat avec le SVM Axe 1 (decision_function)
    3. Retourner le candidat ayant le score le plus eleve

Usage:
    from src.generation.best_of_n import BestOfNSelector
    selector = BestOfNSelector(axe1_clf, axe1_encoder, cfg)
    best, scores = selector.select(op_text, candidates)
"""

import logging

import numpy as np
from omegaconf import DictConfig

log = logging.getLogger(__name__)


class BestOfNSelector:
    """
    Selectionne le meilleur argument parmi N candidats via le SVM Axe 1.

    Args:
        axe1_clf     : SVM Axe 1 charge (sklearn SVC)
        axe1_encoder : encodeur Axe 1 (TfidfEncoder, W2VEncoder, RobertaEncoder ou FeaturesEncoder)
        cfg          : config strategy.best_of_n
    """

    def __init__(self, axe1_clf, axe1_encoder, cfg: DictConfig) -> None:
        self.clf     = axe1_clf
        self.encoder = axe1_encoder
        self.cfg     = cfg

    def score_candidates(
        self, op_text: str, candidates: list[str]
    ) -> np.ndarray:
        """
        Score chaque candidat selon le SVM Axe 1.

        Pour les encodeurs vectoriels (tfidf/w2v/roberta) :
            On concatene op_text + candidate pour simuler le format WAC.

        Pour l'encodeur features (pairwise) :
            On calcule features(candidate, op_text) directement.

        Retourne :
            Array de scores (decision_function), shape (n_candidates,)
        """
        encoder_name = self.cfg.get("axe1_encoder_name", "features")

        if encoder_name == "features":
            # Mode features : on calcule le vecteur de features directement
            from src.features.pairwise import build_feature_vector, get_feature_names
            keys = get_feature_names()
            X = []
            for candidate in candidates:
                fv = build_feature_vector(candidate, op_text)
                X.append([fv[k] for k in keys])
            X = np.array(X)
        else:
            # Mode vectoriel : on encode op_text + candidate
            combined = [f"{op_text} {c}" for c in candidates]
            # Note : le fit a deja ete fait sur les donnees d'entrainement
            # Ici on utilise uniquement transform
            X = self.encoder.transform(combined)

        scores = self.clf.decision_function(X)
        return scores

    def select(
        self, op_text: str, candidates: list[str]
    ) -> tuple[str, np.ndarray]:
        """
        Selectionne le meilleur candidat.

        Args:
            op_text    : texte du post original
            candidates : liste de N arguments generes

        Retourne :
            (best_candidate, all_scores)
        """
        scores   = self.score_candidates(op_text, candidates)
        best_idx = int(np.argmax(scores))

        if self.cfg.get("verbose", False):
            for i, (c, s) in enumerate(zip(candidates, scores)):
                marker = ">>>" if i == best_idx else "   "
                log.info("%s [%.3f] %s...", marker, s, c[:80])

        log.info(
            "Best-of-%d : candidat #%d selectionne (score=%.4f)",
            len(candidates), best_idx, scores[best_idx],
        )
        return candidates[best_idx], scores
