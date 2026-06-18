"""
Evaluateur d'arguments generes — utilise les modeles Axes 1 et 2.

Trois dimensions d'evaluation :
    1. Convaincance  (Axe 1 SVM) → P(success=1), score de decision SVM
    2. Authenticite  (Axe 2 SVM) → P(humain=1), le texte ressemble-t-il a un humain?
    3. Profil features           → comparaison avec les vrais winners WAC

Usage:
    from src.generation.evaluator import ArgumentEvaluator
    evaluator = ArgumentEvaluator(axe1_clf, axe2_clf, feature_names)
    report = evaluator.evaluate(op_text, generated_text, reference_winners)
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.features.pairwise  import build_feature_vector, get_feature_names
from src.features.style     import style_features

log = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    """Rapport d'evaluation d'un argument genere."""
    text:               str
    axe1_score:         float   # Score de decision SVM Axe 1 (plus eleve = plus convaincant)
    axe2_authenticity:  float   # Score de decision SVM Axe 2 (plus eleve = plus humain)
    feature_profile:    dict    = field(default_factory=dict)   # features du texte genere
    feature_delta:      dict    = field(default_factory=dict)   # ecart vs winners reels


class ArgumentEvaluator:
    """
    Evalue la qualite des arguments generes sur 3 dimensions.

    Args:
        axe1_clf      : SVM Axe 1 charge (sklearn SVC)
        axe2_clf      : SVM Axe 2 charge (sklearn SVC)
        axe1_encoder  : encodeur Axe 1 (pour les modes non-features)
        axe2_encoder  : encodeur Axe 2
        axe1_enc_name : nom de l'encodeur Axe 1 ('features'|'tfidf'|'w2v'|'roberta')
        axe2_enc_name : nom de l'encodeur Axe 2
    """

    def __init__(
        self,
        axe1_clf,
        axe2_clf,
        axe1_encoder=None,
        axe2_encoder=None,
        axe1_enc_name: str = "features",
        axe2_enc_name: str = "roberta",
    ) -> None:
        self.axe1_clf     = axe1_clf
        self.axe2_clf     = axe2_clf
        self.axe1_enc_name = axe1_enc_name
        self.axe2_enc_name = axe2_enc_name
        self.feature_names = get_feature_names()
        
        self.axe1_encoder = axe1_encoder
        self.axe2_encoder = axe2_encoder
        
        if self.axe2_encoder is None and self.axe2_enc_name == "roberta":
            from src.encoders.roberta_encoder import RobertaEncoder
            from omegaconf import OmegaConf
            cfg = OmegaConf.create({"model_name": "all-roberta-large-v1", "batch_size": 32, "show_progress_bar": False, "normalize_embeddings": True})
            self.axe2_encoder = RobertaEncoder(cfg)

    def _encode_axe1(self, text: str, op_text: str) -> np.ndarray:
        """Encode un texte pour l'Axe 1."""
        if self.axe1_enc_name == "features":
            fv   = build_feature_vector(text, op_text)
            keys = self.feature_names
            return np.array([[fv[k] for k in keys]])
        else:
            combined = f"{op_text} {text}"
            return self.axe1_encoder.transform([combined])

    def _encode_axe2(self, text: str) -> np.ndarray:
        """Encode un texte pour l'Axe 2."""
        if self.axe2_encoder is not None:
            return self.axe2_encoder.transform([text])
        # Fallback : features stylistiques si pas d'encodeur Axe 2
        fv = style_features(text)
        return np.array([[fv[k] for k in sorted(fv.keys())]])

    def evaluate(
        self,
        op_text: str,
        generated_text: str,
        reference_winners: list[str] | None = None,
    ) -> EvaluationReport:
        """
        Evalue un argument genere.

        Args:
            op_text           : texte du post original
            generated_text    : argument genere a evaluer
            reference_winners : liste de vrais arguments gagnants WAC
                                (pour calculer l'ecart de features)

        Retourne :
            EvaluationReport avec les 3 scores
        """
        # 1. Score Axe 1 (convaincance)
        X1         = self._encode_axe1(generated_text, op_text)
        axe1_score = float(self.axe1_clf.decision_function(X1)[0])

        # 2. Score Axe 2 (authenticite)
        X2              = self._encode_axe2(generated_text)
        axe2_auth_score = float(self.axe2_clf.decision_function(X2)[0])

        # 3. Profil features stylistiques
        gen_features = style_features(generated_text)

        # Ecart vs vrais winners (si fournis)
        feature_delta = {}
        if reference_winners:
            ref_profiles = [style_features(w) for w in reference_winners]
            ref_mean     = {
                k: np.mean([p[k] for p in ref_profiles])
                for k in gen_features
            }
            feature_delta = {
                k: gen_features[k] - ref_mean[k]
                for k in gen_features
            }

        report = EvaluationReport(
            text=generated_text,
            axe1_score=axe1_score,
            axe2_authenticity=axe2_auth_score,
            feature_profile=gen_features,
            feature_delta=feature_delta,
        )

        log.info(
            "Evaluation | Axe1=%.4f (convaincance) | Axe2=%.4f (authenticite)",
            axe1_score, axe2_auth_score,
        )
        return report

    def evaluate_batch(
        self,
        results: list[tuple[str, str]],
        reference_winners: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Evalue un batch de paires (op_text, generated_text).

        Retourne :
            DataFrame avec une ligne par argument genere
        """
        rows = []
        for op_text, gen_text in results:
            r = self.evaluate(op_text, gen_text, reference_winners)
            rows.append({
                "op_text":          op_text[:80] + "...",
                "generated_text":   gen_text[:120] + "...",
                "axe1_score":       r.axe1_score,
                "axe2_authenticity": r.axe2_authenticity,
                **{f"feat_{k}": v for k, v in list(r.feature_profile.items())[:10]},
            })
        return pd.DataFrame(rows)
