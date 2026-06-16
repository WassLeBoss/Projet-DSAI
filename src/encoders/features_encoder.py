"""
Features Encoder — représentation pairwise par features stylistiques.

Ce mode est spécifique au dataset WAC. Il utilise les features stylistiques
et d'interplay pour construire un vecteur différentiel (winner - loser).

Usage:
    from src.encoders.features_encoder import FeaturesEncoder
    enc = FeaturesEncoder(cfg.encoder)
    X_train, X_test, y_train, y_test = enc.fit_transform_pairwise(pairs_train, pairs_test)
"""

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler

from src.features.pairwise import build_pairwise_dataset, get_feature_names


class FeaturesEncoder:
    """
    Encode les paires (winner, loser) en vecteurs différentiels de features stylistiques.

    Paramètres (depuis cfg.encoder) :
        scale        : appliquer StandardScaler sur les features
        random_state : graine pour la symétrie pairwise
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg     = cfg
        self._scaler = StandardScaler() if cfg.scale else None

    def fit_transform_pairwise(
        self,
        pairs_train: pd.DataFrame,
        pairs_test:  pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Construit les datasets pairwise train/test.

        Args:
            pairs_train : DataFrame paires d'entraînement
            pairs_test  : DataFrame paires de test

        Retourne :
            X_train, X_test, y_train, y_test
        """
        X_train, y_train = build_pairwise_dataset(pairs_train, self.cfg.random_state)
        X_test,  y_test  = build_pairwise_dataset(pairs_test,  self.cfg.random_state)

        if self._scaler is not None:
            X_train = self._scaler.fit_transform(X_train)
            X_test  = self._scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    @staticmethod
    def feature_names() -> list[str]:
        """Retourne la liste des noms de features."""
        return get_feature_names()
