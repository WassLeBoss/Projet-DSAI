"""
RoBERTa Sentence Encoder — embeddings via SentenceTransformers.

Usage:
    from src.encoders.roberta_encoder import RobertaEncoder
    enc = RobertaEncoder(cfg.encoder)
    X_train, X_test = enc.fit_transform(texts_train, texts_test)
"""

import numpy as np
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize


class RobertaEncoder:
    """
    Encode des textes via un modèle SentenceTransformers (RoBERTa par défaut).

    Aucun entraînement : le modèle est pré-entraîné et utilisé en inférence.

    Paramètres (depuis cfg.encoder) :
        model_name             : identifiant HuggingFace du modèle
        batch_size             : taille de batch pour l'encodage
        show_progress_bar      : afficher la barre de progression
        normalize_embeddings   : normaliser les embeddings (L2)
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self._model = SentenceTransformer(cfg.model_name)

    def _encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self._model.encode(
            texts,
            batch_size=self.cfg.batch_size,
            show_progress_bar=self.cfg.show_progress_bar,
        )
        if self.cfg.normalize_embeddings:
            embeddings = normalize(embeddings)
        return embeddings

    def fit_transform(
        self, texts_train: list[str], texts_test: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Encode train et test (pas de fit — modèle pré-entraîné).

        Retourne :
            X_train, X_test : arrays (n_samples, embedding_dim)
        """
        X_train = self._encode(texts_train)
        X_test = self._encode(texts_test)
        return X_train, X_test

    def transform(self, texts: list[str]) -> np.ndarray:
        """Encode de nouveaux textes pour l'inférence/évaluation."""
        return self._encode(texts)
