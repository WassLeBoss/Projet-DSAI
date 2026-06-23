"""RoBERTa Sentence Encoder — embeddings via SentenceTransformers."""

import numpy as np
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize


class RobertaEncoder:
    """Encode des textes via un modèle SentenceTransformers (RoBERTa par défaut)."""

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
        """Encode train et test (pas de fit — modèle pré-entraîné)."""
        X_train = self._encode(texts_train)
        X_test = self._encode(texts_test)
        return X_train, X_test

    def transform(self, texts: list[str]) -> np.ndarray:
        """Encode de nouveaux textes pour l'inférence/évaluation."""
        return self._encode(texts)
