"""
TF-IDF Encoder — représentation sac-de-mots pondérée.

Usage:
    from src.encoders.tfidf_encoder import TfidfEncoder
    enc = TfidfEncoder(cfg.encoder)
    X_train, X_test = enc.fit_transform(texts_train, texts_test)
"""

import numpy as np
from omegaconf import DictConfig
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.preprocessing import clean_text


class TfidfEncoder:
    """
    Encode des textes via TF-IDF.

    Paramètres (depuis cfg.encoder) :
        max_features      : nombre max de features (default: 10 000)
        ngram_range       : plage de n-grammes (default: [1, 2])
        use_lemmatization : appliquer lemmatisation avant vectorisation
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        ngram = tuple(cfg.ngram_range)
        self._vectorizer = TfidfVectorizer(
            max_features=cfg.max_features,
            ngram_range=ngram,
        )

    def _preprocess(self, texts: list[str]) -> list[str]:
        if self.cfg.use_lemmatization:
            return [clean_text(t, lemmatize=True) for t in texts]
        return texts

    def fit_transform(
        self, texts_train: list[str], texts_test: list[str]
    ):
        """
        Fit sur le train, transform train + test.

        Retourne :
            X_train, X_test : matrices scipy.sparse (n_samples, n_features)
        """
        train_clean = self._preprocess(texts_train)
        test_clean  = self._preprocess(texts_test)

        X_train = self._vectorizer.fit_transform(train_clean)
        X_test  = self._vectorizer.transform(test_clean)
        return X_train, X_test

    def transform(self, texts: list[str]):
        """Encode de nouveaux textes (utilise pour la generation/evaluation)."""
        clean = self._preprocess(texts)
        return self._vectorizer.transform(clean)
