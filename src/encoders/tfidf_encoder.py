"""TF-IDF Encoder — représentation sac-de-mots pondérée."""

import numpy as np
from omegaconf import DictConfig
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.preprocessing import clean_text


class TfidfEncoder:
    """Encode des textes via TF-IDF."""

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

    def fit_transform(self, texts_train: list[str], texts_test: list[str]):
        """Fit sur le train, transform train + test."""
        train_clean = self._preprocess(texts_train)
        test_clean = self._preprocess(texts_test)

        X_train = self._vectorizer.fit_transform(train_clean)
        X_test = self._vectorizer.transform(test_clean)
        return X_train, X_test

    def transform(self, texts: list[str]):
        """Encode de nouveaux textes (utilise pour la generation/evaluation)."""
        clean = self._preprocess(texts)
        return self._vectorizer.transform(clean)
