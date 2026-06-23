"""
Word2Vec Encoder — embeddings de phrases par moyenne de vecteurs de mots.

Usage:
    from src.encoders.w2v_encoder import W2VEncoder
    enc = TfidfEncoder(cfg.encoder)
    X_train, X_test = enc.fit_transform(texts_train, texts_test)
"""

import numpy as np
from gensim.models import Word2Vec
from omegaconf import DictConfig

from src.data.preprocessing import tokenize_lemmatize


class W2VEncoder:
    """
    Encode des textes via Word2Vec (entraîné sur les données d'entraînement).

    Chaque texte est représenté par la moyenne de ses vecteurs de mots.

    Paramètres (depuis cfg.encoder) :
        vector_size       : dimension des embeddings (default: 100)
        window            : fenêtre de contexte (default: 5)
        min_count         : fréquence minimale d'un mot (default: 1)
        workers           : threads parallèles (default: 4)
        use_lemmatization : appliquer lemmatisation avant encodage
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self._model: Word2Vec | None = None

    def _tokenize(self, texts: list[str]) -> list[list[str]]:
        if self.cfg.use_lemmatization:
            return [tokenize_lemmatize(t) for t in texts]
        return [t.lower().split() for t in texts]

    def _mean_vector(self, tokens: list[str]) -> np.ndarray:
        size = self.cfg.vector_size
        vectors = [self._model.wv[w] for w in tokens if w in self._model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(size)

    def fit_transform(
        self, texts_train: list[str], texts_test: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Entraîne Word2Vec sur le train, encode train + test.

        Retourne :
            X_train, X_test : arrays (n_samples, vector_size)
        """
        train_sentences = self._tokenize(texts_train)
        test_sentences = self._tokenize(texts_test)

        self._model = Word2Vec(
            sentences=train_sentences,
            vector_size=self.cfg.vector_size,
            window=self.cfg.window,
            min_count=self.cfg.min_count,
            workers=self.cfg.workers,
            seed=42,
        )

        X_train = np.array([self._mean_vector(s) for s in train_sentences])
        X_test = np.array([self._mean_vector(s) for s in test_sentences])
        return X_train, X_test
