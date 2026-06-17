"""
SVM Classifier — entraînement et évaluation.

Usage:
    from src.models.svm_classifier import SVMClassifier
    clf = SVMClassifier(cfg.model)
    metrics = clf.train_and_evaluate(X_train, X_test, y_train, y_test)
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, roc_auc_score

log = logging.getLogger(__name__)


@dataclass
class EvalMetrics:
    """Résultats d'évaluation du classifieur."""
    auc:    float
    report: str


class SVMClassifier:
    """
    Classifieur SVM configurable via Hydra.

    Paramètres (depuis cfg.model) :
        kernel       : noyau SVM ('linear', 'rbf', …)
        class_weight : pondération des classes ('balanced' ou None)
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        if cfg.kernel == "linear":
            self._clf = LinearSVC(
                class_weight=cfg.class_weight,
                dual=False, # Recommandé quand n_samples > n_features
                max_iter=5000
            )
        else:
            self._clf = SVC(
                kernel=cfg.kernel,
                class_weight=cfg.class_weight,
            )

    def train_and_evaluate(
        self,
        X_train: np.ndarray,
        X_test:  np.ndarray,
        y_train: np.ndarray,
        y_test:  np.ndarray,
    ) -> EvalMetrics:
        """
        Entraîne le SVM et évalue sur le test set.

        Retourne :
            EvalMetrics avec AUC et rapport de classification complet
        """
        log.info("Entraînement du SVM (kernel=%s, class_weight=%s)…",
                 self.cfg.kernel, self.cfg.class_weight)

        self._clf.fit(X_train, y_train)

        y_pred  = self._clf.predict(X_test)
        y_score = self._clf.decision_function(X_test)

        report = classification_report(y_test, y_pred)
        auc    = roc_auc_score(y_test, y_score)

        log.info("\n%s", report)
        log.info("AUC : %.4f", auc)

        return EvalMetrics(auc=auc, report=report)

    def save(self, path: str) -> None:
        """Sauvegarde le SVM entraîné sur disque (pickle)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._clf, f)
        log.info("Modele sauvegarde : %s", path)

    @classmethod
    def load_clf(cls, path: str):
        """Charge un SVM précédemment sauvegardé."""
        with open(path, "rb") as f:
            clf = pickle.load(f)
        log.info("Modele charge : %s", path)
        return clf
