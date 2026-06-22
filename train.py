"""
Point d'entrée principal — Entraînement et évaluation.

Utilisation :
    # Défaut : features stylistiques sur WAC
    python train.py

    # Changer l'encodeur
    python train.py encoder=roberta
    python train.py encoder=tfidf
    python train.py encoder=w2v

    # Changer le dataset (GriD pour détection synthétique)
    python train.py dataset=grid encoder=roberta

    # Multi-run (sweep sur tous les encodeurs)
    python train.py -m encoder=tfidf,w2v,roberta,features

    # Inclure ou non le texte du post original (op_text) pour TF-IDF/RoBERTa/Word2Vec
    python train.py include_op=true
    python train.py include_op=false

    # Afficher la config sans lancer l'entraînement
    python train.py --cfg job
"""

import logging

import hydra
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)


def _build_encoder(cfg: DictConfig):
    """
    Fabrique l'encodeur correspondant au nom dans cfg.encoder.name.
    On passe l'objet cfg complet au constructeur — pas de deballage kwargs.
    """
    name = cfg.encoder.name
    if name == "tfidf":
        from src.encoders.tfidf_encoder import TfidfEncoder
        return TfidfEncoder(cfg.encoder)
    elif name == "w2v":
        from src.encoders.w2v_encoder import W2VEncoder
        return W2VEncoder(cfg.encoder)
    elif name == "roberta":
        from src.encoders.roberta_encoder import RobertaEncoder
        return RobertaEncoder(cfg.encoder)
    else:
        raise ValueError(f"Encodeur inconnu : {name}")



@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("─" * 60)
    log.info("Dataset  : %s", cfg.dataset.name)
    log.info("Encoder  : %s", cfg.encoder.name)
    log.info("Model    : %s", cfg.model.name)
    log.info("─" * 60)

    # ── Chargement des données ───────────────────────────────────────────────
    if cfg.encoder.name == "features":
        # Mode pairwise (WAC uniquement)
        from src.data.loader           import load_wac_pairs
        from src.encoders.features_encoder import FeaturesEncoder

        pairs = load_wac_pairs(cfg)
        split = int(len(pairs) * (1 - cfg.dataset.test_size))
        pairs_train = pairs.iloc[:split].reset_index(drop=True)
        pairs_test  = pairs.iloc[split:].reset_index(drop=True)

        encoder = FeaturesEncoder(cfg.encoder)
        X_train, X_test, y_train, y_test = encoder.fit_transform_pairwise(
            pairs_train, pairs_test
        )

    else:
        # Mode vectoriel (TF-IDF / W2V / RoBERTa)
        if cfg.dataset.name == "wac":
            from src.data.loader import load_wac
            texts, labels = load_wac(cfg)
        elif cfg.dataset.name == "grid":
            from src.data.loader import load_grid
            texts, labels = load_grid(cfg)
        elif cfg.dataset.name == "hc3":
            from src.data.loader import load_hc3
            texts, labels = load_hc3(cfg)
        elif cfg.dataset.name == "m4gt":
            from src.data.loader import load_m4gt
            texts, labels = load_m4gt(cfg)
        else:
            raise ValueError(f"Dataset inconnu : {cfg.dataset.name}")

        X_raw_train, X_raw_test, y_train, y_test = train_test_split(
            texts, labels,
            test_size=cfg.dataset.test_size,
            random_state=cfg.seed,
            stratify=labels if cfg.dataset.stratify else None,
        )

        encoder = _build_encoder(cfg)
        X_train, X_test = encoder.fit_transform(X_raw_train, X_raw_test)

    log.info("Train : %d exemples | Test : %d exemples", X_train.shape[0], X_test.shape[0])

    # ── Entraînement & Évaluation ────────────────────────────────────────────
    from src.models.svm_classifier import SVMClassifier
    classifier = SVMClassifier(cfg.model)
    metrics = classifier.train_and_evaluate(X_train, X_test, y_train, y_test)

    log.info("AUC final : %.4f", metrics.auc)

    # ── Sauvegarde du modele (pour reutilisation dans Axe 3) ─────────────────
    model_path = f"axe1_svm_{cfg.encoder.name}_{cfg.dataset.name}.pkl"
    classifier.save(model_path)
    log.info("Modele sauvegarde -> %s (utilisable par Axe 3)", model_path)


if __name__ == "__main__":
    main()
