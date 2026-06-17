"""
Point d'entrée — Visualisation PCA des représentations.

Produit un graphique PCA coloré par classe et le sauvegarde dans outputs/.

Utilisation :
    # Défaut : features sur WAC
    python evaluate.py

    # Visualiser les embeddings RoBERTa
    python evaluate.py encoder=roberta

    # Visualiser TF-IDF sur WAC
    python evaluate.py encoder=tfidf dataset=wac

    # Afficher la config sans lancer la visualisation
    python evaluate.py --cfg job
"""

import logging

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)


def _build_encoder(cfg: DictConfig):
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
    log.info("Visualisation PCA — encoder=%s | dataset=%s", cfg.encoder.name, cfg.dataset.name)

    # ── Chargement & encodage ────────────────────────────────────────────────
    if cfg.encoder.name == "features":
        from src.data.loader               import load_wac_pairs
        from src.encoders.features_encoder import FeaturesEncoder

        pairs = load_wac_pairs(cfg)
        split = int(len(pairs) * (1 - cfg.dataset.test_size))
        pairs_train = pairs.iloc[:split].reset_index(drop=True)
        pairs_test  = pairs.iloc[split:].reset_index(drop=True)

        encoder = FeaturesEncoder(cfg.encoder)
        X_train, X_test, y_train, y_test = encoder.fit_transform_pairwise(pairs_train, pairs_test)
        X = np.concatenate([X_train, X_test])
        y = np.concatenate([y_train, y_test])

    else:
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

        texts_train, texts_test, y_train, y_test = train_test_split(
            texts, labels,
            test_size=cfg.dataset.test_size,
            random_state=cfg.seed,
            stratify=labels if cfg.dataset.stratify else None,
        )

        encoder = _build_encoder(cfg)
        X_train, X_test = encoder.fit_transform(texts_train, texts_test)
        from scipy import sparse
        if sparse.issparse(X_train):
            X = sparse.vstack([X_train, X_test])
        else:
            X = np.concatenate([X_train, X_test])
        y = np.concatenate([y_train, y_test])

    # ── Réduction t-SNE 2D ───────────────────────────────────────────────────
    from sklearn.manifold import TSNE
    
    # Sous-échantillonnage pour t-SNE (très lent sur 60k points)
    MAX_SAMPLES = 2000
    if X.shape[0] > MAX_SAMPLES:
        log.info("Sous-échantillonnage à %d points pour t-SNE...", MAX_SAMPLES)
        indices = np.random.choice(X.shape[0], MAX_SAMPLES, replace=False)
        X = X[indices]
        y = y[indices]

    from scipy import sparse
    if sparse.issparse(X):
        # SVD préalable recommandée avant t-SNE pour le texte
        from sklearn.decomposition import TruncatedSVD
        log.info("SVD préalable (50 dimensions)...")
        X_reduced = TruncatedSVD(n_components=min(50, X.shape[1]-1)).fit_transform(X)
    else:
        from sklearn.decomposition import PCA
        log.info("PCA préalable (50 dimensions)...")
        X_reduced = PCA(n_components=min(50, X.shape[1]-1)).fit_transform(X)

    log.info("Calcul t-SNE en cours (peut prendre 10-30 sec)...")
    coords = TSNE(n_components=2, perplexity=30, random_state=cfg.seed).fit_transform(X_reduced)
    title  = f"t-SNE  —  {cfg.encoder.name.upper()}  |  {cfg.dataset.name.upper()}"

    # ── Tracé ────────────────────────────────────────────────────────────────
    palette = {1: ("steelblue", "Positif / Succès"), 0: ("tomato", "Négatif / Échec")}
    plt.figure(figsize=(9, 6))
    for label, (color, name) in palette.items():
        mask = np.array(y) == label
        plt.scatter(coords[mask, 0], coords[mask, 1],
                    c=color, s=10, alpha=0.5, label=name)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend()
    plt.tight_layout()

    out_path = "pca_visualization.png"
    plt.savefig(out_path, dpi=150)
    log.info("Figure sauvegardée : %s", out_path)
    plt.show()


if __name__ == "__main__":
    main()
