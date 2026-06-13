"""
Méthode pairwise : 

Principe : pour chaque paire (winner, loser), on calcule aléatoirement 
    X = features(winner) - features(loser)  →  y = 1
    ou
    X = features(loser)  - features(winner) →  y = 0

Cette symétrie double le dataset et évite tout biais d'ordre.
"""

import numpy as np
import pandas as pd
from features_interplay import interplay_features
from features_style import style_features

def build_feature_vector(text: str, op_text: str) -> dict:
    feats = {}
    feats.update(style_features(text))
    feats.update(interplay_features(text, op_text))
    return feats


def get_feature_names() -> list[str]:
    """
    Retourne la liste triée des noms de features.
    Utile pour reconstruire les noms après np.array.
    """
    return sorted(build_feature_vector("dummy text", "dummy op").keys())


def build_pairwise_dataset(pairs_df: pd.DataFrame,
                            random_state: int = 42
                            ) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    X_rows, y_rows = [], []
    keys = get_feature_names()

    for _, row in pairs_df.iterrows():
        fw = build_feature_vector(row["winner_text"], row["op_text"])
        fl = build_feature_vector(row["loser_text"],  row["op_text"])

        if rng.random() < 0.5:
            diff = np.array([fw[k] - fl[k] for k in keys])
            y_rows.append(1)
        else:
            diff = np.array([fl[k] - fw[k] for k in keys])
            y_rows.append(0)

        X_rows.append(diff)

    return np.array(X_rows), np.array(y_rows)