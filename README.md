# Projet-DSAI

Projet de la filière DSAI — Analyse d'arguments et détection de textes synthétiques.  
En collaboration avec M. Tristan JIN.

---

## Structure du projet

```
Projet-DSAI/
│
├── configs/                    # Configs Hydra (YAML)
│   ├── config.yaml             # Config principale
│   ├── dataset/                # wac.yaml | grid.yaml
│   ├── encoder/                # tfidf | w2v | roberta | features
│   └── model/                  # svm.yaml
│
├── src/
│   ├── data/                   # Chargement & preprocessing
│   ├── features/               # Features stylistiques & pairwise
│   ├── encoders/               # TF-IDF, W2V, RoBERTa, Features
│   └── models/                 # SVM classifier
│
├── datasets/                   # Données brutes (non versionnées)
│   ├── WinningArgCorpus/       # WAC.csv
│   └── GriD/                   # reddit_filtered_dataset.csv
│
├── outputs/                    # Résultats Hydra (auto-créé)
│
├── train.py                    # Point d'entrée entraînement
├── evaluate.py                 # Point d'entrée visualisation PCA
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Utilisation

### Entraînement

```bash
# Défaut : features stylistiques sur WAC
python train.py

# Changer l'encodeur
python train.py encoder=roberta
python train.py encoder=tfidf
python train.py encoder=w2v

# Détection de textes synthétiques (GriD)
python train.py dataset=grid encoder=roberta

# Sweep complet sur tous les encodeurs
python train.py -m encoder=tfidf,w2v,roberta,features

# Afficher la config sans lancer
python train.py --cfg job
```

### Visualisation PCA

```bash
python evaluate.py
python evaluate.py encoder=roberta
```

---

## Configuration des chemins datasets

Modifier `configs/dataset/wac.yaml` :
```yaml
csv_path: datasets/WinningArgCorpus/WAC.csv
```

Modifier `configs/dataset/grid.yaml` :
```yaml
csv_path: datasets/GriD/reddit_filtered_dataset.csv
```

---

## Résultats

Les résultats de chaque run sont sauvegardés automatiquement dans `outputs/` par Hydra :
```
outputs/
└── 2026-06-16/
    └── 15-00-00_features_wac/
        ├── train.log
        └── .hydra/config.yaml
```
