# Projet DSAI : Persuasion, Détection d'IA et Génération Stratégique

Ce dépôt contient le code source de notre projet réalisé dans le cadre de la filière DSAI. Le projet porte sur l'analyse d'arguments (dataset WAC), la détection de textes générés par IA (datasets HC3 & M4GT) et la génération de discours persuasif à l'aide de Modèles de Langage (LLMs).

Projet supervisé par : M. Tristan JIN.

---

## 1. Structure du projet

```text
Projet-DSAI/
│
├── configs/                    # Fichiers de configuration Hydra (YAML)
│   ├── dataset/                # wac, hc3, m4gt, grid
│   ├── encoder/                # tfidf, w2v, roberta, features
│   ├── model/                  # svm
│   └── generation/             # Configuration de l'Axe 3 (Génération)
│
├── src/
│   ├── data/                   # Chargement et prétraitement (WAC, HC3, M4GT)
│   ├── features/               # Extraction de features (stylistiques, Jaccard)
│   ├── encoders/               # Encodeurs : TF-IDF, Word2Vec, RoBERTa
│   ├── models/                 # Modèles de classification (SVM)
│   └── generation/             # Logique de génération (Best-of-N, prompt engineering)
│
├── archives/                   # Anciens rapports et scripts obsolètes
├── datasets/                   # Dossier contenant les données brutes
├── outputs/                    # Résultats d'exécution et logs générés par Hydra
│
├── train.py                    # Entraînement des classifieurs SVM (Axes 1 & 2)
├── finetune_roberta.py         # Script de fine-tuning de RoBERTa (Axe 2 M4GT)
├── finetune.py                 # Fine-tuning des LLMs (Axe 3 - LoRA/bitsandbytes)
├── generate.py                 # Génération d'arguments et création du rapport final
├── evaluate.py                 # Évaluation et visualisation (PCA / t-SNE)
│
├── scripts/                    # Scripts utilitaires (téléchargement, PDF generation)
├── projetSD.tex                # Code source du rapport en LaTeX
└── requirements.txt            # Liste des dépendances Python
```

---

## 2. Installation de l'environnement

Il est recommandé d'utiliser un environnement virtuel (Python 3.10+).

```bash
# Création de l'environnement virtuel
python -m venv env

# Activation (Windows)
.\env\Scripts\activate
# Activation (Linux / macOS)
source env/bin/activate

# Installation des dépendances
pip install -r requirements.txt
```

*(Note : Pour le fine-tuning avec Mistral, une machine disposant d'un GPU compatible CUDA est nécessaire pour utiliser `bitsandbytes` et `peft`)*.

Pour télécharger et formater les datasets requis par le projet, lancez le script utilitaire suivant :
```bash
python -X utf8 scripts/download_datasets.py --all
```

---

## 3. Commandes d'exécution

Le projet est paramétré à l'aide d'**Hydra**, ce qui permet de modifier les paramètres directement depuis la ligne de commande.

### Axe 1 : Évaluation de la persuasion (Winning Argument Corpus)
Entraînement d'un SVM pour prédire si un argument réussit à changer l'opinion d'un utilisateur.
```bash
python train.py dataset=wac encoder=features
```

### Axe 2 : Détection d'IA (Human vs ChatGPT)
Entraînement de modèles pour différencier les textes humains de ceux générés par IA.
```bash
# Baseline avec SVM et TF-IDF sur le dataset HC3
python train.py dataset=hc3 encoder=tfidf

# Fine-tuning complet de RoBERTa sur le dataset M4GT
python finetune_roberta.py dataset=m4gt
```

### Axe 3 : Génération Stratégique
La génération utilise les modèles des Axes 1 et 2 pour guider la création de texte.

```bash
# 1. Fine-tuning du LLM sur les arguments gagnants
python finetune.py llm=gpt2

# 2. Génération et évaluation Best-of-N
python generate.py strategy=best_of_n \
    llm.model_id=outputs/finetuned_gpt2/ \
    axe1.model_path=axe1_svm_features_wac.pkl \
    axe1.encoder_name=features \
    axe2.model_path=outputs/roberta_finetuned_m4gt/best_model \
    axe2.encoder_name=roberta
```
À l'issue de la génération, un rapport au format HTML (`generation_report.html`) et un fichier CSV sont créés avec les textes produits et leurs scores respectifs.

---

## 4. Équipe

Projet réalisé par :
- **Nicolas CASTEL** 
- **Wassim SMATI** 

Outils d'organisation : Git/GitHub pour le versioning, Discord pour les réunions, et utilisation des serveurs de l'école (`nodecpu01`, `nodemm01`) pour les calculs lourds.


