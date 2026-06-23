# Projet DSAI : Persuasion, Détection d'IA et Génération d'arguments

Ce dépôt contient le code source de notre projet réalisé dans le cadre de la filière DSAI. Le projet porte sur l'analyse d'arguments (dataset WAC), la détection de textes générés par IA (datasets GRID, HC3 & M4GT) et la génération de discours persuasif à l'aide de Modèles de Langage (LLMs).

Projet supervisé par Mathieu LABEAU et Pierre FIHEY. 

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

Pour télécharger et formater les datasets requis :
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
- **Tristan JIN** 
- **Wassim SMATI** 

Contributions : 

Wassim SMATI: I was responsible for Axis 2 (AI-generated text detection) and Axis
3 (text generation). I designed the project’s Hydra-based architecture and configuration
management system, ensuring reproducibility and scalability across experiments. For Axis
2, I developed and evaluated AI-generated text detectors on the HC3, GriD, and M4GT
datasets, implemented and compared multiple text representations (TF-IDF, Word2Vec,
and RoBERTa-based Sentence-Transformers embeddings), and built a RoBERTa fine-
tuning pipeline using the Hugging Face Trainer API. For Axis 3, I fine-tuned GPT-2 and
Qwen 2.5 3B models using QLoRA under hardware constraints, and implemented both
Best-of-N generation and prompt engineering strategies. I also developed interpretability
tools, including t-SNE and UMAP visualizations to compare generated and human-written
texts.

• Tristan JIN: I was mainly responsible for Axis 1: searching for papers and the WAC
dataset; proposing the simplification of Axis 1; exploring the dataset and a first failed
cleaning attempt. Implementation of the Encoder (W2V, TF-IDF, RoBERTa) + SVM
method (tested with LogReg initially) with and without OP; implementation of the paper’s
features and SHAP analysis; implementation of the pairwise method; failed attempt at
RoBERTa fine-tuning.

• Yanis DAHASSE: In this project, I focused mainly on Axis 1. I handled the cleaning and
formatting of the CMV dataset, as well as data preparation , train/validation/test splitting
by post identifier and position bias handling. On the modelling side, I performed the fine-
tuning of RoBERTa in Cross-Encoder architecture, which constitutes our best result on this
task, and explored several DeBERTa approaches that did not succeed within the allotted
time. In parallel, I conducted a literature review of reference works and developed the
project timeline.

Outils d'organisation : GitHub et utilisation des serveurs gpu de l'école pour les calculs lourds.


