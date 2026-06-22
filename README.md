# Projet-DSAI : Persuasion, Détection d'IA et Génération Stratégique

Projet de la filière DSAI — Analyse d'arguments, détection de textes synthétiques et génération par Modèles de Langage (LLMs).  
En collaboration avec M. Tristan JIN.

---

## Structure du projet

```text
Projet-DSAI/
│
├── configs/                    # Configs Hydra (YAML)
│   ├── config.yaml             # Config principale Axes 1 & 2
│   ├── dataset/                # wac, grid, hc3, m4gt
│   ├── encoder/                # tfidf, w2v, roberta, features
│   ├── model/                  # svm
│   └── generation/             # Configs pour l'Axe 3 (Génération)
│       ├── config_generate.yaml
│       ├── llm/                # gpt2, mistral
│       └── strategy/           # best_of_n, prompt_eng
│
├── src/
│   ├── data/                   # Chargement & preprocessing
│   ├── features/               # Features stylistiques & pairwise (Jaccard)
│   ├── encoders/               # TF-IDF, W2V, RoBERTa, Features
│   ├── models/                 # SVM classifier
│   └── generation/             # Génération (LLMs), Fine-tuning (QLoRA), Evaluateurs
│
├── datasets/                   # Données brutes (non versionnées)
│
├── outputs/                    # Résultats, Logs et CSV générés par Hydra
│
├── train.py                    # Entraînement Axes 1 & 2 (SVM)
├── evaluate.py                 # Évaluation et visualisation (t-SNE)
├── finetune.py                 # Entraînement LLM Axe 3 (LoRA / bitsandbytes)
├── generate.py                 # Génération Best-of-N et création du rapport CSV/HTML
├── rapport_projet_dsai.tex     # Rapport académique final en LaTeX
└── requirements.txt
```

---

## Installation et Environnement

```bash
# 1. Créer l'environnement virtuel
python -m venv env
source env/bin/activate  # Sur Windows : .\env\Scripts\activate

# 2. Installer les dépendances
pip install -r requirements.txt
```
*(Note : Pour le fine-tuning Mistral sur GPU, assurez-vous d'avoir une machine compatible CUDA car le package `bitsandbytes` sera utilisé).*

---

## Utilisation et Exécution (Pipeline Complet)

Le projet utilise **Hydra** pour une configuration dynamique et sans hardcoding.

### 1. Axe 1 et 2 : Entraînement des Classifieurs (SVM)

```bash
# Axe 1 : Prédire les arguments gagnants (WAC) avec les features stylistiques
python train.py dataset=wac encoder=features

# Axe 2 : Détecter les IA (HC3) avec l'encodeur RoBERTa
python train.py dataset=hc3 encoder=roberta
```
Les modèles entraînés (`.pkl`) seront sauvegardés dans le dossier d'exécution et devront être utilisés pour l'Axe 3.

### 2. Axe 3 : Fine-Tuning du LLM

```bash
# Fine-tuner un LLM (Recommandé sur nœud GPU ex: nodemm01)
python finetune.py llm=gpt2
# ou pour Mistral avec QLoRA : python finetune.py llm=mistral
```

### 3. Axe 3 : Génération Stratégique et Évaluation
```bash
# Générer des arguments et les évaluer automatiquement
python generate.py strategy=best_of_n \
    llm.model_id=outputs/finetuned_gpt2/ \
    axe1.model_path=axe1_svm_features_wac.pkl \
    axe1.encoder_name=features \
    axe2.model_path=axe2_svm_roberta_hc3.pkl \
    axe2.encoder_name=roberta
```
> **Rapport :** À la fin du script, un fichier `generation_report.csv` et un fichier visuel `generation_report.html` seront générés contenant les textes créés, leurs scores de persuasion (Axe 1) et leurs scores d'authenticité humaine (Axe 2).

---

## Contributions de l'Équipe

*Veuillez remplacer les crochets par les prénoms, noms et tâches spécifiques de chaque membre de l'équipe.*

- **[Prénom Nom] :** [À compléter - Ex: Conception de l'Axe 1, feature engineering, intégration VADER, optimisation RAM (scipy.sparse).]
- **[Prénom Nom] :** [À compléter - Ex: Conception de l'Axe 2, implémentation des encodeurs RoBERTa/W2V, visualisation t-SNE, gestion du dépôt Git.]
- **Wassim [Nom de Famille] :** [À compléter - Ex: Responsable de l'Axe 3, développement du pipeline QLoRA pour Mistral, génération Best-of-N, évaluation couplée et rédaction du rapport final LaTeX.]

**Organisation du groupe :**
- Réunions synchrones via Discord/Teams.
- Suivi du code et gestion de version via Git/GitHub.
- Expérimentations lourdes mutualisées sur les serveurs de l'école (`nodecpu01`, `nodemm01`) via `tmux` et `nohup`.
