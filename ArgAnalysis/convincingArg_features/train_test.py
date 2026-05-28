"""
train_model.py
--------------
1. Charge le dataset CSV (produit par build_dataset.py)
2. Applique extract_features() sur chaque paire (op_text, reply_text)
3. Entraîne un SVM et une Régression Logistique
4. Affiche les performances et sauvegarde le meilleur modèle

Usage :
    python3 train_model.py
"""

import re
import csv
import nltk
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

# ── Téléchargements NLTK ────────────────────────────────────────────────────
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# ── Chemins ─────────────────────────────────────────────────────────────────
DATASET_PATH = "Datasets/WinningArgCorpus/dataset.csv"
# ── Constantes linguistiques ─────────────────────────────────────────────────
STOP_WORDS = set(stopwords.words('english'))
HEDGES     = {"could", "would", "may", "might", "perhaps", "possibly",
              "probably", "seem", "seems", "appear", "appears"}
EXAMPLES   = {"for example", "for instance", "e.g."}

# ── Fonctions utilitaires ────────────────────────────────────────────────────
def safe_divide(n, d):
    return n / d if d > 0 else 0.0

def get_word_sets(text):
    if not isinstance(text, str):
        return set(), set(), 0
    words       = word_tokenize(text.lower())
    words_alpha = {w for w in words if w.isalnum()}
    stop_set    = words_alpha.intersection(STOP_WORDS)
    content_set = words_alpha.difference(STOP_WORDS)
    return stop_set, content_set, len(words_alpha)

def extract_features(op_text, reply_text):
    features = {}
    if not isinstance(reply_text, str) or not isinstance(op_text, str):
        return features

    op_stops, op_content, op_len     = get_word_sets(op_text)
    rep_stops, rep_content, rep_len  = get_word_sets(reply_text)

    features['word_count']       = rep_len

    inter_stops  = rep_stops.intersection(op_stops)
    union_stops  = rep_stops.union(op_stops)
    features['jaccard_stopwords'] = safe_divide(len(inter_stops), len(union_stops))

    inter_content = rep_content.intersection(op_content)
    union_content = rep_content.union(op_content)
    features['jaccard_content']   = safe_divide(len(inter_content), len(union_content))

    links = re.findall(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        reply_text
    )
    features['num_links']         = len(links)
    features['has_pdf_link']      = int(any(".pdf" in l.lower() for l in links))

    reply_lower = reply_text.lower()
    features['num_examples']      = sum(reply_lower.count(ex) for ex in EXAMPLES)

    features['num_hedges']        = sum(1 for w in rep_content if w in HEDGES)

    num_questions = reply_text.count('?')
    features['question_ratio']    = safe_divide(num_questions, rep_len)

    features['links_per_word']    = safe_divide(features['num_links'],   rep_len)
    features['hedges_per_word']   = safe_divide(features['num_hedges'],  rep_len)

    return features

# ── Chargement du dataset ────────────────────────────────────────────────────
def load_dataset(path):
    X_raw, y = [], []
    skipped  = 0

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            feats = extract_features(row["op_text"], row["reply_text"])
            if not feats:
                skipped += 1
                continue
            X_raw.append(feats)
            y.append(int(row["success"]))

    if skipped:
        print(f"  {skipped} lignes ignorées (texte manquant)")

    # Convertit en matrice numpy en garantissant l'ordre des colonnes
    feature_names = list(X_raw[0].keys())
    X = np.array([[row[k] for k in feature_names] for row in X_raw])
    return X, np.array(y), feature_names

# ── Entraînement & évaluation ────────────────────────────────────────────────
def evaluate_model(name, pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n--- {name} ---")
    print(f"{'':>4}  {'precision':>10}  {'recall':>7}  {'f1-score':>9}  {'support':>8}")
    print()
    for label, display in [("0", "0"), ("1", "1")]:
        r = report[label]
        print(f"  {display:>4}  {r['precision']:>10.2f}  {r['recall']:>7.2f}  {r['f1-score']:>9.2f}  {int(r['support']):>8}")
    print()
    print(f"  {'accuracy':>12}  {'':>7}  {report['accuracy']:>9.2f}  {int(report['weighted avg']['support']):>8}")
    print(f"  {'macro avg':>12}  {report['macro avg']['precision']:>7.2f}  {report['macro avg']['recall']:>7.2f}  {report['macro avg']['f1-score']:>9.2f}  {int(report['macro avg']['support']):>8}")
    print(f"  {'weighted avg':>12}  {report['weighted avg']['precision']:>7.2f}  {report['weighted avg']['recall']:>7.2f}  {report['weighted avg']['f1-score']:>9.2f}  {int(report['weighted avg']['support']):>8}")

    cv_scores = cross_val_score(pipeline,
                                np.vstack([X_train, X_test]),
                                np.concatenate([y_train, y_test]),
                                cv=5, scoring="f1")
    print(f"\n  F1 cross-val (5 folds) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return cv_scores.mean()

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"Chargement du dataset : {DATASET_PATH}")
    X, y, feature_names = load_dataset(DATASET_PATH)
    print(f"  {len(y)} exemples chargés  |  features : {feature_names}")
    print(f"  Distribution : {int(y.sum())} positifs / {int((y==0).sum())} négatifs")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(kernel="rbf", class_weight="balanced", random_state=42))
        ]),
    }

    best_name, best_score = None, -1

    for name, pipeline in models.items():
        score = evaluate_model(name, pipeline, X_train, X_test, y_train, y_test)
        if score > best_score:
            best_score = score
            best_name  = name

    print(f"\n✓ Meilleur modèle : {best_name} (F1 = {best_score:.4f})")

if __name__ == "__main__":
    main()