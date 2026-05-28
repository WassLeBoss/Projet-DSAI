import re
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer

#prétraitement
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)

#dataset
data = pd.read_csv("Datasets/WinningArgCorpus/wac_v2.csv")
data["text_clean"] = data["text"].astype(str).apply(clean_text)

X = data["text_clean"]
y = data["label"]

#BERT
# hargement d'un modèle BERT optimisé pour générer des embeddings de phrases.
print("Chargement du modèle BERT et encodage des textes (cela peut prendre un moment)...")
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# L'encodage transforme notre liste de textes en une matrice dense de vecteurs
X_bert = bert_model.encode(X.tolist(), show_progress_bar=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_bert, y, test_size=0.2, random_state=42
)

#modele, entrainement et évaluation
models = {
    "SVM (RBF)": SVC(kernel="rbf",class_weight="balanced")
}

print("\nBERT :")
for name, model in models.items():
    print(f"\n--- {name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))