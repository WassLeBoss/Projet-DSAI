import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

nltk.download('wordnet')
nltk.download('stopwords')

INCLUDE_OP = False  

#dataset
data = pd.read_csv(r"/Users/tristanjin/Documents/TELECOM_PARIS/2A_COURS/SD/Projet-DSAI/Datasets/WinningArgCorpus/WAC.csv")

if INCLUDE_OP:
    data["input_text"] = data["op_text"].fillna("").astype(str) + " " + data["text"].fillna("").astype(str)
else:
    data["input_text"] = data["text"].fillna("").astype(str)

X_raw = data["input_text"]
y     = data["success"]

print(f"Entrée : {'op_text + text' if INCLUDE_OP else 'text seul'}")


#preprocessing
lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words("english"))

def clean_text(text):
    tokens = text.lower().split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)

X_clean = X_raw.apply(clean_text)




#TF-IDF
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_vectors = tfidf.fit_transform(X_clean)



#split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectors, y, test_size=0.2, random_state=42, stratify=y
)



#modèle et évaluation
models = {
    "SVM (Linear)": SVC(kernel="linear", class_weight="balanced"),
}

print("TF-IDF :")
for name, model in models.items():
    print(f"\n--- {name} ---")
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_score = model.decision_function(X_test)
    print(classification_report(y_test, y_pred))
    print(f"AUC : {roc_auc_score(y_test, y_score):.4f}")