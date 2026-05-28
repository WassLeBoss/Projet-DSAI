import re
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from gensim.models import Word2Vec
import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return tokens 

#datset
data = pd.read_csv("Datasets/WinningArgCorpus/wac_v2.csv")
X = data["text"].astype(str)
y = data["label"]

sentences = X.apply(clean_text).tolist()

#Word2Vec
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4, seed=42)

def get_mean_vector(sentence, model, vector_size):
    vectors = [model.wv[word] for word in sentence if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(vector_size)

X_vectors = np.array([get_mean_vector(sent, w2v_model, 100) for sent in sentences])
X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, test_size=0.2, random_state=42)

#modele, entrainement et évaluation
models = {
    "SVM (RBF)": SVC(kernel="rbf",class_weight="balanced"),
}

print("Word2Vec :")
for name, model in models.items():
    print(f"\n--- {name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))