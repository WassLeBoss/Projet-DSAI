import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer

INCLUDE_OP = True
CSV_PATH   = "/Users/tristanjin/Documents/TELECOM_PARIS/2A_COURS/SD/Projet-DSAI/Datasets/WinningArgCorpus/WAC.csv"
ENCODER    = "roberta"  # "roberta", "w2v" ou "tfidf"

data = pd.read_csv(CSV_PATH)
data["input_text"] = (data["op_text"].fillna("") + " " + data["text"].fillna("")) if INCLUDE_OP else data["text"].fillna("")
texts = data["input_text"].tolist()
y     = data["success"].values

if ENCODER == "roberta":
    from sentence_transformers import SentenceTransformer
    X = SentenceTransformer("all-roberta-large-v1").encode(texts, batch_size=32, show_progress_bar=True)

elif ENCODER == "w2v":
    import gensim.downloader as api
    w2v = api.load("word2vec-google-news-300")
    def sentence_vector(text):
        tokens = [w for w in text.lower().split() if w in w2v]
        return np.mean(w2v[tokens], axis=0) if tokens else np.zeros(300)
    X = np.array([sentence_vector(t) for t in texts])

elif ENCODER == "tfidf":
    X = TfidfVectorizer(max_features=5000).fit_transform(texts).toarray()


coords = PCA(n_components=2).fit_transform(X)
title  = f"PCA  —  {ENCODER.upper()}"

plt.figure(figsize=(8, 6))
for label, color, name in [(1, "steelblue", "Succès"), (0, "tomato", "Échec")]:
    mask = y == label
    plt.scatter(coords[mask, 0], coords[mask, 1], c=color, s=10, alpha=0.5, label=name)
plt.title(title)
plt.legend()
plt.tight_layout()
plt.savefig("visualize.png", dpi=150)
plt.show()