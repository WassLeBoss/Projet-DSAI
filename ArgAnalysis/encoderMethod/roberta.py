import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
from sentence_transformers import SentenceTransformer

INCLUDE_OP = True 
MODEL_NAME =  "all-roberta-large-v1"  



#dataset
data = pd.read_csv(r"/Users/tristanjin/Documents/TELECOM_PARIS/2A_COURS/SD/Projet-DSAI/Datasets/WinningArgCorpus/WAC.csv")

if INCLUDE_OP:
    data["input_text"] = data["op_text"].fillna("").astype(str) + " " + data["text"].fillna("").astype(str)
else:
    data["input_text"] = data["text"].fillna("").astype(str)

X_raw = data["input_text"].tolist()
y = data["success"].values

print(f"Entrée      : {'op_text + text' if INCLUDE_OP else 'text seul'}")



#modèle d'encodage
model = SentenceTransformer(MODEL_NAME)
X_vectors = model.encode(X_raw, batch_size=32, show_progress_bar=True)




#split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectors, y, test_size=0.2, random_state=42, stratify=y
)


#modèle et évaluation
models = {
    "SVM (Linear)": SVC(kernel="linear", class_weight="balanced"),
}

print("\n Roberta + SVM :")
for name, clf in models.items():
    print(f"\n--- {name} ---")
    clf.fit(X_train, y_train)
    y_pred  = clf.predict(X_test)
    y_score = clf.decision_function(X_test)
    print(classification_report(y_test, y_pred))
    print(f"AUC : {roc_auc_score(y_test, y_score):.4f}")