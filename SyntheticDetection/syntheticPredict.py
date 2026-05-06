import re 
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.preprocessing import normalize

df = pd.read_csv("C:\\Users\\Wess9\\Desktop\\ProjetDSAI\\Projet-DSAI\\Datasets\\GriD\\reddit_filtered_dataset.csv")
data = df["Data"].astype(str).to_list()
labels = df["Labels"].astype(int).to_list()

model = SentenceTransformer('all-MiniLM-L6-v2')

data = [re.sub(r"\.(?=[A-Z])", ". ", sentence) for sentence in data]
encoded_data = model.encode(data, show_progress_bar=True)
encoded_data = normalize(encoded_data)

X_train, X_test, y_train, y_test = train_test_split(
    encoded_data, labels, test_size=0.2, random_state=42
)

clf = SVC(kernel="linear")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))