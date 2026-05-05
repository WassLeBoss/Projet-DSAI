from convokit import Corpus, download
import shutil

path = download("winning-args-corpus")
print(path)

shutil.copytree(
    path,
    r"C:\Users\Wess9\Desktop\ProjetDSAI\Projet-DSAI\Datasets\WinningArgCorpus",
    dirs_exist_ok=True
)