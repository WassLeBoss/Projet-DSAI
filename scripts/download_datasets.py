# -*- coding: utf-8 -*-
"""
Scripts de telechargement des datasets.

Usage :
    # Telecharger HC3 (reddit_eli5)
    python scripts/download_datasets.py --dataset hc3

    # Telecharger le WinningArgCorpus via ConvoKit
    python scripts/download_datasets.py --dataset wac

    # Telecharger tous les datasets
    python scripts/download_datasets.py --all
"""

import argparse
import os
import sys

# Force UTF-8 sur Windows pour eviter les erreurs d'encodage
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def download_hc3(save_dir: str = "datasets/HC3") -> None:
    """
    Telecharge le dataset HC3 (reddit_eli5) depuis HuggingFace
    et le sauvegarde localement en CSV.

    Compatible avec datasets >= 5.0 (utilise huggingface_hub directement).
    """
    import glob

    import pandas as pd
    from huggingface_hub import snapshot_download

    print("[...] Telechargement de HC3 (reddit_eli5) via huggingface_hub...")

    # Telecharge les fichiers bruts du repo (parquet)
    local_dir = snapshot_download(
        repo_id="Hello-SimpleAI/HC3",
        repo_type="dataset",
        allow_patterns=["reddit_eli5*"],
        local_dir=os.path.join(save_dir, "raw"),
    )
    print(f"[OK]  Fichiers bruts -> {local_dir}")

    # Cherche les fichiers parquet telecharges
    parquet_files = glob.glob(
        os.path.join(local_dir, "**", "*.parquet"), recursive=True
    )
    if not parquet_files:
        # Fallback : cherche aussi des JSON/JSONL
        parquet_files = glob.glob(
            os.path.join(local_dir, "**", "*.json*"), recursive=True
        )

    if not parquet_files:
        print("[WARN] Aucun fichier parquet trouve. Contenu du dossier :")
        for f in os.listdir(local_dir):
            print("  -", f)
        return

    os.makedirs(save_dir, exist_ok=True)

    # Charge et concatene tous les fichiers
    dfs = []
    for f in parquet_files:
        if f.endswith(".parquet"):
            dfs.append(pd.read_parquet(f))
        elif f.endswith((".json", ".jsonl")):
            dfs.append(pd.read_json(f, lines=f.endswith(".jsonl")))

    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        out_path = os.path.join(save_dir, "train.csv")
        df.to_csv(out_path, index=False)
        print(f"[OK]  train.csv -> {out_path}  ({len(df)} lignes)")
        print("Colonnes disponibles :", df.columns.tolist())
    else:
        print("[ERROR] Impossible de charger les fichiers.")


def download_wac(save_dir: str = "datasets/WinningArgCorpus") -> None:
    """
    Telecharge le WinningArgCorpus via ConvoKit et le convertit en WAC.csv.

    Colonnes produites :
        pair_id  : identifiant de la paire de commentaires
        text     : texte du commentaire
        op_text  : texte du post original (OP)
        success  : 1 si le commentaire a convaincu, 0 sinon
    """
    import pandas as pd
    from convokit import Corpus, download

    print("[...] Telechargement du WinningArgCorpus via ConvoKit...")
    corpus = Corpus(filename=download("winning-args-corpus"))
    print("[OK]  Corpus charge.")

    os.makedirs(save_dir, exist_ok=True)

    rows = []
    for convo in corpus.iter_conversations():
        op_utt = convo.get_utterance(convo.get_utterance_ids()[0])
        op_text = op_utt.text or ""
        pair_id = convo.id

        for utt in convo.iter_utterances():
            if utt.id == op_utt.id:
                continue  # skip le post OP lui-meme
            meta = utt.meta or {}
            success = meta.get("success", None)
            if success is None:
                continue
            rows.append(
                {
                    "pair_id": pair_id,
                    "text": utt.text or "",
                    "op_text": op_text,
                    "success": int(success),
                }
            )

    df = pd.DataFrame(rows)
    out_path = os.path.join(save_dir, "WAC.csv")
    df.to_csv(out_path, index=False)
    print(
        f"[OK]  WAC.csv -> {out_path}  ({len(df)} lignes, {df['success'].sum()} succes)"
    )
    print("Colonnes :", df.columns.tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Telechargement des datasets du projet."
    )
    parser.add_argument(
        "--dataset", choices=["hc3", "wac"], help="Dataset a telecharger"
    )
    parser.add_argument(
        "--all", action="store_true", help="Telecharger tous les datasets"
    )
    args = parser.parse_args()

    if args.all or args.dataset == "hc3":
        download_hc3()

    if args.all or args.dataset == "wac":
        download_wac()

    if not args.all and not args.dataset:
        parser.print_help()
