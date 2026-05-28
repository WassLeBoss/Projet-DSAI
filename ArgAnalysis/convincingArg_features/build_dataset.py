"""
build_dataset.py
----------------
Convertit utterances.jsonl (Winning Argument Corpus) en un CSV prêt pour extract_features().

Colonnes de sortie :
    - reply_id     : identifiant de la réponse
    - op_id        : identifiant du post original
    - op_text      : texte du post original
    - reply_text   : texte de la réponse
    - success      : 1 (delta accordé) ou 0 (pas de delta)

Usage :
    python build_dataset.py --input utterances.jsonl --output dataset.csv
"""

import json
import csv


def load_utterances(path: str) -> tuple[dict, list]:
    """
    Charge le fichier JSONL et sépare OPs et replies.

    Retourne :
        ops     : dict {id -> texte} pour les posts originaux
        replies : liste de dicts pour les réponses avec success non-null
    """
    ops = {}
    replies = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            entry = json.loads(line)
            entry_id = entry.get("id")
            root = entry.get("root")
            reply_to = entry.get("reply-to")
            success = entry["meta"].get("success")
            text = entry.get("text", "")

            # Post original : id == root et pas de reply-to
            if entry_id == root and reply_to is None:
                ops[entry_id] = text

            # Réponse directe à l'OP avec un label valide (0 ou 1)
            elif reply_to is not None and success is not None:
                replies.append({
                    "reply_id": entry_id,
                    "op_id": root,
                    "reply_text": text,
                    "success": int(success),
                })

    return ops, replies


def build_csv(ops: dict, replies: list, output_path: str) -> int:
    """
    Joint chaque reply à son OP et écrit le CSV.
    Retourne le nombre de lignes écrites.
    """
    fieldnames = ["reply_id", "op_id", "op_text", "reply_text", "success"]
    skipped = 0
    written = 0

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for reply in replies:
            op_text = ops.get(reply["op_id"])
            if op_text is None:
                # OP absent du fichier (thread partiel), on ignore
                skipped += 1
                continue

            writer.writerow({
                "reply_id":   reply["reply_id"],
                "op_id":      reply["op_id"],
                "op_text":    op_text,
                "reply_text": reply["reply_text"],
                "success":    reply["success"],
            })
            written += 1

    return written, skipped


def main():
    input_path  = "Datasets/WinningArgCorpus/utterances.jsonl"
    output_path = "Datasets/WinningArgCorpus/dataset.csv"

    print(f"Chargement de {input_path}...")
    ops, replies = load_utterances(input_path)
    print(f"  {len(ops)} posts originaux trouvés")
    print(f"  {len(replies)} réponses avec label non-null")

    print(f"Construction du CSV → {output_path}...")
    written, skipped = build_csv(ops, replies, output_path)
    print(f"  {written} lignes écrites")
    if skipped:
        print(f"  {skipped} réponses ignorées (OP introuvable)")

    print("Terminé.")


if __name__ == "__main__":
    main()