import json
import csv
 
input_path = "Datasets/WinningArgCorpus/utterances.jsonl"   
output_path = "Datasets/WinningArgCorpus/wac_v2.csv"
 
rows = []
 
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        text = entry.get("text", "")
        success = entry.get("meta", {}).get("success", None)
 
        # On garde uniquement les entrées avec un label défini (0 ou 1)
        if success is not None:
            rows.append({"text": text, "label": int(success)})
 
with open(output_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["text", "label"])
    writer.writeheader()
    writer.writerows(rows)
 
print(f"Done. {len(rows)} rows exported.")
label_1 = sum(1 for r in rows if r["label"] == 1)
label_0 = sum(1 for r in rows if r["label"] == 0)
print(f"  label=1 (winning): {label_1}")
print(f"  label=0 (losing):  {label_0}")