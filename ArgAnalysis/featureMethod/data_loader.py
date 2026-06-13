import pandas as pd

def load_pairs(csv_path):
    df = pd.read_csv(csv_path)
    pairs = []
    for pair_id, group in df.groupby("pair_id"):
        winners = group[group["success"] == 1.0]
        losers  = group[group["success"] == 0.0]

        if winners.empty or losers.empty:
            continue  # paire incomplète : ignorée

        w = winners.iloc[0]
        l = losers.iloc[0]

        pairs.append({
            "pair_id":     pair_id,
            "winner_text": str(w["text"]),
            "loser_text":  str(l["text"]),
            "op_text":     str(w["op_text"]),
        })

    return pd.DataFrame(pairs)
