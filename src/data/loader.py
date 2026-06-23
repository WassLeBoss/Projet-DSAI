"""Dataset loaders for WAC, GriD and HC3 datasets."""

import os

import pandas as pd
from omegaconf import DictConfig


def _get_wac_path(cfg: DictConfig) -> str:
    csv_path = cfg.dataset.csv_path
    if not os.path.exists(csv_path):
        if csv_path.endswith("WAC.csv"):
            alt_path = csv_path.replace("WAC.csv", "WAC_final.csv")
            if os.path.exists(alt_path):
                return alt_path
        elif csv_path.endswith("WAC_final.csv"):
            alt_path = csv_path.replace("WAC_final.csv", "WAC.csv")
            if os.path.exists(alt_path):
                return alt_path
    return csv_path


def load_wac(cfg: DictConfig) -> tuple[list[str], list[int]]:
    """Charge le WinningArgCorpus (WAC)."""
    csv_path = _get_wac_path(cfg)
    df = pd.read_csv(csv_path)

    if cfg.include_op:
        texts = (
            df[cfg.dataset.op_col].fillna("").astype(str)
            + " "
            + df[cfg.dataset.text_col].fillna("").astype(str)
        ).tolist()
    else:
        texts = df[cfg.dataset.text_col].fillna("").astype(str).tolist()

    labels = df[cfg.dataset.label_col].astype(int).tolist()
    return texts, labels


def load_wac_pairs(cfg: DictConfig) -> pd.DataFrame:
    """Charge le WAC sous forme de paires (winner, loser) pour le mode pairwise."""
    csv_path = _get_wac_path(cfg)
    df = pd.read_csv(csv_path)
    pairs = []

    for pair_id, group in df.groupby("pair_id"):
        winners = group[group[cfg.dataset.label_col] == 1.0]
        losers = group[group[cfg.dataset.label_col] == 0.0]

        if winners.empty or losers.empty:
            continue  # paire incomplète : ignorée

        w = winners.iloc[0]
        l = losers.iloc[0]

        pairs.append(
            {
                "pair_id": pair_id,
                "winner_text": str(w[cfg.dataset.text_col]),
                "loser_text": str(l[cfg.dataset.text_col]),
                "op_text": str(w[cfg.dataset.op_col]),
            }
        )

    return pd.DataFrame(pairs)


def load_grid(cfg: DictConfig) -> tuple[list[str], list[int]]:
    """Charge le dataset GriD (détection de textes synthétiques)."""
    df = pd.read_csv(cfg.dataset.csv_path)
    texts = df[cfg.dataset.text_col].astype(str).tolist()
    labels = df[cfg.dataset.label_col].astype(int).tolist()
    return texts, labels


def load_hc3(cfg: DictConfig) -> tuple[list[str], list[int]]:
    """Charge le dataset HC3 (Human ChatGPT Comparison Corpus — reddit_eli5)."""
    import ast

    df = pd.read_csv(cfg.dataset.csv_path)
    texts, labels = [], []

    for _, row in df.iterrows():
        # human_answers peut être une liste sérialisée en string
        human_answers = row.get("human_answers", [])
        chatgpt_answers = row.get("chatgpt_answers", [])

        if isinstance(human_answers, str):
            try:
                human_answers = ast.literal_eval(human_answers)
            except (ValueError, SyntaxError):
                human_answers = [human_answers]

        if isinstance(chatgpt_answers, str):
            try:
                chatgpt_answers = ast.literal_eval(chatgpt_answers)
            except (ValueError, SyntaxError):
                chatgpt_answers = [chatgpt_answers]

        for ans in human_answers:
            if ans and str(ans).strip():
                texts.append(str(ans).strip())
                labels.append(0)  # humain

        for ans in chatgpt_answers:
            if ans and str(ans).strip():
                texts.append(str(ans).strip())
                labels.append(1)  # ChatGPT

    return texts, labels


def load_m4gt(cfg: DictConfig) -> tuple[list[str], list[int]]:
    """Charge le dataset M4GT-Bench (SubtaskA.jsonl)."""
    import json

    texts, labels = [], []
    max_samples = cfg.dataset.get("max_samples", None)

    with open(cfg.dataset.path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                data = json.loads(line)
                texts.append(str(data[cfg.dataset.text_col]))
                labels.append(int(data[cfg.dataset.label_col]))
            except (json.JSONDecodeError, KeyError):
                continue

    return texts, labels
