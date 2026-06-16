"""
Point d'entree — Fine-tuning du LLM sur les arguments gagnants du WAC.

Utilisation :
    # Fine-tune GPT-2 (CPU-friendly, ~2-3h)
    python finetune.py llm=gpt2

    # Fine-tune Mistral 7B avec LoRA (GPU requis)
    python finetune.py llm=mistral

    # Afficher la config sans lancer
    python finetune.py --cfg job
"""

import logging

import hydra
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs/generation", config_name="config_generate")
def main(cfg: DictConfig) -> None:
    log.info("=" * 60)
    log.info("Fine-tuning — LLM : %s", cfg.llm.name)
    log.info("Dataset WAC   : %s", cfg.wac_csv_path)
    log.info("=" * 60)

    # ── Chargement du WAC ────────────────────────────────────────────────────
    log.info("Chargement du WAC...")
    df = pd.read_csv(cfg.wac_csv_path)

    # Garde uniquement les arguments gagnants (success=1)
    winners = df[df["success"] == 1].copy()
    log.info("%d arguments gagnants disponibles pour le fine-tuning", len(winners))

    # Construction des paires (op_text, winner_text) par pair_id
    pairs = []
    for pair_id, group in df.groupby("pair_id"):
        w_rows = group[group["success"] == 1]
        if w_rows.empty:
            continue
        w = w_rows.iloc[0]
        pairs.append({
            "op_text":     str(w["op_text"]),
            "winner_text": str(w["text"]),
        })

    pairs_df = pd.DataFrame(pairs)
    log.info("%d paires (op, winner) construites", len(pairs_df))

    # Split train / validation
    train_pairs, val_pairs = train_test_split(
        pairs_df, test_size=0.1, random_state=cfg.seed
    )

    # ── Fine-tuning ──────────────────────────────────────────────────────────
    from src.generation.finetuning import WACFinetuner

    output_dir = f"finetuned_{cfg.llm.name}"
    finetuner  = WACFinetuner(cfg.llm)
    finetuner.train(train_pairs, output_dir=output_dir)
    finetuner.save(output_dir)

    log.info("Modele fine-tune disponible dans : %s/", output_dir)
    log.info("Pour generer des arguments : python generate.py llm.model_id=%s", output_dir)


if __name__ == "__main__":
    main()
