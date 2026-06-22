"""
Point d'entree — Generation d'arguments convaincants (Axe 3) sans evaluation.

Necessite :
    - Le SVM Axe 1 entraine et sauvegarde (via train.py)
    - Le LLM fine-tune (via finetune.py) OU GPT-2 pre-entraine pour les tests

Utilisation :

    # Strategie 1 : Best-of-N avec LLM fine-tune
    python generate_csv.py strategy=best_of_n \\
        axe1.model_path=outputs/YYYY-MM-DD/.../axe1_svm_features_wac.pkl

    # Strategie 2 : Prompt engineering (zero-shot)
    python generate_csv.py strategy=prompt_eng \\
        axe1.model_path=outputs/YYYY-MM-DD/.../axe1_svm_features_wac.pkl
"""

import logging
import pickle

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def _load_axe1_model(cfg: DictConfig):
    """Charge le SVM Axe 1 ou RoBERTa depuis le chemin specifie dans la config."""
    model_path = cfg.axe1.model_path
    log.info("Chargement modele Axe 1 : %s", model_path)
    
    if model_path.endswith(".pt") or model_path.endswith(".bin") or "roberta" in cfg.axe1.encoder_name.lower():
        import torch
        from transformers import RobertaForMultipleChoice, RobertaTokenizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Chargement de RoBERTa fine-tuned pour l'Axe 1 sur %s...", device)
        
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForMultipleChoice.from_pretrained('roberta-base')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        return {"model": model, "tokenizer": tokenizer, "type": "roberta_finetuned"}
        
    with open(model_path, "rb") as f:
        clf = pickle.load(f)
    return clf


@hydra.main(version_base=None, config_path="configs/generation", config_name="config_generate")
def main(cfg: DictConfig) -> None:
    log.info("=" * 60)
    log.info("Generation Axe 3 (CSV uniquement)")
    log.info("LLM      : %s", cfg.llm.name)
    log.info("Strategie: %s", cfg.strategy.name)
    log.info("=" * 60)

    # ── Chargement des modeles ───────────────────────────────────────────────
    # On desactive la contrainte sur axe2.model_path car on n'evalue pas l'IA
    OmegaConf.set_struct(cfg, False)
    if "axe2" in cfg:
        cfg.axe2.model_path = None

    # Axe 1 est charge car necessaire pour les strategies best_of_n et prompt_eng
    axe1_clf = _load_axe1_model(cfg)

    # ── Chargement des OPs de test depuis le WAC ─────────────────────────────
    log.info("Chargement des OPs depuis : %s", cfg.wac_csv_path)
    df = pd.read_csv(cfg.wac_csv_path)

    # On prend des OPs uniques (une par conversation)
    op_texts = (
        df.groupby("pair_id")["op_text"]
        .first()
        .dropna()
        .sample(n=min(cfg.n_op_samples, len(df["pair_id"].unique())),
                random_state=cfg.seed)
        .tolist()
    )
    log.info("%d OPs selectionnes pour la generation", len(op_texts))

    # ── Initialisation du generateur ─────────────────────────────────────────
    from src.generation.generator import ArgumentGenerator
    generator = ArgumentGenerator(cfg.llm.model_id, cfg.llm)

    # ── Strategie de generation ───────────────────────────────────────────────
    results = []

    if cfg.strategy.name == "best_of_n":
        from src.generation.best_of_n import BestOfNSelector
        # Injecte le nom de l'encodeur Axe 1 dans la config strategie
        strategy_cfg = OmegaConf.to_container(cfg.strategy, resolve=True)
        strategy_cfg["axe1_encoder_name"] = cfg.axe1.encoder_name
        selector = BestOfNSelector(axe1_clf, None, OmegaConf.create(strategy_cfg))

        for i, op_text in enumerate(op_texts):
            log.info("[%d/%d] Generation Best-of-%d...", i + 1, len(op_texts), cfg.strategy.n_candidates)
            candidates = generator.generate_n(op_text, n=cfg.strategy.n_candidates)
            best, scores = selector.select(op_text, candidates)
            results.append((op_text, best))

    elif cfg.strategy.name == "prompt_eng":
        from src.generation.prompt_engineering import PromptEngineer
        from src.features.pairwise import get_feature_names

        feature_names = get_feature_names()
        pe = PromptEngineer(axe1_clf, feature_names, cfg.strategy)

        log.info("Strategies identifiees depuis Axe 1 :")
        for feat, instr in pe.get_strategy_summary().items():
            log.info("  [%s] -> %s", feat, instr)

        for i, op_text in enumerate(op_texts):
            log.info("[%d/%d] Generation avec prompt engineering...", i + 1, len(op_texts))
            engineered_prompt = pe.build_prompt(op_text)
            candidates = generator.generate_n(engineered_prompt, n=cfg.strategy.n_candidates)

            # Best-of-N parmi les candidats generes avec prompt engineering
            from src.generation.best_of_n import BestOfNSelector
            strategy_cfg = OmegaConf.create({"axe1_encoder_name": cfg.axe1.encoder_name, "verbose": False})
            selector = BestOfNSelector(axe1_clf, None, strategy_cfg)
            best, _ = selector.select(op_text, candidates)
            results.append((op_text, best))

    else:
        raise ValueError(f"Strategie inconnue : {cfg.strategy.name}")

    # ── Sauvegarde des resultats ──────────────────────────────────────────────
    log.info("\n%s\nSAUVEGARDE DES ARGUMENTS GENERES\n%s", "=" * 60, "=" * 60)
    report_df = pd.DataFrame(results, columns=["op_text", "generated_text"])

    # Sauvegarde du rapport CSV simple
    report_path = "generated_arguments.csv"
    report_df.to_csv(report_path, index=False)
    log.info("Arguments generes sauvegardes -> %s", report_path)

    # Affichage de quelques exemples
    log.info("\n--- Exemples generes ---")
    for _, row in report_df.head(3).iterrows():
        log.info("OP : %s", row["op_text"])
        log.info("GEN : %s\n", row["generated_text"])


if __name__ == "__main__":
    main()
