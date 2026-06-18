"""
Point d'entree — Generation d'arguments convaincants (Axe 3).

Necessite :
    - Le SVM Axe 1 entraine et sauvegarde (via train.py)
    - Le LLM fine-tune (via finetune.py) OU GPT-2 pre-entraine pour les tests

Utilisation :

    # Strategie 1 : Best-of-N avec LLM fine-tune
    python generate.py strategy=best_of_n \\
        axe1.model_path=outputs/YYYY-MM-DD/.../axe1_svm_features_wac.pkl

    # Strategie 2 : Prompt engineering (zero-shot)
    python generate.py strategy=prompt_eng \\
        axe1.model_path=outputs/YYYY-MM-DD/.../axe1_svm_features_wac.pkl

    # Utiliser un modele fine-tune specifique
    python generate.py strategy=best_of_n llm.model_id=finetuned_gpt2/

    # Afficher la config sans lancer
    python generate.py --cfg job
"""

import logging
import pickle

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def _load_axe1_model(cfg: DictConfig):
    """Charge le SVM Axe 1 depuis le chemin specifie dans la config."""
    model_path = cfg.axe1.model_path
    log.info("Chargement modele Axe 1 : %s", model_path)
    with open(model_path, "rb") as f:
        clf = pickle.load(f)
    return clf


def _load_axe2_model(cfg: DictConfig):
    """Charge le SVM Axe 2 depuis le chemin specifie dans la config."""
    model_path = cfg.axe2.model_path
    log.info("Chargement modele Axe 2 : %s", model_path)
    with open(model_path, "rb") as f:
        clf = pickle.load(f)
    return clf


@hydra.main(version_base=None, config_path="configs/generation", config_name="config_generate")
def main(cfg: DictConfig) -> None:
    log.info("=" * 60)
    log.info("Generation Axe 3")
    log.info("LLM      : %s", cfg.llm.name)
    log.info("Strategie: %s", cfg.strategy.name)
    log.info("=" * 60)

    # ── Chargement des modeles Axes 1 & 2 ────────────────────────────────────
    axe1_clf = _load_axe1_model(cfg)
    axe2_clf = _load_axe2_model(cfg)

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

    # Vrais winners pour le delta de features dans l'evaluation
    reference_winners = df[df["success"] == 1]["text"].dropna().tolist()[:200]

    # ── Initialisation du generateur ─────────────────────────────────────────
    from src.generation.generator import ArgumentGenerator
    generator = ArgumentGenerator(cfg.llm.model_id, cfg.llm)

    # ── Evaluateur ───────────────────────────────────────────────────────────
    from src.generation.evaluator import ArgumentEvaluator
    evaluator = ArgumentEvaluator(
        axe1_clf=axe1_clf,
        axe2_clf=axe2_clf,
        axe1_enc_name=cfg.axe1.encoder_name,
        axe2_enc_name=cfg.axe2.encoder_name,
    )

    # ── Test d'evaluation rapide (fail-fast) ─────────────────────────────────
    log.info("Verification de la compatibilite des modeles (dry-run)...")
    try:
        evaluator.evaluate("Test OP", "Test Generated")
        log.info("Verification OK ! Les dimensions correspondent.")
    except Exception as e:
        log.error("ERREUR CRITIQUE : Les dimensions des modeles ne correspondent pas !")
        log.error("Details: %s", str(e))
        raise RuntimeError("Echec du dry-run d'evaluation.") from e

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

    # ── Evaluation des resultats ──────────────────────────────────────────────
    log.info("\n%s\nEVALUATION DES ARGUMENTS GENERES\n%s", "=" * 60, "=" * 60)
    report_df = evaluator.evaluate_batch(results, reference_winners)

    log.info("\n%s", report_df[["axe1_score", "axe2_authenticity"]].describe().to_string())

    # Sauvegarde du rapport
    report_path = "generation_report.csv"
    report_df.to_csv(report_path, index=False)
    log.info("Rapport sauvegarde -> %s", report_path)

    # Sauvegarde version HTML lisible
    html_path = "generation_report.html"
    html_style = """
    <style>
      body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f9f9f9;}
      table { border-collapse: collapse; width: 100%; background-color: white; box-shadow: 0 1px 3px rgba(0,0,0,0.2); }
      th, td { border: 1px solid #ddd; padding: 15px; text-align: left; vertical-align: top; }
      th { background-color: #4CAF50; color: white; position: sticky; top: 0; }
      tr:nth-child(even) { background-color: #f2f2f2; }
      tr:hover { background-color: #e8f5e9; }
      .text-cell { white-space: pre-wrap; font-family: monospace; font-size: 13px; line-height: 1.4; max-width: 600px;}
      .score-cell { font-weight: bold; font-size: 16px;}
      .score-good { color: #2e7d32; }
      .score-bad { color: #c62828; }
    </style>
    """
    
    html_content = f"<!DOCTYPE html>\n<html>\n<head><meta charset='utf-8'>{html_style}</head>\n<body>\n<h2>Rapport de Generation - Axe 3</h2>\n"
    html_content += "<table>\n<tr><th>OP (Message original)</th><th>Reponse de l'IA (LLM)</th><th>Score Axe 1 (Convaincant > 0)</th><th>Score Axe 2 (Humain < 0)</th></tr>\n"
    
    for _, row in report_df.iterrows():
        a1_class = "score-good" if row['axe1_score'] > 0 else "score-bad"
        a2_class = "score-good" if row['axe2_authenticity'] < 0 else "score-bad"
        
        html_content += f"""
        <tr>
            <td class="text-cell">{str(row['op_text']).replace('<', '&lt;')}</td>
            <td class="text-cell">{str(row['generated_text']).replace('<', '&lt;')}</td>
            <td class="score-cell {a1_class}">{row['axe1_score']:.3f}</td>
            <td class="score-cell {a2_class}">{row['axe2_authenticity']:.3f}</td>
        </tr>
        """
    html_content += "</table>\n</body>\n</html>"
    
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    log.info("Rapport HTML lisible sauvegarde -> %s", html_path)

    # Affichage de quelques exemples
    log.info("\n--- Exemples generes ---")
    for _, row in report_df.head(3).iterrows():
        log.info("OP : %s", row["op_text"])
        log.info("GEN [axe1=%.3f | auth=%.3f] : %s\n",
                 row["axe1_score"], row["axe2_authenticity"], row["generated_text"])


if __name__ == "__main__":
    main()
