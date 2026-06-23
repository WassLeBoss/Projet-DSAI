"""Point d'entree — Generation d'arguments convaincants (Axe 3)."""

import logging
import pickle

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def _load_axe1_model(cfg: DictConfig):
    """Charge le SVM Axe 1 ou RoBERTa depuis le chemin specifie dans la config."""
    model_path = cfg.axe1.model_path
    log.info("Chargement modele Axe 1 : %s", model_path)

    if (
        model_path.endswith(".pt")
        or model_path.endswith(".bin")
        or "roberta" in cfg.axe1.encoder_name.lower()
    ):
        import torch
        from transformers import RobertaForMultipleChoice, RobertaTokenizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Chargement de RoBERTa fine-tuned pour l'Axe 1 sur %s...", device)

        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        model = RobertaForMultipleChoice.from_pretrained("roberta-base")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        return {"model": model, "tokenizer": tokenizer, "type": "roberta_finetuned"}

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


@hydra.main(
    version_base=None, config_path="configs/generation", config_name="config_generate"
)
def main(cfg: DictConfig) -> None:
    log.info("=" * 60)
    log.info("Generation Axe 3")
    log.info("LLM      : %s", cfg.llm.name)
    log.info("Strategie: %s", cfg.strategy.name)
    log.info("=" * 60)

    # Chargement des modeles Axes 1 & 2
    axe1_clf = _load_axe1_model(cfg)
    axe2_clf = _load_axe2_model(cfg)

    # Chargement des OPs de test depuis le WAC
    log.info("Chargement des OPs depuis : %s", cfg.wac_csv_path)
    df = pd.read_csv(cfg.wac_csv_path)

    # On prend des OPs uniques (une par conversation)
    op_texts = (
        df.groupby("pair_id")["op_text"]
        .first()
        .dropna()
        .sample(
            n=min(cfg.n_op_samples, len(df["pair_id"].unique())), random_state=cfg.seed
        )
        .tolist()
    )
    log.info("%d OPs selectionnes pour la generation", len(op_texts))

    # Vrais winners pour le delta de features dans l'evaluation
    reference_winners = df[df["success"] == 1]["text"].dropna().tolist()[:200]

    # Initialisation du generateur
    from src.generation.generator import ArgumentGenerator

    generator = ArgumentGenerator(cfg.llm.model_id, cfg.llm)

    # Evaluateur
    from src.generation.evaluator import ArgumentEvaluator

    evaluator = ArgumentEvaluator(
        axe1_clf=axe1_clf,
        axe2_clf=axe2_clf,
        axe1_enc_name=cfg.axe1.encoder_name,
        axe2_enc_name=cfg.axe2.encoder_name,
    )

    # Test d'evaluation rapide (fail-fast)
    log.info("Verification de la compatibilite des modeles (dry-run)...")
    try:
        evaluator.evaluate("Test OP", "Test Generated")
        log.info("Verification OK ! Les dimensions correspondent.")
    except Exception as e:
        log.error("ERREUR CRITIQUE : Les dimensions des modeles ne correspondent pas !")
        log.error("Details: %s", str(e))
        raise RuntimeError("Echec du dry-run d'evaluation.") from e

    # Strategie de generation
    results = []

    if cfg.strategy.name == "best_of_n":
        from src.generation.best_of_n import BestOfNSelector

        # Injecte le nom de l'encodeur Axe 1 dans la config strategie
        strategy_cfg = OmegaConf.to_container(cfg.strategy, resolve=True)
        strategy_cfg["axe1_encoder_name"] = cfg.axe1.encoder_name
        selector = BestOfNSelector(axe1_clf, None, OmegaConf.create(strategy_cfg))

        for i, op_text in enumerate(op_texts):
            log.info(
                "[%d/%d] Generation Best-of-%d...",
                i + 1,
                len(op_texts),
                cfg.strategy.n_candidates,
            )
            candidates = generator.generate_n(op_text, n=cfg.strategy.n_candidates)
            best, scores = selector.select(op_text, candidates)
            results.append((op_text, best))

    elif cfg.strategy.name == "prompt_eng":
        from src.features.pairwise import get_feature_names
        from src.generation.prompt_engineering import PromptEngineer

        feature_names = get_feature_names()
        pe = PromptEngineer(axe1_clf, feature_names, cfg.strategy)

        log.info("Strategies identifiees depuis Axe 1 :")
        for feat, instr in pe.get_strategy_summary().items():
            log.info("  [%s] -> %s", feat, instr)

        for i, op_text in enumerate(op_texts):
            log.info(
                "[%d/%d] Generation avec prompt engineering...", i + 1, len(op_texts)
            )
            engineered_prompt = pe.build_prompt(op_text)
            candidates = generator.generate_n(
                engineered_prompt, n=cfg.strategy.n_candidates
            )

            # Best-of-N parmi les candidats generes avec prompt engineering
            from src.generation.best_of_n import BestOfNSelector

            strategy_cfg = OmegaConf.create(
                {"axe1_encoder_name": cfg.axe1.encoder_name, "verbose": False}
            )
            selector = BestOfNSelector(axe1_clf, None, strategy_cfg)
            best, _ = selector.select(op_text, candidates)
            results.append((op_text, best))

    else:
        raise ValueError(f"Strategie inconnue : {cfg.strategy.name}")

    # Evaluation des resultats
    log.info("\n%s\nEVALUATION DES ARGUMENTS GENERES\n%s", "=" * 60, "=" * 60)
    report_df = evaluator.evaluate_batch(results, reference_winners)

    log.info(
        "\n%s", report_df[["axe1_score", "axe2_authenticity"]].describe().to_string()
    )

    # Sauvegarde du rapport
    report_path = "generation_report.csv"
    report_df.to_csv(report_path, index=False)
    log.info("Rapport sauvegarde -> %s", report_path)

    import math

    def sigmoid(x):
        # Clip pour éviter les overflows mathématiques si le score est trop extrême
        x = max(min(x, 100), -100)
        return 1 / (1 + math.exp(-x))

    # Sauvegarde version HTML lisible
    html_path = "generation_report.html"
    html_style = """<style>"""

    html_content = f"<!DOCTYPE html>\n<html>\n<head><meta charset='utf-8'>{html_style}</head>\n<body>\n<h2>Rapport de Generation - Axe 3</h2>\n"
    html_content += "<table>\n<tr><th>OP (Message original)</th><th>Reponse de l'IA (LLM)</th><th>Axe 1 (% Convaincant)</th><th>Axe 2 (% Généré par IA)</th></tr>\n"

    for _, row in report_df.iterrows():
        # Transformation Sigmoïde des logits SVM vers [0, 1] puis pourcentage
        p_convincing = sigmoid(row["axe1_score"]) * 100
        p_ai = sigmoid(row["axe2_authenticity"]) * 100

        # Axe 1 : > 50% = Convaincant (Vert)
        a1_class = "score-good" if p_convincing > 50 else "score-bad"

        # Axe 2 : < 50% = Humain (Vert, car on veut tromper l'Axe 2), > 50% = IA détectée (Rouge)
        a2_class = "score-good" if p_ai < 50 else "score-bad"

        html_content += f"""<tr>"""
    html_content += "</table>\n</body>\n</html>"

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    log.info("Rapport HTML lisible sauvegarde -> %s", html_path)

    # Affichage de quelques exemples
    log.info("\n--- Exemples generes ---")
    for _, row in report_df.head(3).iterrows():
        log.info("OP : %s", row["op_text"])
        log.info(
            "GEN [axe1=%.3f | auth=%.3f] : %s\n",
            row["axe1_score"],
            row["axe2_authenticity"],
            row["generated_text"],
        )

    # Visualisation UMAP (Generated vs M4GT)
    log.info("\n%s\nVISUALISATION UMAP AVEC M4GT\n%s", "=" * 60, "=" * 60)
    try:
        import matplotlib.pyplot as plt
        import umap
        from scipy import sparse

        m4gt_cfg = OmegaConf.create(
            {
                "dataset": {
                    "name": "m4gt",
                    "path": "datasets/M4GT-Bench/SubtaskA.jsonl",
                    "text_col": "text",
                    "label_col": "label",
                    "max_samples": 2000,
                }
            }
        )
        from src.data.loader import load_m4gt

        m4gt_texts, m4gt_labels = load_m4gt(m4gt_cfg)

        log.info("Encodage de %d textes M4GT avec l'encodeur Axe 2...", len(m4gt_texts))
        if evaluator.axe2_encoder is not None:
            X_m4gt = evaluator.axe2_encoder.transform(m4gt_texts)
        else:
            X_m4gt = np.vstack([evaluator._encode_axe2(t) for t in m4gt_texts])

        gen_texts = report_df["generated_text"].tolist()
        log.info("Encodage de %d textes generes...", len(gen_texts))
        if evaluator.axe2_encoder is not None:
            X_gen = evaluator.axe2_encoder.transform(gen_texts)
        else:
            X_gen = np.vstack([evaluator._encode_axe2(t) for t in gen_texts])

        if sparse.issparse(X_m4gt):
            X_all = sparse.vstack([X_m4gt, X_gen])
            from sklearn.decomposition import TruncatedSVD

            X_reduced = TruncatedSVD(
                n_components=min(50, X_all.shape[1] - 1)
            ).fit_transform(X_all)
        else:
            X_all = np.vstack([X_m4gt, X_gen])
            from sklearn.decomposition import PCA

            X_reduced = PCA(n_components=min(50, X_all.shape[1] - 1)).fit_transform(
                X_all
            )

        log.info("Calcul UMAP en cours (peut prendre quelques secondes)...")
        reducer = umap.UMAP(
            n_components=2, n_neighbors=15, min_dist=0.1, random_state=cfg.seed
        )
        coords = reducer.fit_transform(X_reduced)

        coords_m4gt = coords[: len(m4gt_texts)]
        coords_gen = coords[len(m4gt_texts) :]
        y_m4gt = np.array(m4gt_labels)

        plt.figure(figsize=(10, 7))
        # M4GT Humain (label 0)
        mask_hum = y_m4gt == 0
        plt.scatter(
            coords_m4gt[mask_hum, 0],
            coords_m4gt[mask_hum, 1],
            c="tomato",
            s=15,
            alpha=0.4,
            label="M4GT Humain",
        )
        # M4GT Machine (label 1)
        mask_mac = y_m4gt == 1
        plt.scatter(
            coords_m4gt[mask_mac, 0],
            coords_m4gt[mask_mac, 1],
            c="steelblue",
            s=15,
            alpha=0.4,
            label="M4GT Machine",
        )
        # Nos generations
        plt.scatter(
            coords_gen[:, 0],
            coords_gen[:, 1],
            c="lime",
            s=120,
            marker="*",
            edgecolor="black",
            linewidth=1.5,
            label="Nos Générations (Axe 3)",
        )

        plt.title(
            f"UMAP — Générations vs M4GT ({cfg.axe2.encoder_name.upper()})",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.legend()
        plt.tight_layout()

        umap_path = "umap_generation_m4gt.png"
        plt.savefig(umap_path, dpi=150)
        log.info("Visualisation UMAP sauvegardee -> %s", umap_path)
    except Exception as e:
        log.error("Impossible de generer l'UMAP avec M4GT : %s", e)


if __name__ == "__main__":
    main()
