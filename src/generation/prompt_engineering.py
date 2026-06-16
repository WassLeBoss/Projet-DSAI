"""
Prompt Engineering base sur les features Axe 1.

Principe :
    1. Analyser les features les plus discriminantes du SVM Axe 1 (coef_)
    2. Traduire chaque feature en instruction naturelle pour le LLM
    3. Construire un system prompt qui guide le LLM vers un argument convaincant

Usage:
    from src.generation.prompt_engineering import PromptEngineer
    pe = PromptEngineer(axe1_clf, cfg)
    prompt = pe.build_prompt(op_text)
    candidates = generator.generate_n(prompt, n=5)
"""

import logging

import numpy as np
from omegaconf import DictConfig

log = logging.getLogger(__name__)


# Traduction features → instructions naturelles (en anglais pour les LLMs)
FEATURE_TO_INSTRUCTION: dict[str, str] = {
    # Longueur
    "n_words":         "Write a detailed response of at least 150 words.",
    "n_sentences":     "Structure your argument in multiple clear sentences.",
    "n_paragraphs":    "Organize your response into distinct paragraphs.",
    # Liens et references
    "n_links":         "Support your claims with relevant links or references.",
    "n_edu_links":     "Cite academic or educational sources (.edu) when possible.",
    "n_examples":      "Use concrete examples (e.g., 'for example', 'for instance').",
    "n_quotes":        "Quote relevant passages to support your argument.",
    # Pronoms et engagement
    "n_2p":            "Directly address the original poster using 'you'.",
    "n_1sg":           "Use first-person singular ('I think', 'I believe') to personalize.",
    "n_1pl":           "Use inclusive language ('we', 'our') to create common ground.",
    # Hedges et nuance
    "n_hedges":        "Acknowledge uncertainty where appropriate ('it seems', 'perhaps').",
    # Lisibilite
    "flesch_kincaid":  "Keep sentences concise and easy to read.",
    # Sentiment
    "n_positive":      "Frame your argument positively and constructively.",
    # Markdown
    "n_bullet_list":   "Use bullet points to list your key arguments clearly.",
    "n_bolds":         "Bold the most important points in your response.",
    # Richesse lexicale
    "type_token":      "Use varied and rich vocabulary.",
    "word_entropy":    "Maintain lexical diversity throughout your response.",
}


class PromptEngineer:
    """
    Construit des system prompts a partir des features les plus
    discriminantes du SVM Axe 1.

    Args:
        axe1_clf      : SVM Axe 1 (sklearn SVC avec kernel lineaire)
        feature_names : liste ordonnee des noms de features
        cfg           : config strategy.prompt_eng
    """

    def __init__(self, axe1_clf, feature_names: list[str], cfg: DictConfig) -> None:
        self.clf           = axe1_clf
        self.feature_names = feature_names
        self.cfg           = cfg
        self._top_features = self._extract_top_features()

    def _extract_top_features(self) -> list[str]:
        """
        Extrait les features les plus positivement correlees au succes.

        Pour SVM lineaire : utilise clf.coef_ (coefficients du separateur).
        Retourne les top_k features avec coefficient positif le plus eleve.
        """
        if not hasattr(self.clf, "coef_"):
            log.warning("Le SVM n'a pas de coef_ (kernel non lineaire). "
                        "Utilisation de toutes les features disponibles.")
            return list(FEATURE_TO_INSTRUCTION.keys())[:self.cfg.top_k_features]

        coefs = self.clf.coef_[0]  # shape (n_features,)
        top_indices = np.argsort(coefs)[::-1][:self.cfg.top_k_features]
        top_features = [self.feature_names[i] for i in top_indices
                        if self.feature_names[i] in FEATURE_TO_INSTRUCTION]

        log.info("Top features pour le prompt : %s", top_features)
        return top_features

    def build_system_prompt(self) -> str:
        """
        Construit le system prompt avec les instructions derivees de l'Axe 1.

        Retourne :
            System prompt en anglais
        """
        instructions = [
            FEATURE_TO_INSTRUCTION[f]
            for f in self._top_features
            if f in FEATURE_TO_INSTRUCTION
        ]

        prompt = (
            "You are an expert debater. Your goal is to write a highly convincing "
            "reply to a Reddit post. Follow these guidelines:\n\n"
        )
        for i, instr in enumerate(instructions, 1):
            prompt += f"{i}. {instr}\n"

        prompt += "\nNow write a convincing reply to the following post:\n"
        return prompt

    def build_prompt(self, op_text: str) -> str:
        """
        Construit le prompt complet (system + OP).

        Args:
            op_text : texte du post original

        Retourne :
            Prompt complet pret pour le LLM
        """
        system = self.build_system_prompt()
        return f"{system}\n[POST]: {op_text}\n\n[YOUR REPLY]: "

    def get_strategy_summary(self) -> dict[str, str]:
        """Retourne un dictionnaire feature → instruction pour logging/rapport."""
        return {
            f: FEATURE_TO_INSTRUCTION[f]
            for f in self._top_features
            if f in FEATURE_TO_INSTRUCTION
        }
