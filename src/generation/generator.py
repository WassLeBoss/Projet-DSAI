"""
Generateur d'arguments — interface unifiee pour tout LLM HuggingFace CausalLM.

Usage:
    from src.generation.generator import ArgumentGenerator
    gen = ArgumentGenerator.from_pretrained("gpt2", cfg.llm)
    argument = gen.generate(op_text)
    candidates = gen.generate_n(op_text, n=10)
"""

import logging

import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

log = logging.getLogger(__name__)

OP_TOKEN    = "[OP]:"
REPLY_TOKEN = "[REPLY]:"


class ArgumentGenerator:
    """
    Genere des arguments a partir d'un post original (OP).

    Construit le prompt : "[OP]: {op_text}\\n[REPLY]: "
    et laisse le LLM completer la reponse.

    Args:
        model_path : chemin vers le modele (fine-tune) ou ID HuggingFace
        cfg        : config llm (temperature, top_p, max_new_tokens...)
    """

    def __init__(self, model_path: str, cfg: DictConfig) -> None:
        self.cfg = cfg
        log.info("Chargement du generateur : %s", model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.eval()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        log.info("Generateur pret sur : %s", self.device)

    def _build_prompt(self, op_text: str) -> str:
        """Construit le prompt de generation."""
        return f"{OP_TOKEN} {op_text}\n{REPLY_TOKEN} "

    @torch.no_grad()
    def generate(self, op_text: str) -> str:
        """
        Genere un argument pour un OP donne.

        Retourne :
            Le texte genere (sans le prompt)
        """
        prompt = self._build_prompt(op_text)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            do_sample=self.cfg.do_sample,
            repetition_penalty=self.cfg.repetition_penalty,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Decoде uniquement les nouveaux tokens (pas le prompt)
        n_prompt_tokens = inputs["input_ids"].shape[1]
        generated_ids   = output_ids[0, n_prompt_tokens:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text.strip()

    def generate_n(self, op_text: str, n: int) -> list[str]:
        """
        Genere N arguments independants pour un meme OP.

        Retourne :
            Liste de N strings
        """
        return [self.generate(op_text) for _ in range(n)]
