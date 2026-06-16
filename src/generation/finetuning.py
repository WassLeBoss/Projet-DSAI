"""
Fine-tuning d'un LLM sur les arguments gagnants du WAC.

Le modele apprend a continuer un debat de maniere convaincante.
Format d'entree :
    "[OP]: {op_text}\\n[REPLY]: {winning_argument}"

La loss est calculee uniquement sur la partie [REPLY] (next-token prediction).

Usage:
    from src.generation.finetuning import WACFinetuner
    finetuner = WACFinetuner(cfg)
    finetuner.train(train_pairs)
    finetuner.save("outputs/finetuned_gpt2/")
"""

import logging
from pathlib import Path

import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

log = logging.getLogger(__name__)

# Tokens speciaux pour structurer le prompt
OP_TOKEN    = "[OP]:"
REPLY_TOKEN = "[REPLY]:"


class WACDebateDataset(Dataset):
    """
    Dataset PyTorch pour le fine-tuning sur le WAC.

    Chaque exemple = un argument gagnant (success=1) avec son OP.
    Format : "[OP]: {op_text}\\n[REPLY]: {winning_text}<eos>"
    La loss est masquee sur la partie [OP] — on predit uniquement la reponse.
    """

    def __init__(
        self,
        pairs_df: pd.DataFrame,
        tokenizer,
        max_length: int = 512,
    ) -> None:
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.examples   = self._build_examples(pairs_df)

    def _build_examples(self, df: pd.DataFrame) -> list[dict]:
        examples = []
        for _, row in df.iterrows():
            prompt = f"{OP_TOKEN} {row['op_text']}\n{REPLY_TOKEN} "
            full   = prompt + row["winner_text"] + self.tokenizer.eos_token

            enc = self.tokenizer(
                full,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids      = enc["input_ids"].squeeze()
            attention_mask = enc["attention_mask"].squeeze()

            # Masque la loss sur la partie prompt (labels = -100)
            prompt_enc = self.tokenizer(
                prompt, truncation=True, max_length=self.max_length
            )
            prompt_len = len(prompt_enc["input_ids"])

            labels = input_ids.clone()
            labels[:prompt_len] = -100  # pas de loss sur le prompt OP

            examples.append({
                "input_ids":      input_ids,
                "attention_mask": attention_mask,
                "labels":         labels,
            })
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


class WACFinetuner:
    """
    Fine-tune un LLM causal (GPT-2 ou Mistral+LoRA) sur les arguments
    gagnants du WinningArgCorpus.

    Args:
        cfg : config Hydra (cfg.llm)
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        log.info("Chargement du modele : %s", cfg.model_id)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_id)

        if cfg.use_lora:
            self._apply_lora()

    def _apply_lora(self) -> None:
        """Applique LoRA au modele (necessite peft)."""
        from peft import LoraConfig, get_peft_model, TaskType
        lora_config = LoraConfig(
            r=self.cfg.lora_r,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=list(self.cfg.lora_target_modules),
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def train(self, train_pairs: pd.DataFrame, output_dir: str = "outputs/finetuned") -> None:
        """
        Lance le fine-tuning.

        Args:
            train_pairs : DataFrame avec colonnes winner_text, op_text
            output_dir  : dossier de sauvegarde du modele
        """
        log.info("Construction du dataset de fine-tuning (%d paires)...", len(train_pairs))
        dataset = WACDebateDataset(train_pairs, self.tokenizer, self.cfg.max_input_length)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.cfg.num_train_epochs,
            per_device_train_batch_size=self.cfg.batch_size,
            learning_rate=self.cfg.learning_rate,
            save_strategy="epoch",
            logging_steps=50,
            report_to="none",
            fp16=torch.cuda.is_available(),
            gradient_accumulation_steps=getattr(self.cfg, "gradient_accumulation_steps", 1),
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )

        log.info("Debut du fine-tuning...")
        trainer.train()
        log.info("Fine-tuning termine.")

    def save(self, output_dir: str) -> None:
        """Sauvegarde le modele et le tokenizer."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        log.info("Modele sauvegarde -> %s", output_dir)
