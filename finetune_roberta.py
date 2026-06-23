"""Point d'entree pour le fine-tuning direct de RoBERTa sur le dataset M4GT (Axe 2)."""

import logging
import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from datasets import Dataset

log = logging.getLogger(__name__)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
    }


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("=" * 60)
    log.info("Fine-Tuning RoBERTa pour Détection IA")
    log.info("Dataset : %s", cfg.dataset.name)
    log.info("=" * 60)

    # 1. Chargement du dataset (ex: M4GT)
    if cfg.dataset.name == "m4gt":
        from src.data.loader import load_m4gt

        texts, labels = load_m4gt(cfg)
    elif cfg.dataset.name == "hc3":
        from src.data.loader import load_hc3

        texts, labels = load_hc3(cfg)
    else:
        raise ValueError("Dataset non supporté pour ce script.")

    log.info("Chargement de %d documents.", len(texts))

    texts_train, texts_val, labels_train, labels_val = train_test_split(
        texts, labels, test_size=0.15, random_state=cfg.seed, stratify=labels
    )

    # 2. Tokenizer et préparation des données
    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_data(texts, labels):
        encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
        dataset = Dataset.from_dict(
            {
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"],
                "labels": labels,
            }
        )
        return dataset

    train_dataset = tokenize_data(texts_train, labels_train)
    val_dataset = tokenize_data(texts_val, labels_val)

    # 3. Initialisation du Modèle (Sequence Classification)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    # 4. Paramètres d'entraînement
    output_dir = f"outputs/roberta_finetuned_{cfg.dataset.name}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        fp16=torch.cuda.is_available(),  # Mixed precision si GPU dispo
    )

    # 5. Entraînement
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    log.info("Démarrage de l'entraînement sur %s...", device)
    trainer.train()

    # 6. Sauvegarde
    final_path = f"{output_dir}/best_model"
    trainer.save_model(final_path)
    log.info("Modèle sauvegardé dans : %s", final_path)


if __name__ == "__main__":
    main()
