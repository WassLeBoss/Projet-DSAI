"""
Script de Fine-Tuning de RoBERTa sur le Winning Argument Corpus (Axe 1).
Permet d'entraîner RobertaForMultipleChoice pour évaluer la persuasion.

Utilisation :
    python scripts/finetune_roberta_axe1.py
"""

import os
import sys
import random
import warnings
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForMultipleChoice, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm.auto import tqdm


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def locate_dataset():
    possible_paths = [
        "WAC_final.csv",
        "WAC.csv",
        "datasets/WinningArgCorpus/WAC_final.csv",
        "datasets/WinningArgCorpus/WAC.csv",
        "Datasets/WinningArgCorpus/WAC_final.csv",
        "Datasets/WinningArgCorpus/WAC.csv",
        "../WAC_final.csv",
        "../WAC.csv",
        "../../WAC_final.csv",
        "../../WAC.csv",
        "../../Datasets/WinningArgCorpus/WAC.csv",
        "../../datasets/WinningArgCorpus/WAC_final.csv",
        "../../datasets/WinningArgCorpus/WAC.csv",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            print(f"[OK] Dataset trouvé à l'emplacement : {path}")
            return path
    raise FileNotFoundError("Impossible de trouver WAC.csv ou WAC_final.csv dans les répertoires courants ou parents.")


def get_save_path():
    # Si on exécute depuis le sous-dossier scripts, on sauvegarde à la racine du projet
    if os.path.basename(os.getcwd()) == "scripts":
        return "../best_roberta_cmv.pt"
    return "best_roberta_cmv.pt"


def format_cmv_data(raw_df):
    print("Reformatage des données en cours...")
    winners_df = raw_df[raw_df['success'] == 1].set_index('pair_id')
    losers_df = raw_df[raw_df['success'] == 0].set_index('pair_id')
    
    valid_pairs = winners_df.index.intersection(losers_df.index)
    formatted_data = []
    
    for p_id in valid_pairs:
        op = winners_df.loc[p_id, 'op_text']
        winning_arg = winners_df.loc[p_id, 'text']
        losing_arg = losers_df.loc[p_id, 'text']
        
        if isinstance(op, pd.Series): op = op.iloc[0]
        if isinstance(winning_arg, pd.Series): winning_arg = winning_arg.iloc[0]
        if isinstance(losing_arg, pd.Series): losing_arg = losing_arg.iloc[0]
        
        if random.random() > 0.5:
            arg_0, arg_1 = winning_arg, losing_arg
            label = 0
        else:
            arg_0, arg_1 = losing_arg, winning_arg
            label = 1
            
        formatted_data.append({
            'pair_id': p_id,
            'op': str(op),
            'arg_0': str(arg_0),
            'arg_1': str(arg_1),
            'label': label
        })
        
    print(f"Extraction réussie : {len(formatted_data)} paires prêtes.")
    return formatted_data


class CMVDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenisation de l'Option 0 : [CLS] OP [SEP] Arg_0 [SEP]
        enc_0 = self.tokenizer(
            item['op'], item['arg_0'], 
            truncation=True, max_length=self.max_length, 
            padding='max_length', return_tensors='pt'
        )
        
        # Tokenisation de l'Option 1 : [CLS] OP [SEP] Arg_1 [SEP]
        enc_1 = self.tokenizer(
            item['op'], item['arg_1'], 
            truncation=True, max_length=self.max_length, 
            padding='max_length', return_tensors='pt'
        )
        
        return {
            'input_ids': torch.cat([enc_0['input_ids'], enc_1['input_ids']], dim=0),
            'attention_mask': torch.cat([enc_0['attention_mask'], enc_1['attention_mask']], dim=0),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }


def main():
    warnings.filterwarnings('ignore')
    set_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Appareil utilisé pour l'entraînement : {device}")
    if torch.cuda.is_available():
        print(f"Modèle du GPU : {torch.cuda.get_device_name(0)}")
        
    # Localisation du dataset
    try:
        csv_path = locate_dataset()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return
        
    print(f"Lecture du dataset : {csv_path}")
    df = pd.read_csv(csv_path)
    processed_data = format_cmv_data(df)
    
    train_data, temp_data = train_test_split(processed_data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    print(f"Répartition : {len(train_data)} Train | {len(val_data)} Val | {len(test_data)} Test")
    
    print("Chargement du Tokenizer RoBERTa...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    train_dataset = CMVDataset(train_data, tokenizer)
    val_dataset = CMVDataset(val_data, tokenizer)
    test_dataset = CMVDataset(test_data, tokenizer)
    
    BATCH_SIZE = 8
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Hyperparamètres d'entraînement
    EPOCHS = 4
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    PATIENCE = 2
    
    print("Initialisation du modèle RobertaForMultipleChoice...")
    model = RobertaForMultipleChoice.from_pretrained('roberta-base')
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    total_steps = len(train_loader) * EPOCHS
    num_warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps
    )
    
    best_val_acc = 0.0
    patience_counter = 0
    save_path = get_save_path()
    
    print("Début de l'entraînement...")
    for epoch in range(EPOCHS):
        # --- ENTRAÎNEMENT ---
        model.train()
        total_train_loss = 0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for batch in train_loop:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loop.set_postfix(loss=loss.item())
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- VALIDATION ---
        model.eval()
        val_preds, val_labels = [], []
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
        
        with torch.no_grad():
            for batch in val_loop:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
        val_acc = accuracy_score(val_labels, val_preds)
        print(f"\n--- Bilan de l'Époque {epoch+1} ---")
        print(f"Perte d'entraînement : {avg_train_loss:.4f} | Précision de Validation : {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"=> Nouveau record ! Modèle sauvegardé sous '{save_path}'")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"=> Pas d'amélioration. Patience : {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print("\nArrêt précoce déclenché. Fin de l'entraînement.")
                break
                
    # --- EVALUATION TEST ---
    print("\nChargement des meilleurs poids sauvegardés...")
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    
    test_preds = []
    test_labels = []
    test_loop = tqdm(test_loader, desc="Évaluation Test")
    
    with torch.no_grad():
        for batch in test_loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            
    final_accuracy = accuracy_score(test_labels, test_preds)
    final_f1 = f1_score(test_labels, test_preds, average='macro')
    
    print("\n" + "="*40)
    print("RÉSULTATS FINAUX SUR LE JEU DE TEST")
    print("="*40)
    print(f"Précision (Accuracy) : {final_accuracy:.4f}")
    print(f"Score F1 (Macro)     : {final_f1:.4f}")
    print("\nRapport Détaillé :")
    print(classification_report(test_labels, test_preds, target_names=["Option 0 Gagne", "Option 1 Gagne"]))


if __name__ == '__main__':
    main()
