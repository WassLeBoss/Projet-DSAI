import os
import sys
import random
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import RobertaForMultipleChoice, RobertaTokenizer
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm

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

class SingleCMVDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        enc = self.tokenizer(
            item['op'], item['arg'], 
            truncation=True, max_length=self.max_length, 
            padding='max_length', return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'], 
            'attention_mask': enc['attention_mask'], 
            'label': torch.tensor(item['label'], dtype=torch.long)
        }

def find_best_threshold(model_path, csv_path):
    if not os.path.exists(model_path):
        print(f"[ERROR] Le fichier de modèle '{model_path}' n'a pas été trouvé.")
        print("Veuillez télécharger le fichier 'best_roberta_cmv.pt' et le placer à la racine du projet.")
        return None

    if not os.path.exists(csv_path):
        print(f"[ERROR] Le fichier de données '{csv_path}' n'existe pas.")
        return None

    print(f"[...] Chargement du dataset depuis {csv_path}...")
    df = pd.read_csv(csv_path)
    processed_data = format_cmv_data(df)

    # Même configuration aléatoire que dans le notebook pour aligner les splits
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    train_data, temp_data = train_test_split(processed_data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Conversion des paires en arguments uniques pour la validation binaire
    single_val_data = []
    for item in val_data:
        op = item['op']
        if item['label'] == 0:
            single_val_data.append({'op': op, 'arg': item['arg_0'], 'label': 1})  # Gagnant (success=1)
            single_val_data.append({'op': op, 'arg': item['arg_1'], 'label': 0})  # Perdant (success=0)
        else:
            single_val_data.append({'op': op, 'arg': item['arg_0'], 'label': 0})  # Perdant (success=0)
            single_val_data.append({'op': op, 'arg': item['arg_1'], 'label': 1})  # Gagnant (success=1)

    print(f"[OK] {len(single_val_data)} exemples de validation uniques construits.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[...] Chargement de RoBERTa sur {device} (ceci peut prendre quelques instants)...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForMultipleChoice.from_pretrained('roberta-base')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    val_dataset = SingleCMVDataset(single_val_data, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    print("[...] Calcul des scores (logits) sur le jeu de validation...")
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']

            # Les tenseurs sont déjà de taille (batch_size, 1, 512)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            scores = outputs.logits[:, 0].cpu().numpy()
            
            all_scores.extend(scores)
            all_labels.extend(labels.numpy())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Recherche du meilleur seuil
    best_threshold = 0.0
    best_f1 = 0.0
    best_acc = 0.0

    print("[...] Recherche du seuil optimal de classification...")
    thresholds = np.linspace(all_scores.min() - 0.5, all_scores.max() + 0.5, 300)
    for t in thresholds:
        preds = (all_scores > t).astype(int)
        f1 = f1_score(all_labels, preds, average='macro')
        acc = accuracy_score(all_labels, preds)
        
        if f1 > best_f1:
            best_f1 = f1
            best_acc = acc
            best_threshold = t

    print("\n" + "="*60)
    print("RÉSULTATS DE LA RECHERCHE DE SEUIL OPTIMAL")
    print("="*60)
    print(f"Seuil optimal (Logit brut) : {best_threshold:.4f}")
    print(f"Seuil en probabilité (Sigmoïde) : {torch.sigmoid(torch.tensor(best_threshold)).item():.4f}")
    print(f"Meilleur Score F1 (Macro)    : {best_f1:.4f}")
    print(f"Précision (Accuracy)         : {best_acc:.4f}")
    print("="*60)

    # Sauvegarde du seuil et des métadonnées
    meta_path = os.path.join(os.path.dirname(model_path), "roberta_persuasion_metadata.json")
    metadata = {
        "optimal_threshold": float(best_threshold),
        "optimal_probability_threshold": float(torch.sigmoid(torch.tensor(best_threshold)).item()),
        "validation_macro_f1": float(best_f1),
        "validation_accuracy": float(best_acc)
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"[OK] Métadonnées et seuil sauvegardés dans : {meta_path}")
    return best_threshold

if __name__ == "__main__":
    # Recherche du modèle à la racine ou dans le dossier courant
    model_file = "best_roberta_cmv.pt"
    if not os.path.exists(model_file):
        model_file = "src/models/best_roberta_cmv.pt"

    # Recherche du jeu de données
    csv_file = "datasets/WinningArgCorpus/WAC_final.csv"
    if not os.path.exists(csv_file):
        csv_file = "datasets/WinningArgCorpus/WAC.csv"

    find_best_threshold(model_file, csv_file)
