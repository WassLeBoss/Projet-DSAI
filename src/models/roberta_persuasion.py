import os
import torch
from transformers import RobertaForMultipleChoice, RobertaTokenizer
import json

class RobertaPersuasionClassifier:
    """
    Classe wrapper pour utiliser facilement le modèle RoBERTa fine-tuné
    afin de classifier un argument individuel par rapport à un OP.
    """
    def __init__(self, model_dir_or_file: str):
        # Résolution du chemin des poids et des métadonnées (seuil)
        if os.path.isdir(model_dir_or_file):
            self.model_path = os.path.join(model_dir_or_file, "best_roberta_cmv.pt")
            self.meta_path = os.path.join(model_dir_or_file, "roberta_persuasion_metadata.json")
        else:
            self.model_path = model_dir_or_file
            self.meta_path = os.path.join(os.path.dirname(model_dir_or_file), "roberta_persuasion_metadata.json")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.model = RobertaForMultipleChoice.from_pretrained("roberta-base")
        
        # Chargement des poids du modèle
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        else:
            raise FileNotFoundError(f"Fichier de modèle non trouvé à l'emplacement : {self.model_path}")
            
        self.model.to(self.device)
        self.model.eval()
        
        # Chargement du seuil optimal de décision
        self.threshold = 0.0  # Seuil par défaut
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                self.threshold = metadata.get("optimal_threshold", 0.0)
                
    def get_persuasion_score(self, op: str, arg: str) -> float:
        """
        Calcule le score brut (logit) de persuasivité de l'argument pour l'OP donné.
        """
        inputs = self.tokenizer(
            op, arg, 
            truncation=True, max_length=512, 
            padding='max_length', return_tensors='pt'
        )
        
        # Passage au format attendu par RobertaForMultipleChoice : (batch_size, num_choices, seq_length)
        # Ici batch_size = 1 et num_choices = 1
        input_ids = inputs['input_ids'].unsqueeze(0).to(self.device)
        attention_mask = inputs['attention_mask'].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            score = outputs.logits[0, 0].item()
            
        return score
        
    def predict_probability(self, op: str, arg: str) -> float:
        """
        Retourne la probabilité (via sigmoïde) que l'argument soit persuasif.
        """
        score = self.get_persuasion_score(op, arg)
        return torch.sigmoid(torch.tensor(score)).item()
        
    def predict(self, op: str, arg: str) -> bool:
        """
        Prédit si l'argument est persuasif ou non en utilisant le seuil optimal (True/False).
        """
        score = self.get_persuasion_score(op, arg)
        return score > self.threshold
