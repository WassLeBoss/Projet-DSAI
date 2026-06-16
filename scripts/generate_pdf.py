# -*- coding: utf-8 -*-
"""
Generateur du PDF "Guide de demarrage" pour Projet-DSAI.
"""

from fpdf import FPDF
from fpdf.enums import XPos, YPos
import os

# ── Palette de couleurs ──────────────────────────────────────────────────────
BLACK      = (15,  15,  15)
WHITE      = (255, 255, 255)
ACCENT     = (67,  97,  238)    # bleu principal
ACCENT2    = (114, 9,   183)    # violet
SUCCESS    = (34,  197, 94)     # vert
WARNING    = (234, 179, 8)      # jaune
DARK_BG    = (30,  30,  46)     # fond fonce header
LIGHT_BG   = (245, 247, 255)    # fond clair section
CODE_BG    = (30,  30,  46)     # fond code
CODE_FG    = (166, 226, 46)     # texte code (vert terminal)
GRAY       = (100, 100, 120)
DIVIDER    = (220, 220, 235)

AXE_COLORS = {
    1: (67,  97,  238),    # Axe 1 = bleu
    2: (114, 9,   183),    # Axe 2 = violet
    3: (34,  197, 94),     # Axe 3 = vert
}


class GuidePDF(FPDF):

    def header(self):
        pass  # header custom par page

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GRAY)
        self.cell(0, 10, f"Projet-DSAI — Guide de demarrage   |   Page {self.page_no()}", align="C")

    def cover_page(self):
        # Fond sombre
        self.set_fill_color(*DARK_BG)
        self.rect(0, 0, 210, 297, "F")

        # Bande coloree en haut
        self.set_fill_color(*ACCENT)
        self.rect(0, 0, 210, 6, "F")

        # Titre principal
        self.set_y(60)
        self.set_font("Helvetica", "B", 32)
        self.set_text_color(*WHITE)
        self.cell(0, 14, "PROJET-DSAI", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.set_font("Helvetica", "", 16)
        self.set_text_color(180, 180, 200)
        self.cell(0, 10, "Guide de demarrage complet", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Sous-titre
        self.ln(6)
        self.set_font("Helvetica", "", 12)
        self.set_text_color(140, 140, 170)
        self.cell(0, 8, "De zero a la generation d'arguments convaincants", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Axes badges
        self.ln(20)
        badge_labels = ["AXE 1 — Analyse", "AXE 2 — Detection", "AXE 3 — Generation"]
        badge_colors = [ACCENT, ACCENT2, SUCCESS]
        x_start = 25
        for label, color in zip(badge_labels, badge_colors):
            self.set_fill_color(*color)
            self.set_text_color(*WHITE)
            self.set_font("Helvetica", "B", 10)
            self.set_xy(x_start, self.get_y())
            self.cell(52, 10, label, align="C", fill=True, new_x=XPos.RIGHT)
            x_start += 57

        # Ligne d'info bas de page
        self.set_y(250)
        self.set_fill_color(40, 40, 60)
        self.rect(0, 248, 210, 30, "F")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(160, 160, 180)
        self.cell(0, 8, "Python 3.10+  |  PyTorch  |  scikit-learn  |  Hydra  |  HuggingFace Transformers", align="C")

    def section_header(self, title: str, color: tuple = ACCENT):
        self.ln(4)
        self.set_fill_color(*color)
        self.rect(self.get_x(), self.get_y(), 190, 0.6, "F")
        self.ln(3)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*color)
        self.cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*BLACK)
        self.ln(1)

    def axe_banner(self, n: int, title: str, subtitle: str):
        color = AXE_COLORS[n]
        self.set_fill_color(*color)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 12, f"  AXE {n} — {title}", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_fill_color(color[0]+30, color[1]+30, min(color[2]+30, 255))
        self.set_font("Helvetica", "I", 9)
        self.cell(0, 7, f"  {subtitle}", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*BLACK)
        self.ln(3)

    def body_text(self, text: str, size: int = 10):
        self.set_font("Helvetica", "", size)
        self.set_text_color(*BLACK)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def bullet(self, text: str, indent: int = 8):
        self.set_x(self.l_margin + indent)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*BLACK)
        self.cell(5, 5.5, "\u2022")
        self.set_x(self.l_margin + indent + 5)
        self.multi_cell(0, 5.5, text)

    def note_box(self, text: str, color: tuple = WARNING):
        self.set_fill_color(color[0], color[1], color[2], )
        x = self.get_x()
        y = self.get_y()
        self.set_fill_color(255, 251, 230)
        self.set_draw_color(*color)
        self.set_line_width(0.8)
        # Barre gauche
        self.set_fill_color(*color)
        self.rect(self.l_margin, y, 2, 14, "F")
        self.set_fill_color(255, 251, 230)
        self.rect(self.l_margin + 2, y, 186, 14, "F")
        self.set_xy(self.l_margin + 5, y + 2)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*color)
        self.cell(12, 5, "NOTE")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(80, 80, 80)
        self.set_x(self.l_margin + 17)
        self.multi_cell(171, 5, text)
        self.set_draw_color(0, 0, 0)
        self.set_line_width(0.2)
        self.ln(3)

    def code_block(self, lines: list[str], title: str = ""):
        if title:
            self.set_font("Helvetica", "B", 8)
            self.set_text_color(*GRAY)
            self.cell(0, 5, f"  # {title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        h_per_line = 5.5
        total_h = h_per_line * len(lines) + 6
        x = self.l_margin
        y = self.get_y()

        self.set_fill_color(*CODE_BG)
        self.rect(x, y, 190, total_h, "F")

        self.set_y(y + 3)
        for line in lines:
            self.set_x(x + 4)
            self.set_font("Courier", "", 8.5)
            # Coloration syntaxique simple
            if line.strip().startswith("#"):
                self.set_text_color(117, 113, 94)
            elif line.strip().startswith("python ") or line.strip().startswith("pip ") or line.strip().startswith("git "):
                self.set_text_color(*CODE_FG)
            elif line.strip().startswith("cd "):
                self.set_text_color(102, 217, 239)
            else:
                self.set_text_color(248, 248, 242)
            self.cell(0, h_per_line, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.set_text_color(*BLACK)
        self.ln(3)

    def step_number(self, n: int, title: str):
        self.set_fill_color(*ACCENT)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 9)
        y = self.get_y()
        self.rect(self.l_margin, y, 7, 7, "F")
        self.set_xy(self.l_margin, y)
        self.cell(7, 7, str(n), align="C")
        self.set_text_color(*BLACK)
        self.set_font("Helvetica", "B", 11)
        self.set_x(self.l_margin + 10)
        self.cell(0, 7, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)


def build_pdf(output_path: str):
    pdf = GuidePDF()
    pdf.set_margins(12, 12, 12)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # ═══════════════════════════════════════════════════════════
    # PAGE 1 : COUVERTURE
    # ═══════════════════════════════════════════════════════════
    pdf.cover_page()

    # ═══════════════════════════════════════════════════════════
    # PAGE 2 : PREREQUIS & INSTALLATION
    # ═══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_header("0. PREREQUIS & INSTALLATION INITIALE", DARK_BG)

    pdf.step_number(1, "Prerequisites systeme")
    pdf.bullet("Python 3.10 ou superieur")
    pdf.bullet("Git installe et configure")
    pdf.bullet("Au moins 8 Go de RAM (16 Go recommandes)")
    pdf.bullet("GPU optionnel (requis uniquement pour Mistral 7B / RoBERTa en production)")
    pdf.ln(3)

    pdf.step_number(2, "Cloner le depot Git")
    pdf.code_block([
        "git clone https://github.com/votre-org/Projet-DSAI.git",
        "cd Projet-DSAI",
    ], "Clonage et navigation")

    pdf.step_number(3, "Creer et activer le virtual environment")
    pdf.body_text("Windows :")
    pdf.code_block([
        "python -m venv env",
        "env\\Scripts\\activate",
    ], "Windows")
    pdf.body_text("macOS / Linux :")
    pdf.code_block([
        "python -m venv env",
        "source env/bin/activate",
    ], "macOS / Linux")

    pdf.note_box("Le prompt de votre terminal doit afficher (env) pour confirmer l'activation.", WARNING)

    pdf.step_number(4, "Installer les dependances")
    pdf.code_block([
        "pip install --upgrade pip",
        "pip install -r requirements.txt",
        "",
        "# Dependances principales installes :",
        "# numpy, pandas, scikit-learn, matplotlib",
        "# nltk, gensim, sentence-transformers, transformers, torch",
        "# hydra-core, omegaconf",
        "# convokit, datasets (HuggingFace)",
        "# fpdf2 (pour regen le PDF)",
    ], "Installation")

    pdf.step_number(5, "Telecharger les datasets")
    pdf.code_block([
        "# WinningArgCorpus (Axes 1 & 2)",
        "python -X utf8 scripts/download_datasets.py --dataset wac",
        "",
        "# HC3 — Human ChatGPT Comparison Corpus (Axe 2)",
        "python -X utf8 scripts/download_datasets.py --dataset hc3",
        "",
        "# Ou tout d'un coup :",
        "python -X utf8 scripts/download_datasets.py --all",
    ], "Telechargement")

    pdf.note_box(
        "Les datasets sont sauvegardes dans datasets/WinningArgCorpus/WAC.csv "
        "et datasets/HC3/train.csv. Ces dossiers sont dans .gitignore.",
        SUCCESS
    )

    # ═══════════════════════════════════════════════════════════
    # PAGE 3 : AXE 1
    # ═══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.axe_banner(1, "ANALYSE D'ARGUMENTS", "Predire quels arguments sont convaincants (WAC dataset)")

    pdf.body_text(
        "L'Axe 1 entraine un SVM sur le WinningArgCorpus (WAC) pour predire si un argument "
        "va convaincre l'auteur du post original. Quatre encodeurs sont disponibles. "
        "Chaque run sauvegarde automatiquement le modele entraine (axe1_svm_*.pkl)."
    )

    pdf.section_header("Approche A — Features stylistiques (pairwise)", ACCENT)
    pdf.body_text(
        "Methode la plus rapide sur CPU. Construit des features numeriques (nb de mots, "
        "hedges, liens, lisibilite...) et compare paire gagnant/perdant."
    )
    pdf.code_block([
        "python train.py encoder=features",
        "",
        "# Sortie attendue :",
        "#   Train : 7 xxx exemples | Test : 1 xxx exemples",
        "#   AUC final : 0.5x - 0.6x",
        "#   Modele sauvegarde -> axe1_svm_features_wac.pkl",
    ])

    pdf.section_header("Approche B — TF-IDF + SVM", ACCENT)
    pdf.body_text("Representation sac-de-mots pondere. Bon point de comparaison baseline.")
    pdf.code_block([
        "python train.py encoder=tfidf",
        "",
        "# AUC attendu : ~0.58 - 0.64",
    ])

    pdf.section_header("Approche C — Word2Vec + SVM", ACCENT)
    pdf.body_text("Embeddings de mots entraines sur le corpus, moyenne par texte.")
    pdf.code_block([
        "python train.py encoder=w2v",
    ])

    pdf.section_header("Approche D — RoBERTa + SVM", ACCENT)
    pdf.body_text("Embeddings contextuels pre-entraines (tres lent sur CPU, recommande GPU).")
    pdf.code_block([
        "python train.py encoder=roberta",
        "",
        "# ATTENTION : tres lent sur CPU (~30-60 min)",
    ])

    pdf.note_box(
        "CONSEIL CPU : Commencer par encoder=features (< 5 min), puis tfidf (~2-5 min), "
        "puis w2v (~5-10 min). Garder roberta pour un environnement GPU.",
        WARNING
    )

    pdf.section_header("Sweep complet (tous les encodeurs)", ACCENT)
    pdf.code_block([
        "# Lance les 4 encodeurs en sequence",
        "python train.py -m encoder=tfidf,w2v,features",
        "",
        "# Visualisation PCA des representations",
        "python evaluate.py encoder=features",
        "python evaluate.py encoder=tfidf",
    ])

    pdf.section_header("Outputs Axe 1", ACCENT)
    pdf.body_text("Chaque run cree un dossier horodate dans outputs/ :")
    pdf.code_block([
        "outputs/",
        "  2026-06-16/",
        "    15-30-00_features_wac/",
        "      train.log          <- rapport complet",
        "      .hydra/config.yaml <- config exacte du run",
        "      axe1_svm_features_wac.pkl  <- modele SVM",
    ])

    # ═══════════════════════════════════════════════════════════
    # PAGE 4 : AXE 2
    # ═══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.axe_banner(2, "DETECTION DE TEXTES SYNTHETIQUES", "Distinguer textes humains vs generes par ChatGPT")

    pdf.body_text(
        "L'Axe 2 entraine un SVM sur le dataset HC3 (Human ChatGPT Comparison Corpus) "
        "pour detecter si un texte a ete ecrit par un humain (label 0) ou par ChatGPT (label 1). "
        "Le pipeline est identique a l'Axe 1, seul le dataset change."
    )

    pdf.note_box(
        "S'assurer d'avoir telecharge HC3 : "
        "python -X utf8 scripts/download_datasets.py --dataset hc3",
        ACCENT
    )

    pdf.section_header("Commandes (dans l'ordre recommande)", ACCENT2)
    pdf.code_block([
        "# Approche A : RoBERTa + SVM (meilleure performance attendue)",
        "python train.py dataset=hc3 encoder=roberta",
        "",
        "# Approche B : TF-IDF + SVM (baseline rapide)",
        "python train.py dataset=hc3 encoder=tfidf",
        "",
        "# Approche C : Word2Vec + SVM",
        "python train.py dataset=hc3 encoder=w2v",
        "",
        "# Sweep complet",
        "python train.py -m dataset=hc3 encoder=tfidf,w2v,roberta",
    ])

    pdf.section_header("Visualisation PCA", ACCENT2)
    pdf.code_block([
        "# Visualiser la separation humain vs ChatGPT",
        "python evaluate.py dataset=hc3 encoder=roberta",
        "",
        "# La figure PCA est sauvegardee dans outputs/",
    ])

    pdf.section_header("Outputs Axe 2", ACCENT2)
    pdf.code_block([
        "outputs/",
        "  2026-06-16/",
        "    15-45-00_roberta_hc3/",
        "      train.log",
        "      axe1_svm_roberta_hc3.pkl  <- modele SVM Axe 2",
        "      pca_visualization.png     <- si evaluate.py lance",
    ])

    # ═══════════════════════════════════════════════════════════
    # PAGE 5 : AXE 3
    # ═══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.axe_banner(3, "GENERATION D'ARGUMENTS CONVAINCANTS",
                   "Fine-tuning LLM + Best-of-N + Prompt Engineering")

    pdf.body_text(
        "L'Axe 3 utilise les modeles des Axes 1 et 2 pour generer des arguments convaincants. "
        "PREREQUIS : avoir lance train.py pour Axe 1 et recupere le fichier axe1_svm_*.pkl."
    )

    pdf.note_box(
        "PREREQUIS OBLIGATOIRE avant de lancer Axe 3 :\n"
        "1. Axe 1 : python train.py encoder=features  -> genere axe1_svm_features_wac.pkl\n"
        "2. Axe 2 : python train.py dataset=hc3 encoder=roberta  -> genere axe1_svm_roberta_hc3.pkl",
        WARNING
    )

    pdf.section_header("Etape 1 — Fine-tuner le LLM sur les arguments gagnants", SUCCESS)
    pdf.body_text(
        "On fine-tune GPT-2 (ou Mistral avec LoRA) sur les arguments gagnants du WAC. "
        "Le modele apprend a 'parler comme un argument convaincant'."
    )
    pdf.code_block([
        "# GPT-2 small — tourne sur CPU (~2-3 heures)",
        "python finetune.py llm=gpt2",
        "",
        "# Mistral 7B avec LoRA — necessite GPU",
        "python finetune.py llm=mistral",
        "",
        "# Sortie : dossier finetuned_gpt2/ avec le modele",
    ])

    pdf.section_header("Etape 2a — Generation Best-of-N", SUCCESS)
    pdf.body_text(
        "Genere N arguments pour chaque OP, les score avec le SVM Axe 1, et retient le meilleur."
    )
    pdf.code_block([
        "# Remplacer le chemin par votre fichier .pkl reel",
        "python generate.py strategy=best_of_n \\",
        '    axe1.model_path="outputs/2026-06-16/.../axe1_svm_features_wac.pkl" \\',
        '    axe2.model_path="outputs/2026-06-16/.../axe1_svm_roberta_hc3.pkl"',
        "",
        "# Avec le modele fine-tune specifique",
        "python generate.py strategy=best_of_n llm.model_id=finetuned_gpt2/",
        "",
        "# Changer le nombre de candidats",
        "python generate.py strategy=best_of_n strategy.n_candidates=20",
    ])

    pdf.section_header("Etape 2b — Generation par Prompt Engineering (zero-shot)", SUCCESS)
    pdf.body_text(
        "Analyse les features Axe 1 les plus discriminantes et les traduit en instructions "
        "pour guider le LLM. Aucun fine-tuning requis."
    )
    pdf.code_block([
        "python generate.py strategy=prompt_eng \\",
        '    axe1.model_path="outputs/2026-06-16/.../axe1_svm_features_wac.pkl" \\',
        '    axe2.model_path="outputs/2026-06-16/.../axe1_svm_roberta_hc3.pkl"',
        "",
        "# Changer le nombre de features utilisees pour le prompt",
        "python generate.py strategy=prompt_eng strategy.top_k_features=5",
    ])

    pdf.section_header("Outputs Axe 3", SUCCESS)
    pdf.code_block([
        "outputs/generation/",
        "  2026-06-16/",
        "    15-00-00_gpt2_best_of_n/",
        "      generate.log          <- logs de generation",
        "      generation_report.csv <- rapport complet",
        "",
        "finetuned_gpt2/             <- modele fine-tune",
    ])

    # ═══════════════════════════════════════════════════════════
    # PAGE 6 : REFERENCE RAPIDE
    # ═══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_header("REFERENCE RAPIDE — Toutes les commandes", DARK_BG)

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*DARK_BG)
    pdf.cell(0, 7, "INSTALLATION", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.code_block([
        "git clone <url> && cd Projet-DSAI",
        "python -m venv env && env\\Scripts\\activate",
        "pip install -r requirements.txt",
        "python -X utf8 scripts/download_datasets.py --all",
    ])

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*ACCENT)
    pdf.cell(0, 7, "AXE 1 — Analyse d'arguments", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.code_block([
        "python train.py encoder=features   # Recommande CPU",
        "python train.py encoder=tfidf",
        "python train.py encoder=w2v",
        "python train.py encoder=roberta    # GPU recommande",
        "python evaluate.py encoder=features",
    ])

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*ACCENT2)
    pdf.cell(0, 7, "AXE 2 — Detection synthetique", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.code_block([
        "python train.py dataset=hc3 encoder=roberta",
        "python train.py dataset=hc3 encoder=tfidf",
        "python evaluate.py dataset=hc3 encoder=roberta",
    ])

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*SUCCESS)
    pdf.cell(0, 7, "AXE 3 — Generation", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.code_block([
        "python finetune.py llm=gpt2",
        'python generate.py strategy=best_of_n axe1.model_path="..."',
        'python generate.py strategy=prompt_eng axe1.model_path="..."',
    ])

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*BLACK)
    pdf.cell(0, 7, "HYDRA — Commandes utiles", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.code_block([
        "python train.py --cfg job           # Voir la config sans lancer",
        "python train.py -m encoder=tfidf,w2v,features  # Sweep multi-run",
        "python train.py encoder=tfidf seed=123  # Override une valeur",
    ])

    pdf.section_header("STRUCTURE DU PROJET", DARK_BG)
    pdf.code_block([
        "Projet-DSAI/",
        "  configs/         <- Configs Hydra YAML",
        "    dataset/       <- wac.yaml | hc3.yaml | grid.yaml",
        "    encoder/       <- tfidf | w2v | roberta | features",
        "    model/         <- svm.yaml",
        "    generation/    <- config_generate.yaml | llm/ | strategy/",
        "  src/",
        "    data/          <- loaders & preprocessing",
        "    features/      <- features stylistiques",
        "    encoders/      <- TF-IDF, W2V, RoBERTa, Features",
        "    models/        <- SVM classifier",
        "    generation/    <- finetuning, generator, best_of_n, evaluator",
        "  datasets/        <- donnees (non versionnees)",
        "  outputs/         <- resultats Hydra (non versionnes)",
        "  train.py         <- point d'entree entrainement",
        "  evaluate.py      <- visualisation PCA",
        "  finetune.py      <- fine-tuning LLM (Axe 3)",
        "  generate.py      <- generation d'arguments (Axe 3)",
        "  scripts/         <- download_datasets.py",
    ])

    # Barre coloree finale
    pdf.set_fill_color(*ACCENT)
    pdf.rect(0, 288, 210, 4, "F")

    # Sauvegarde
    pdf.output(output_path)
    print(f"[OK] PDF genere : {output_path}")


if __name__ == "__main__":
    out = "Guide_Demarrage_Projet_DSAI.pdf"
    build_pdf(out)
