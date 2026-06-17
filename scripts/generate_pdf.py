# -*- coding: utf-8 -*-
"""
Generateur du PDF "Guide de demarrage" pour Projet-DSAI.
"""

from fpdf import FPDF
from fpdf.enums import XPos, YPos

FONT_REGULAR = "C:/Windows/Fonts/arial.ttf"
FONT_BOLD    = "C:/Windows/Fonts/arialbd.ttf"
FONT_ITALIC  = "C:/Windows/Fonts/ariali.ttf"
FONT_MONO    = "C:/Windows/Fonts/cour.ttf"
FONT_MONO_B  = "C:/Windows/Fonts/courbd.ttf"

BLACK    = (15,  15,  15)
WHITE    = (255, 255, 255)
ACCENT   = (67,  97,  238)
ACCENT2  = (114, 9,   183)
SUCCESS  = (34,  197, 94)
WARNING_COLOR = (200, 130, 0)
DARK_BG  = (30,  30,  46)
CODE_BG  = (30,  30,  46)
CODE_FG  = (166, 226, 46)
GRAY     = (100, 100, 120)

AXE_COLORS = {1: ACCENT, 2: ACCENT2, 3: SUCCESS}


class GuidePDF(FPDF):

    def setup_fonts(self):
        self.add_font("Sans",  "",  FONT_REGULAR)
        self.add_font("Sans",  "B", FONT_BOLD)
        self.add_font("Sans",  "I", FONT_ITALIC)
        self.add_font("Mono",  "",  FONT_MONO)
        self.add_font("Mono",  "B", FONT_MONO_B)

    def header(self):
        pass

    def footer(self):
        self.set_y(-14)
        self.set_font("Sans", "I", 8)
        self.set_text_color(*GRAY)
        self.cell(0, 10,
                  f"Projet-DSAI - Guide de demarrage   |   Page {self.page_no()}",
                  align="C")

    # ── Composants UI ────────────────────────────────────────────────────────

    def cover_page(self):
        self.set_fill_color(*DARK_BG)
        self.rect(0, 0, 210, 297, "F")

        self.set_fill_color(*ACCENT)
        self.rect(0, 0, 210, 5, "F")

        self.set_y(55)
        self.set_font("Sans", "B", 34)
        self.set_text_color(*WHITE)
        self.cell(0, 14, "PROJET-DSAI", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.set_font("Sans", "", 16)
        self.set_text_color(180, 180, 210)
        self.cell(0, 10, "Guide de demarrage complet", align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.ln(5)
        self.set_font("Sans", "I", 11)
        self.set_text_color(140, 140, 170)
        self.cell(0, 8, "De zero a la generation d'arguments convaincants",
                  align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.ln(18)
        labels = ["AXE 1 - Analyse", "AXE 2 - Detection", "AXE 3 - Generation"]
        colors = [ACCENT, ACCENT2, SUCCESS]
        x = 24
        for label, color in zip(labels, colors):
            self.set_fill_color(*color)
            self.set_text_color(*WHITE)
            self.set_font("Sans", "B", 9)
            self.set_xy(x, self.get_y())
            self.cell(54, 10, label, align="C", fill=True, new_x=XPos.RIGHT)
            x += 57

        self.set_y(200)
        self.set_font("Sans", "", 9)
        self.set_text_color(130, 130, 160)
        self.cell(0, 8,
                  "Python 3.10+  |  PyTorch  |  scikit-learn  |  Hydra  |  HuggingFace",
                  align="C")

        self.set_y(245)
        self.set_fill_color(40, 40, 60)
        self.rect(0, 243, 210, 32, "F")
        self.set_font("Sans", "B", 10)
        self.set_text_color(*WHITE)
        self.cell(0, 7, "Ordre d'execution recommande", align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)
        steps = [
            "1. Installation & Datasets",
            "2. Axe 1 : train.py encoder=features",
            "3. Axe 2 : train.py dataset=hc3 encoder=roberta",
            "4. Axe 3 : finetune.py + generate.py",
        ]
        self.set_font("Sans", "", 9)
        self.set_text_color(200, 200, 220)
        for s in steps:
            self.cell(0, 5.5, f"    {s}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def section_header(self, title: str, color: tuple = ACCENT):
        self.ln(5)
        self.set_fill_color(*color)
        self.rect(self.l_margin, self.get_y(), 190, 0.5, "F")
        self.ln(2)
        self.set_font("Sans", "B", 12)
        self.set_text_color(*color)
        self.cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*BLACK)
        self.ln(1)

    def axe_banner(self, n: int, title: str, subtitle: str):
        color = AXE_COLORS[n]
        self.set_fill_color(*color)
        self.set_text_color(*WHITE)
        self.set_font("Sans", "B", 13)
        self.cell(0, 11, f"  AXE {n} - {title}", fill=True,
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        r, g, b = color
        self.set_fill_color(min(r+40, 255), min(g+40, 255), min(b+40, 255))
        self.set_font("Sans", "I", 9)
        self.cell(0, 6, f"  {subtitle}", fill=True,
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*BLACK)
        self.ln(3)

    def body_text(self, text: str, size: int = 10, link: str = ""):
        self.set_font("Sans", "", size)
        self.set_text_color(*BLACK)
        if link:
            self.set_text_color(0, 0, 255) # Blue for link
            self.cell(0, 5.5, text, link=link, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_text_color(*BLACK)
        else:
            self.multi_cell(0, 5.5, text)
        self.ln(1)

    def bullet(self, text: str):
        self.set_x(self.l_margin + 5)
        self.set_font("Sans", "", 10)
        self.set_text_color(*BLACK)
        self.cell(4, 5.5, "-")
        self.set_x(self.l_margin + 9)
        self.multi_cell(0, 5.5, text)

    def note_box(self, text: str, color: tuple = WARNING_COLOR):
        y = self.get_y()
        self.set_fill_color(*color)
        self.rect(self.l_margin, y, 2.5, 20, "F")
        self.set_fill_color(245, 245, 250)
        self.rect(self.l_margin + 2.5, y, 187.5, 20, "F")
        self.set_xy(self.l_margin + 5, y + 2)
        self.set_font("Sans", "B", 8)
        self.set_text_color(*color)
        self.cell(14, 5, "NOTA")
        self.set_font("Sans", "", 8.5)
        self.set_text_color(50, 50, 60)
        self.set_x(self.l_margin + 18)
        self.multi_cell(180, 5, text)
        self.set_text_color(*BLACK)
        self.ln(3)

    def code_block(self, lines: list, title: str = ""):
        if title:
            self.set_font("Sans", "I", 7.5)
            self.set_text_color(*GRAY)
            self.cell(0, 5, f"  # {title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        h = 5.2
        total = h * len(lines) + 6
        x = self.l_margin
        y = self.get_y()
        self.set_fill_color(*CODE_BG)
        self.rect(x, y, 190, total, "F")
        self.set_y(y + 3)
        for line in lines:
            self.set_x(x + 4)
            self.set_font("Mono", "", 8)
            if line.strip().startswith("#"):
                self.set_text_color(117, 113, 94)
            elif any(line.strip().startswith(k) for k in ("python ", "pip ", "git ", "cd ")):
                self.set_text_color(*CODE_FG)
            else:
                self.set_text_color(248, 248, 242)
            self.cell(0, h, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*BLACK)
        self.ln(3)

    def step_num(self, n: int, title: str):
        self.set_fill_color(*ACCENT)
        y = self.get_y()
        self.rect(self.l_margin, y, 7, 7, "F")
        self.set_xy(self.l_margin, y)
        self.set_font("Sans", "B", 8)
        self.set_text_color(*WHITE)
        self.cell(7, 7, str(n), align="C")
        self.set_font("Sans", "B", 11)
        self.set_text_color(*DARK_BG)
        self.set_x(self.l_margin + 10)
        self.cell(0, 7, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*BLACK)
        self.ln(1)


# ── Construction du PDF ──────────────────────────────────────────────────────

def build_pdf(output_path: str):
    pdf = GuidePDF()
    pdf.setup_fonts()
    pdf.set_margins(12, 12, 12)
    pdf.set_auto_page_break(auto=True, margin=18)

    # ── COUVERTURE ───────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.cover_page()

    # ── PAGE 2 : INSTALLATION ────────────────────────────────────────────────
    pdf.add_page()
    pdf.section_header("0. PREREQUIS & INSTALLATION", DARK_BG)

    pdf.step_num(1, "Prerequisites systeme")
    pdf.bullet("Python 3.10 ou superieur  (verifier : python --version)")
    pdf.bullet("Git installe et configure")
    pdf.bullet("8 Go de RAM minimum (16 Go recommandes)")
    pdf.bullet("GPU optionnel (necessaire uniquement pour RoBERTa / Mistral 7B)")
    pdf.ln(3)

    pdf.step_num(2, "Cloner le depot Git")
    pdf.code_block([
        "git clone https://github.com/votre-org/Projet-DSAI.git",
        "cd Projet-DSAI",
    ], "Clonage et navigation")

    pdf.step_num(3, "Creer et activer l'environnement virtuel")
    pdf.body_text("Windows :")
    pdf.code_block([
        "python -m venv env",
        "env\\Scripts\\activate",
        "# Le prompt affiche maintenant : (env) ...",
    ], "Windows")
    pdf.body_text("macOS / Linux :")
    pdf.code_block([
        "python -m venv env",
        "source env/bin/activate",
    ], "macOS / Linux")
    pdf.note_box(
        "Toujours activer le venv avant de lancer une commande. "
        "Le prompt doit afficher (env) au debut.",
        WARNING_COLOR
    )

    pdf.step_num(4, "Installer les dependances")
    pdf.code_block([
        "pip install --upgrade pip",
        "pip install -r requirements.txt",
        "",
        "# Dependances principales :",
        "# numpy, pandas, scikit-learn, matplotlib",
        "# nltk, gensim, sentence-transformers, transformers, torch",
        "# hydra-core, omegaconf, convokit, datasets, fpdf2",
    ], "Installation")

    pdf.step_num(5, "Telecharger les datasets")
    
    pdf.body_text("Le projet utilise deux datasets principaux :")
    pdf.bullet("WinningArgCorpus (WAC) pour les Axes 1 et 3")
    pdf.bullet("Human ChatGPT Comparison Corpus (HC3) pour l'Axe 2")
    
    pdf.ln(2)
    pdf.body_text("Source originelle WAC : ", size=9)
    pdf.body_text("https://convokit.cornell.edu/documentation/winning.html", size=9, link="https://convokit.cornell.edu/documentation/winning.html")
    
    pdf.body_text("Source originelle HC3 : ", size=9)
    pdf.body_text("https://huggingface.co/datasets/Hello-SimpleAI/HC3", size=9, link="https://huggingface.co/datasets/Hello-SimpleAI/HC3")
    
    pdf.ln(3)
    pdf.body_text("Pour tout telecharger et formatter correctement :")
    
    pdf.code_block([
        "# WinningArgCorpus (Axes 1 & 3)",
        "python -X utf8 scripts/download_datasets.py --dataset wac",
        "",
        "# HC3 - Human ChatGPT Comparison Corpus (Axe 2)",
        "python -X utf8 scripts/download_datasets.py --dataset hc3",
        "",
        "# Ou les deux d'un coup :",
        "python -X utf8 scripts/download_datasets.py --all",
    ], "Telechargement automatique")
    pdf.note_box(
        "Les datasets seront formattes et sauvegardes dans :\n"
        "  datasets/WinningArgCorpus/WAC.csv  (19 714 arguments)\n"
        "  datasets/HC3/train.csv             (17 112 questions)\n\n"
        "Si le telechargement echoue, vous pouvez telecharger les sources manuellement\n"
        "et les placer dans ces chemins exacts.",
        SUCCESS
    )

    # ── PAGE 3 : AXE 1 ──────────────────────────────────────────────────────
    pdf.add_page()
    pdf.axe_banner(1, "ANALYSE D'ARGUMENTS",
                   "Predire quels arguments convainquent l'auteur du post (WAC dataset)")

    pdf.body_text(
        "L'Axe 1 entraine un SVM sur le WinningArgCorpus pour predire si un argument "
        "va convaincre l'auteur du post original (label success=1). "
        "Quatre encodeurs sont disponibles. Chaque run sauvegarde le modele SVM (.pkl) "
        "dans le dossier outputs/ pour reutilisation par l'Axe 3."
    )

    pdf.section_header("Approche A - Features stylistiques pairwise  [RECOMMANDEE CPU]", ACCENT)
    pdf.body_text(
        "Calcule ~50 features numeriques (nb mots, hedges, liens, lisibilite Flesch-Kincaid...) "
        "et compare paires (gagnant, perdant). La plus rapide sur CPU."
    )
    pdf.code_block([
        "python train.py encoder=features",
        "",
        "# Sortie attendue :",
        "#   Train : ~7 000 exemples | Test : ~1 800 exemples",
        "#   AUC final : 0.55 - 0.65",
        "#   Modele -> outputs/.../axe1_svm_features_wac.pkl",
    ])

    pdf.section_header("Approche B - TF-IDF + SVM  [Baseline rapide]", ACCENT)
    pdf.body_text("Representation sac-de-mots pondere. Bon point de comparaison baseline.")
    pdf.code_block([
        "python train.py encoder=tfidf",
        "# AUC attendu : ~0.58 - 0.64",
    ])

    pdf.section_header("Approche C - Word2Vec + SVM", ACCENT)
    pdf.body_text("Embeddings de mots (100d) entraines sur le corpus, moyenne par texte.")
    pdf.code_block([
        "python train.py encoder=w2v",
    ])

    pdf.section_header("Approche D - RoBERTa + SVM  [GPU recommande]", ACCENT)
    pdf.body_text(
        "Embeddings contextuels pre-entraines (1 024 dimensions). "
        "Meilleure performance attendue mais tres lent sur CPU."
    )
    pdf.code_block([
        "python train.py encoder=roberta",
        "# ATTENTION : ~30-60 min sur CPU",
    ])
    pdf.note_box(
        "ORDRE RECOMMANDE SUR CPU : features (< 5 min) -> tfidf (~3 min) "
        "-> w2v (~8 min) -> roberta (GPU de preference)",
        WARNING_COLOR
    )

    pdf.section_header("Sweep et visualisation", ACCENT)
    pdf.code_block([
        "# Lancer plusieurs encodeurs en sequence",
        "python train.py -m encoder=features,tfidf,w2v",
        "",
        "# Visualisation PCA des representations",
        "python evaluate.py encoder=features",
        "python evaluate.py encoder=tfidf",
        "",
        "# Afficher la config sans lancer",
        "python train.py --cfg job",
    ])

    pdf.section_header("Sorties Axe 1", ACCENT)
    pdf.code_block([
        "outputs/2026-06-16/15-30-00_features_wac/",
        "  train.log                      <- rapport complet (AUC, classification report)",
        "  .hydra/config.yaml             <- config exacte du run",
        "  axe1_svm_features_wac.pkl      <- MODELE SVM (necessaire pour Axe 3)",
        "  pca_visualization.png          <- si evaluate.py lance",
    ])

    # ── PAGE 4 : AXE 2 ──────────────────────────────────────────────────────
    pdf.add_page()
    pdf.axe_banner(2, "DETECTION DE TEXTES SYNTHETIQUES",
                   "Distinguer textes humains vs generes par ChatGPT (HC3 dataset)")

    pdf.body_text(
        "L'Axe 2 entraine un SVM sur le dataset HC3 pour detecter si un texte a ete ecrit "
        "par un humain (label 0) ou par ChatGPT (label 1). "
        "Le pipeline est identique a l'Axe 1 - seul le dataset change."
    )

    pdf.note_box(
        "Prealable : avoir telecharge HC3 :\n"
        "  python -X utf8 scripts/download_datasets.py --dataset hc3",
        ACCENT2
    )

    pdf.section_header("Commandes (ordre recommande)", ACCENT2)
    pdf.code_block([
        "# Approche A : RoBERTa + SVM (meilleure performance)",
        "python train.py dataset=hc3 encoder=roberta",
        "",
        "# Approche B : TF-IDF + SVM (baseline rapide sur CPU)",
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
        "# Visualiser la separation humain vs ChatGPT dans l'espace 2D",
        "python evaluate.py dataset=hc3 encoder=roberta",
    ])

    pdf.section_header("Sorties Axe 2", ACCENT2)
    pdf.code_block([
        "outputs/2026-06-16/15-45-00_roberta_hc3/",
        "  train.log",
        "  axe1_svm_roberta_hc3.pkl       <- MODELE SVM Axe 2 (necessaire pour Axe 3)",
        "  pca_visualization.png",
    ])

    # ── PAGE 5 : AXE 3 ──────────────────────────────────────────────────────
    pdf.add_page()
    pdf.axe_banner(3, "GENERATION D'ARGUMENTS CONVAINCANTS",
                   "Fine-tuning LLM + Best-of-N + Prompt Engineering")

    pdf.body_text(
        "L'Axe 3 utilise les modeles des Axes 1 et 2 pour generer des arguments convaincants. "
        "Deux strategies : (A) Fine-tuning du LLM sur WAC + selection Best-of-N par le SVM Axe 1, "
        "et (B) Prompt Engineering derive des features Axe 1 (zero-shot, pas de fine-tuning)."
    )

    pdf.note_box(
        "PREREQUIS OBLIGATOIRES avant Axe 3 :\n"
        "  1. Axe 1 entraine : outputs/.../axe1_svm_features_wac.pkl\n"
        "  2. Axe 2 entraine : outputs/.../axe1_svm_roberta_hc3.pkl\n"
        "Notez les chemins exacts de ces fichiers .pkl avant de continuer.",
        WARNING_COLOR
    )

    pdf.section_header("Etape 1 - Fine-tuner le LLM sur les arguments gagnants", SUCCESS)
    pdf.body_text(
        "On fine-tune GPT-2 (ou Mistral avec LoRA) en next-token prediction uniquement "
        "sur les arguments success=1 du WAC. Le modele apprend le style des arguments convaincants."
    )
    pdf.code_block([
        "# GPT-2 small - fonctionne sur CPU (~2-3 heures)",
        "python finetune.py llm=gpt2",
        "",
        "# Mistral 7B avec LoRA - GPU requis",
        "python finetune.py llm=mistral",
        "",
        "# Sortie : dossier finetuned_gpt2/ contenant le modele",
    ])

    pdf.section_header("Etape 2a - Generation Best-of-N", SUCCESS)
    pdf.body_text(
        "Pour chaque post original (OP) : genere N arguments, les score avec le SVM Axe 1, "
        "retient le meilleur. La selection utilise la fonction de decision du SVM comme oracle."
    )
    pdf.code_block([
        "# Remplacer les chemins par vos fichiers .pkl reels",
        "python generate.py strategy=best_of_n",
        '    axe1.model_path="outputs/2026-06-16/.../axe1_svm_features_wac.pkl"',
        '    axe2.model_path="outputs/2026-06-16/.../axe1_svm_roberta_hc3.pkl"',
        "",
        "# Avec le modele fine-tune specifique",
        "python generate.py strategy=best_of_n llm.model_id=finetuned_gpt2/",
        "",
        "# Changer le nombre de candidats (defaut : 10)",
        "python generate.py strategy=best_of_n strategy.n_candidates=20",
    ])

    pdf.section_header("Etape 2b - Generation par Prompt Engineering (zero-shot)", SUCCESS)
    pdf.body_text(
        "Analyse les features Axe 1 les plus discriminantes (via les coefficients du SVM) "
        "et les traduit en instructions naturelles dans le prompt. Pas de fine-tuning requis."
    )
    pdf.code_block([
        "python generate.py strategy=prompt_eng",
        '    axe1.model_path="outputs/2026-06-16/.../axe1_svm_features_wac.pkl"',
        '    axe2.model_path="outputs/2026-06-16/.../axe1_svm_roberta_hc3.pkl"',
        "",
        "# Ex. d'instructions generees depuis Axe 1 :",
        "#   'Use at least 2 concrete examples'",
        "#   'Cite external sources with links'",
        "#   'Keep sentences under 20 words'",
    ])

    pdf.section_header("Sorties Axe 3", SUCCESS)
    pdf.code_block([
        "outputs/generation/2026-06-16/15-00_gpt2_best_of_n/",
        "  generate.log              <- logs de generation",
        "  generation_report.csv     <- rapport : axe1_score, axe2_authenticity",
        "",
        "finetuned_gpt2/             <- modele GPT-2 fine-tune (reutilisable)",
    ])

    # ── PAGE 6 : REFERENCE RAPIDE ────────────────────────────────────────────
    pdf.add_page()
    pdf.section_header("REFERENCE RAPIDE - Toutes les commandes dans l'ordre", DARK_BG)

    blocks = [
        ("INSTALLATION", DARK_BG, [
            "git clone <url-du-repo> && cd Projet-DSAI",
            "python -m venv env",
            "env\\Scripts\\activate          # Windows",
            "# source env/bin/activate    # macOS/Linux",
            "pip install -r requirements.txt",
            "python -X utf8 scripts/download_datasets.py --all",
        ]),
        ("AXE 1 - Analyse d'arguments (WAC)", ACCENT, [
            "python train.py encoder=features       # Recommande CPU - ~5 min",
            "python train.py encoder=tfidf          # Baseline - ~3 min",
            "python train.py encoder=w2v            # ~8 min",
            "python train.py encoder=roberta        # GPU recommande",
            "python evaluate.py encoder=features    # Visualisation PCA",
        ]),
        ("AXE 2 - Detection synthetique (HC3)", ACCENT2, [
            "python train.py dataset=hc3 encoder=roberta    # GPU recommande",
            "python train.py dataset=hc3 encoder=tfidf      # Baseline CPU",
            "python evaluate.py dataset=hc3 encoder=roberta # Visualisation",
        ]),
        ("AXE 3 - Generation d'arguments", SUCCESS, [
            "python finetune.py llm=gpt2             # Fine-tuning ~2-3h CPU",
            "python generate.py strategy=best_of_n \\",
            '    axe1.model_path="outputs/.../axe1_svm_features_wac.pkl" \\',
            '    axe2.model_path="outputs/.../axe1_svm_roberta_hc3.pkl"',
            "",
            "python generate.py strategy=prompt_eng \\",
            '    axe1.model_path="outputs/.../axe1_svm_features_wac.pkl" \\',
            '    axe2.model_path="outputs/.../axe1_svm_roberta_hc3.pkl"',
        ]),
        ("HYDRA - Commandes utiles", GRAY, [
            "python train.py --cfg job                 # Voir config sans lancer",
            "python train.py -m encoder=tfidf,w2v,features  # Sweep multi-run",
            "python train.py encoder=tfidf seed=99    # Override n'importe quelle valeur",
        ]),
    ]

    for title, color, lines in blocks:
        self_ref = pdf
        self_ref.set_font("Sans", "B", 9)
        self_ref.set_text_color(*color)
        self_ref.cell(0, 6, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.code_block(lines)
        pdf.ln(1)

    pdf.section_header("STRUCTURE DU PROJET", DARK_BG)
    pdf.code_block([
        "Projet-DSAI/",
        "  configs/                <- Configs Hydra YAML",
        "    dataset/               <- wac.yaml | hc3.yaml | grid.yaml",
        "    encoder/               <- tfidf | w2v | roberta | features",
        "    model/                 <- svm.yaml",
        "    generation/            <- config_generate.yaml | llm/ | strategy/",
        "  src/",
        "    data/                  <- loaders & preprocessing",
        "    features/              <- features stylistiques",
        "    encoders/              <- TF-IDF, W2V, RoBERTa, Features",
        "    models/                <- SVM classifier (save/load)",
        "    generation/            <- finetuning, generator, best_of_n, evaluator",
        "  datasets/                <- donnees brutes (non versionnees)",
        "  outputs/                 <- resultats Hydra (non versionnes)",
        "  train.py                 <- entree entrainement",
        "  evaluate.py              <- visualisation PCA",
        "  finetune.py              <- fine-tuning LLM (Axe 3)",
        "  generate.py              <- generation d'arguments (Axe 3)",
        "  scripts/                 <- download_datasets.py, generate_pdf.py",
    ])

    # Barre finale
    pdf.set_fill_color(*ACCENT)
    pdf.rect(0, 291, 210, 4, "F")

    pdf.output(output_path)
    print(f"[OK] PDF genere : {output_path}")


if __name__ == "__main__":
    build_pdf("Guide_Demarrage_Projet_DSAI.pdf")
