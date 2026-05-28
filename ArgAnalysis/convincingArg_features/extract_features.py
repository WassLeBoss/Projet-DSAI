import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOP_WORDS = set(stopwords.words('english'))
HEDGES = {"could", "would", "may", "might", "perhaps", "possibly", "probably", "seem", "seems", "appear", "appears"}
EXAMPLES = {"for example", "for instance", "e.g."}

def safe_divide(n, d):
    return n / d if d > 0 else 0.0

def get_word_sets(text):
    if not isinstance(text, str):
        return set(), set()
    
    words = word_tokenize(text.lower())
    words_alpha = {w for w in words if w.isalnum()}
    
    stop_set = words_alpha.intersection(STOP_WORDS)
    content_set = words_alpha.difference(STOP_WORDS)
    
    return stop_set, content_set, len(words_alpha)

def extract_features(op_text, reply_text):

    features = {}
    
    if not isinstance(reply_text, str) or not isinstance(op_text, str):
        return features
        
    #prétraitement
    op_stops, op_content, op_len = get_word_sets(op_text)
    rep_stops, rep_content, rep_len = get_word_sets(reply_text)
    
    # feat1 : longueur du texte de la réponse 
    features['word_count'] = rep_len
    
    # feat2 : jaccard sur les Mots de Fonction (Stop Words) : Un vocabulaire partagé de mots de fonction peut signaler une meilleure compréhension mutuelle.
    inter_stops = rep_stops.intersection(op_stops)
    union_stops = rep_stops.union(op_stops)
    features['jaccard_stopwords'] = safe_divide(len(inter_stops), len(union_stops))
    
    # feat3 : jaccard sur les Mots de Contenu
    inter_content = rep_content.intersection(op_content)
    union_content = rep_content.union(op_content)
    features['jaccard_content'] = safe_divide(len(inter_content), len(union_content))
    
    # feat4/5 : nombre de liens
    links = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', reply_text)
    features['num_links'] = len(links)
    features['has_pdf_link'] = int(any(".pdf" in link.lower() for link in links))
    
    # feat6 : nombre d'exemples (marqueurs d'illustration)
    reply_lower = reply_text.lower()
    features['num_examples'] = sum(reply_lower.count(ex) for ex in EXAMPLES)
    
    # feat7 : nombre d'atténuations (Hedging) : Un ton prudent et modéré facilite l'acceptation.
    features['num_hedges'] = sum(1 for word in rep_content if word in HEDGES)
    
    # feat8 : ratio de questions : Poser des questions peut encourager l'engagement et la réflexion.
    num_questions = reply_text.count('?')
    features['question_ratio'] = safe_divide(num_questions, rep_len)
    
    # feat9/10 : densité de liens et d'atténuations (hedges) par mot : Pour beaucoup de métriques, il est préférable de diviser par la longueur du texte
    features['links_per_word'] = safe_divide(features['num_links'], rep_len)
    features['hedges_per_word'] = safe_divide(features['num_hedges'], rep_len)
    
    return features

# --- TEST DU PIPELINE ---
op_example = "I believe that higher taxes on the rich will destroy the economy. It discourages hard work."
reply_example = "While I understand your point, *perhaps* we should look at historical data. For example, in the 1950s taxes were higher, but the economy grew. Here is a link: http://example.com/economy.pdf. Does this make sense?"

features = extract_features(op_example, reply_example)

for key, value in features.items():
    print(f"{key}: {value:.4f}")