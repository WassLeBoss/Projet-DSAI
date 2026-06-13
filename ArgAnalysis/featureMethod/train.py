import sys
import pandas as pd
from sklearn.linear_model  import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from data_loader import load_pairs
from pairwise import build_pairwise_dataset, get_feature_names


DEFAULT_CSV       = "/Users/tristanjin/Documents/TELECOM_PARIS/2A_COURS/SD/Projet-DSAI/Datasets/WinningArgCorpus/WAC.csv"
DEFAULT_TEST_SIZE = 0.2
RANDOM_STATE      = 42
TOP_N_FEATURES    = 5

def train_evaluate(train_csv: str,
                   test_csv:  str | None = None,
                   test_size: float = DEFAULT_TEST_SIZE,
                   random_state: int = RANDOM_STATE) -> dict:
    sep = "─" * 50

    pairs = load_pairs(train_csv)
    print(f"  {len(pairs)} paires construites")

    if test_csv:
        train_pairs = pairs
        test_pairs  = load_pairs(test_csv)
    else:
        train_pairs, test_pairs = train_test_split(
            pairs, test_size=test_size, random_state=random_state
        )

    print(f"  Train : {len(train_pairs)} | Test : {len(test_pairs)}")

    #features
    X_train, y_train = build_pairwise_dataset(train_pairs, random_state=random_state)
    X_test,  y_test  = build_pairwise_dataset(test_pairs,  random_state=random_state)
    print(f"  Dimensions X_train : {X_train.shape}")

    #normalisation
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    #modèle
    model = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=1.0,
        max_iter=1000,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    # ── Évaluation ─────────────────────────────────────────────────────────────
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)


    print(f"\nLogistic Regression L1 :")
    print(classification_report(y_test, y_pred, digits=4))

    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        auc    = roc_auc_score(y_test, y_prob)
        print(f"AUC : {auc:.4f}")
    except Exception:
        print("AUC : non définie")

    # ── Importance des features ────────────────────────────────────────────────
    print(sep)
    feature_names = get_feature_names()
    coefs = pd.Series(model.coef_[0], index=feature_names)

    print(f"\nTop {TOP_N_FEATURES} features pro-persuasion :")
    for name, val in coefs.nlargest(TOP_N_FEATURES).items():
        print(f"  {name:30s}  {val:+.4f}")

    print(f"\nTop {TOP_N_FEATURES} features anti-persuasion :")
    for name, val in coefs.nsmallest(TOP_N_FEATURES).items():
        print(f"  {name:30s}  {val:+.4f}")

    print(sep)


    return {
        "model":    model,
        "scaler":   scaler,
        "accuracy": accuracy,
        "coefs":    coefs,
        "features": feature_names,
    }


# ── Point d'entrée ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_csv = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    test_csv  = sys.argv[2] if len(sys.argv) > 2 else None

    train_evaluate(train_csv, test_csv)

