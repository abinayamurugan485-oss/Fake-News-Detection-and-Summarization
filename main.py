import warnings
warnings.filterwarnings("ignore")
import pickle
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Sumy (extractive summarization)
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
# ----------------------------
# Config
# ----------------------------
CSV_PATH = "mergeddataset.csv"          # Make sure this file is in the same folder
TEST_SIZE = 0.20
RANDOM_STATE = 42
SVD_COMPONENTS = 50
TFIDF_MAX_FEATURES = 5000


# ----------------------------
# Utils
# ----------------------------
#def summarize_text(text: str, sentences_count: int = 3) -> str:
 #   """Extractive summary with Sumy (LexRank)."""
  #  parser = PlaintextParser.from_string(text, Tokenizer("english"))
   # summarizer = LexRankSummarizer()
    #summary = summarizer(parser.document, sentences_count)
    #return " ".join(str(sentence) for sentence in summary)
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from nltk.tokenize import sent_tokenize

# Download NLTK data (run once)
nltk.download('punkt')

def summarize_text(text: str, sentences_count: int = 3) -> str:
    """Extractive summary with Sumy (LexRank) using NLTK for sentence tokenization."""
    # Split text into sentences using NLTK
    sentences = sent_tokenize(text)
    if len(sentences) < sentences_count:
        return " ".join(sentences)  # Return all if too few sentences
    
    # Join sentences back for Sumy parser
    cleaned_text = " ".join(sentences)
    parser = PlaintextParser.from_string(cleaned_text, Tokenizer("english"))
    
    # Initialize LexRank
    summarizer = LexRankSummarizer()
    # Optional: Adjust LexRank threshold for sentence similarity (default is 0.1)
    summarizer.threshold = 0.05  # Lower threshold to consider more sentence connections
    
    # Summarize
    summary = summarizer(parser.document, sentences_count)
    return " ".join(str(sentence) for sentence in summary)


def save_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def print_header(title: str):
    bar = "-" * 10
    print(f"\n{bar} {title} {bar}")


# ----------------------------
# Main pipeline
# ----------------------------
def main():
    # 1) Load data
    print_header("Loading dataset")
    csv_file = Path(CSV_PATH)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV not found: {csv_file.resolve()}")

    df = pd.read_csv(csv_file)
    # Expect these columns: title, text, label
    required_cols = {"title", "text", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Basic clean
    df = df.dropna(subset=["title", "text", "label"]).copy()
    df["full_text"] = df["text"]
    X_text = df["title"]                 # Use headline for classification
    y = df["label"].astype(int)          # Ensure numeric labels 0/1

    print(f"Total rows after cleaning: {len(df)}")

    # 2) TF-IDF
    print_header("Vectorizing (TF-IDF)")
    tfidf = TfidfVectorizer(stop_words="english", max_features=TFIDF_MAX_FEATURES)
    X_vect = tfidf.fit_transform(X_text)

    # 3) Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_vect, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 4) Random Forest
    print_header("Training Random Forest")
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"RF Accuracy: {acc_rf:.4f}")
    print(classification_report(y_test, y_pred_rf, digits=2))

    # 5) KNN (cosine) with SVD
    print_header("Preparing features for KNN (scale + SVD)")
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=RANDOM_STATE)
    X_train_reduced = svd.fit_transform(X_train_scaled)
    X_test_reduced = svd.transform(X_test_scaled)

    print_header("Training KNN (GridSearch on k)")
    #param_grid = {"n_neighbors": list(range(3, 15))}
    param_grid = {"n_neighbors": [3, 5, 7]}
    knn_base = KNeighborsClassifier(metric="cosine", weights="distance")
    knn_grid = GridSearchCV(knn_base, param_grid, cv=3, n_jobs=1)
    knn_grid.fit(X_train_reduced, y_train)
    best_k = knn_grid.best_params_["n_neighbors"]
    print(f"Best k: {best_k}")

    knn = KNeighborsClassifier(n_neighbors=best_k, metric="cosine", weights="distance")
    knn.fit(X_train_reduced, y_train)
    y_pred_knn = knn.predict(X_test_reduced)
    acc_knn = accuracy_score(y_test, y_pred_knn)

    print(f"KNN Accuracy: {acc_knn:.4f}")
    print(classification_report(y_test, y_pred_knn, digits=2))

    # Confusion matrices (optional prints)
    print_header("Confusion Matrices")
    print("RF:\n", confusion_matrix(y_test, y_pred_rf))
    print("\nKNN:\n", confusion_matrix(y_test, y_pred_knn))

    # 6) Save artifacts for reuse
    print_header("Saving models & vectorizer")
    save_pickle(tfidf, "tfidf_vectorizer.pkl")
    save_pickle(rf, "random_forest_model.pkl")
    save_pickle(scaler, "scaler.pkl")
    save_pickle(svd, "svd.pkl")
    save_pickle(knn, "knn_model.pkl")
    print("✅ Saved: tfidf_vectorizer.pkl, random_forest_model.pkl, scaler.pkl, svd.pkl, knn_model.pkl")

    # 7) Quick demo: predict + summarize on a fixed headline/text
    print_header("Demo: Predict + Summarize")
    new_headline = "Kremlin bans fuel exports until the end of the year as Russia’s supply is disrupted by Ukrainian drones"
    new_full_text = (
        "Russia is banning exports of fuel until the end of the year as gas pumps across the country and in the areas "
        "under its occupation are increasingly running dry because of Ukrainian drone attacks. "
        "Kyiv first stepped up drone attacks on Russian refineries, pumping stations and fuel trains in an attempt to "
        "disrupt fuel supply chains over the summer when demand is traditionally high as people drive more during "
        "vacation time. While Russian officials initially blamed the shortages on 'logistical reasons' and promised "
        "gasoline and diesel would begin flowing again, the shortages have only worsened in recent weeks. "
        "The Ukrainian Air Force said it struck multiple Russian fuel production sites and pumping stations this week, "
        "including a major oil refinery in Bashkortostan in southern Russia that is operated by Gazprom."
    )

    new_vec = tfidf.transform([new_headline])
    pred_rf = rf.predict(new_vec)[0]

    # KNN path: TF-IDF -> Scale -> SVD -> KNN
    new_vec_knn = svd.transform(scaler.transform(new_vec))
    pred_knn = knn.predict(new_vec_knn)[0]

    lbl = {0: "Real", 1: "Fake"}
    print("\n--- Classification ---")
    print("Random Forest Prediction:", lbl.get(int(pred_rf), pred_rf))
    print("KNN Prediction (Improved):", lbl.get(int(pred_knn), pred_knn))

    print("\n--- Extractive Summary ---")
    print(summarize_text(new_full_text, sentences_count=2))


if __name__ == "__main__":
    main()