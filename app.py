# app.py
from flask import Flask, render_template, request, redirect, url_for, flash
import pickle
from pathlib import Path
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import matplotlib
matplotlib.use("Agg")  # backend for servers
import matplotlib.pyplot as plt
import io
import os
import time

app = Flask(__name__)
app.secret_key = "replace_with_a_random_secret_key"

# ---- Paths to your pickles (make sure filenames match) ----
TFIDF_PATH = Path("tfidf_vectorizer.pkl")
RF_PATH = Path("random_forest_model.pkl")
SCALER_PATH = Path("scaler.pkl")
SVD_PATH = Path("svd.pkl")
KNN_PATH = Path("knn_model.pkl")
# -----------------------------------------------------------

# Hard-coded accuracies you gave (use as-is)
RF_ACC = 0.9983
KNN_ACC = 0.9337

# Helper: load pickle with friendly error
def load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path.resolve()}")
    with open(path, "rb") as f:
        return pickle.load(f)

# Load models at startup (if missing, show startup_error on page)
try:
    tfidf = load_pickle(TFIDF_PATH)
    rf_model = load_pickle(RF_PATH)
    scaler = load_pickle(SCALER_PATH)
    svd = load_pickle(SVD_PATH)
    knn_model = load_pickle(KNN_PATH)
    startup_error = None
except Exception as e:
    tfidf = rf_model = scaler = svd = knn_model = None
    startup_error = str(e)

# Summarizer (Sumy LexRank)
def summarize_text(text: str, sentences_count: int = 2) -> str:
    if not text or not text.strip():
        return ""
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        summary_sentences = summarizer(parser.document, sentences_count)
        return " ".join(str(s) for s in summary_sentences)
    except Exception:
        return ""  # fallback: empty summary if Sumy fails

# Generate & save accuracy pie chart to static folder
def generate_accuracy_pie(save_path: str = "static/accuracy_chart.png"):
    # Use the numeric accuracies as comparative sizes.
    sizes = [RF_ACC, KNN_ACC]
    labels = [f"Random Forest\n{RF_ACC*100:.2f}%", f"KNN\n{KNN_ACC*100:.2f}%"]

    # Ensure static dir exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.clf()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(sizes, labels=labels, autopct=None, startangle=140)
    ax.axis("equal")
    ax.set_title("Accuracy Comparison (RF vs KNN)")

    # Add legend with exact values
    legend_labels = [f"RF: {RF_ACC*100:.2f}%", f"KNN: {KNN_ACC*100:.2f}%"]
    ax.legend(legend_labels, loc="lower center")

    # Save chart
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

# Map label to display text & bootstrap color
LABEL_MAP = {0: ("Real", "success"), 1: ("Fake", "danger")}

@app.route("/", methods=["GET", "POST"])
def index():
    global startup_error
    # (re)generate chart every time to keep file updated
    chart_path = "static/accuracy_chart.png"
    try:
        generate_accuracy_pie(chart_path)
    except Exception as e:
        # if chart generation fails, continue but show message
        flash_msg = f"Chart generation error: {e}"
        app.logger.error(flash_msg)

    if request.method == "POST":
        title = request.form.get("title", "").strip()
        full_text = request.form.get("full_text", "").strip()
        summary_len = int(request.form.get("summary_len", 2))
        selected_model = request.form.get("model_select", "rf")  # 'rf' or 'knn'

        if not title:
            flash("Please enter a headline/title for detection.", "warning")
            return redirect(url_for("index"))

        if startup_error:
            flash(f"Startup error loading models: {startup_error}", "danger")
            return redirect(url_for("index"))

        try:
            x_tfidf = tfidf.transform([title])

            if selected_model == "rf":
                pred = rf_model.predict(x_tfidf)[0]
                display_model = "Random Forest"
                # attempt probability if available
                try:
                    prob = rf_model.predict_proba(x_tfidf)[0].max()
                except Exception:
                    prob = None
            else:
                # KNN path: tfidf -> scaler -> svd -> knn
                x_scaled = scaler.transform(x_tfidf)
                x_reduced = svd.transform(x_scaled)
                pred = knn_model.predict(x_reduced)[0]
                display_model = "KNN"
                try:
                    prob = knn_model.predict_proba(x_reduced)[0].max()
                except Exception:
                    prob = None

            label_text, badge_class = LABEL_MAP.get(int(pred), (str(pred), "secondary"))
            summary = summarize_text(full_text, sentences_count=summary_len) if full_text else ""

            # Add timestamp to image src to avoid caching
            ts = int(time.time())
            return render_template(
                "index.html",
                title_input=title,
                full_text=full_text,
                summary=summary,
                model_selected=selected_model,
                display_model=display_model,
                label_text=label_text,
                badge_class=badge_class,
                prob=prob,
                chart_ts=ts,
                startup_error=None,
            )
        except Exception as e:
            app.logger.exception("Prediction error")
            flash(f"Error during prediction: {e}", "danger")
            return redirect(url_for("index"))

    # GET request
    ts = int(time.time())
    return render_template("index.html", chart_ts=ts, startup_error=startup_error)

if __name__ == "__main__":
    app.run(debug=True)