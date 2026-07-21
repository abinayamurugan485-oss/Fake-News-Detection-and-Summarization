# 📰 Fake News Detection and Summarization

A Machine Learning web application that detects whether a news headline is **Real** or **Fake** and generates an **extractive summary** of the news article.

## 📌 Features

- Fake News Detection
- Random Forest Classifier
- K-Nearest Neighbors (KNN) Classifier
- TF-IDF Text Vectorization
- Truncated SVD for Dimensionality Reduction
- LexRank Extractive Text Summarization
- Flask Web Application
- Accuracy Comparison Chart

## 🛠 Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- NLTK
- Sumy
- HTML
- CSS

## 📂 Project Structure

```
Fake-News-Detection-and-Summarization/
│
├── app.py
├── main.py
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── accuracy_chart.png
├── tfidf_vectorizer.pkl
├── random_forest_model.pkl
├── knn_model.pkl
├── scaler.pkl
├── svd.pkl
└── README.md
```

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/abinayamurugan485-oss/Fake-News-Detection-and-Summarization.git
```

Go to project folder

```bash
cd Fake-News-Detection-and-Summarization
```

Install dependencies

```bash
pip install flask pandas scikit-learn matplotlib nltk sumy
```

Run the application

```bash
python app.py
```

Open your browser

```
http://127.0.0.1:5000
```

## 📊 Machine Learning Models

| Model | Purpose |
|--------|----------|
| Random Forest | Fake News Classification |
| KNN | Alternative Classification Model |
| TF-IDF | Text Feature Extraction |
| Truncated SVD | Feature Reduction |
| LexRank | Text Summarization |

## 📈 Accuracy

- Random Forest : **99.83%**
- KNN : **93.37%**
  
## 📷 Screenshots

### Home Page

![Home Page](screenshots/Homepage.png)

### Prediction Result

![Prediction](screenshots/Prediction.png)

### Summary Result

![Summary](screenshots/Summary.png)

## 👩‍💻 Author

**Abinaya Murugan**

GitHub:
https://github.com/abinayamurugan485-oss
