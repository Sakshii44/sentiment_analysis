💬 Sentiment Analysis API using FastAPI

This project is a simple Sentiment Analysis API that uses a Logistic Regression model to classify input text as **positive** or **negative** sentiment. It is built using FastAPI and Scikit-learn.

---

## 🔧 Technologies Used

- Logistic Regression (Scikit-learn)
- TF-IDF Vectorizer
- FastAPI
- Uvicorn
- Joblib

---

## 📁 Project Structure

sentiment-api/
├── train_model.py # Trains and saves the model and vectorizer
├── app.py # FastAPI app to expose /predict API
├── sentiment_model.pkl # Saved Logistic Regression model
├── vectorizer.pkl # Saved TF-IDF vectorizer
└── README.md # Project documentation

