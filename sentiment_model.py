import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv("IMDB Dataset.csv")
df = df.sample(frac=1).reset_index(drop=True)

x_text = df["review"]
y = df["sentiment"].map({"positive":1,"negative":0})

vectorizer = CountVectorizer(stop_words="english",max_features=10000)
x = vectorizer.fit_transform(x_text)

model = LogisticRegression(max_iter=1000)
model.fit(x,y)

joblib.dump(model,"sentiment_model.pkl")
joblib.dump(vectorizer,"vectorizer.pkl")

print("Model trained and saved successfully.")