from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
import joblib
import traceback

try:
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except Exception:
    traceback.print_exc()
    raise RuntimeError("Failed to load model/vectorizer")

app = FastAPI(title = "Sentiment Analysis API")

class TextInput(BaseModel):
    text:str

@app.post("/predict")
def predict_sentiment(input_data:TextInput):
    if not input_data.text:
        raise HTTPException(status_code=400,detail="Text cannot be empty")
    
    try:
        x = vectorizer.transform([input_data.text])
        pred = model.predict(x)[0]
        prob = model.predict_proba(x)[0][pred]
        sentiment = "Positive" if pred == 1 else "Negative"
        return{
            "text":input_data.text,
            "prediction":sentiment,
            "confidence":round(prob,2)
        }
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))


