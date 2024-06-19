from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load('model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Classification model for student feedbacks"}

@app.post('/predict/{sentence}')
async def predict_feedback(feedback: str):
    feedback = vectorizer.transform([feedback])
    prediction = model.predict(feedback)
    return {"prediction": int(prediction[0])}
