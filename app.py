from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
import os

app = Flask(__name__)

# Correctly points to the 'Model' folder inside your project directory
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'Model', 'heart.pickle')

try:
    with open(MODEL_PATH, 'rb') as file:
        model1 = pickle.load(file)
    print("✓ Model loaded successfully!")
except FileNotFoundError:
    print(f"❌ Error: 'heart.pickle' missing at {MODEL_PATH}. Check your folder names.")
    model1 = None

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/prediction")
def prediction():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            # Safely extract all 13 values matching the HTML input 'name' attributes
            data = [
                float(request.form.get('age', 0)),
                float(request.form.get('sex', 0)),
                float(request.form.get('cp', 0)),
                float(request.form.get('trestbps', 0)),
                float(request.form.get('chol', 0)),
                float(request.form.get('fbs', 0)),
                float(request.form.get('restecg', 0)),
                float(request.form.get('thalach', 0)),
                float(request.form.get('exang', 0)),
                float(request.form.get('oldpeak', 0)),
                float(request.form.get('slope', 0)),
                float(request.form.get('ca', 0)),
                float(request.form.get('thal', 0))
            ]
            
            if model1 is None:
                return "Backend Error: Model file could not be loaded. Check your VS Code terminal."

            arr = np.array([data])
            pred = model1.predict(arr)
            
            if pred[0] == 0:
                return render_template(
                    "result.html",
                    prediction=0,
                    prediction_text1="Absence of Heart Disease",
                    precautions="Continue regular exercise, maintain healthy weight and avoid smoking.",
                    medications="No medication required unless advised by your doctor.",
                    workout_plan="30 minutes walking, cycling or jogging for at least 5 days/week.",
                    diet_plan="Consume fruits, vegetables, whole grains and lean protein.",
                    food_recommendation="Oats, nuts, fish, berries, spinach and olive oil."
                )
            else:
                return render_template(
                    "result.html",
                    prediction=1,
                    prediction_text1="Presence of Heart Disease",
                    precautions="Consult a cardiologist immediately and follow medical advice.",
                    medications="Follow prescribed medications such as statins or beta blockers.",
                    workout_plan="Only light exercise under doctor's supervision.",
                    diet_plan="Low sodium, low fat diet with plenty of vegetables.",
                    food_recommendation="Leafy greens, salmon, oats, beans, walnuts and fruits."
                )
        except Exception as e:
            return f"Data Processing Error: {str(e)}. Ensure all form inputs contain numbers."

    return redirect(url_for('prediction'))

if __name__ == "__main__":
    app.run(debug=True)