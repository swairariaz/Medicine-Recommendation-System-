from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import json  # ✅ Added for JSON decoding

app = Flask(__name__)

# Load the model and label encoder
model = joblib.load('svc.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Load all datasets from the datasets/ folder
dataset = pd.read_csv('datasets/Training.csv')
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions_df = pd.read_csv("datasets/precautions_df.csv")
workout_df = pd.read_csv("datasets/workout_df.csv")
description_df = pd.read_csv("datasets/description.csv")
medications_df = pd.read_csv('datasets/medications.csv')
diets_df = pd.read_csv("datasets/diets.csv")

# Extract symptom columns
symptom_columns = dataset.columns[:-1]

# ✅ Normalize column names
workout_df.rename(columns={'disease': 'Disease'}, inplace=True)

# Drop unnamed columns
workout_df = workout_df.loc[:, ~workout_df.columns.str.contains('^Unnamed')]


def predict_disease(symptoms):
    """Predicts disease based on selected symptoms and retrieves additional information."""
    input_data = [1 if symptom.strip() in symptoms else 0 for symptom in symptom_columns]
    input_df = pd.DataFrame([input_data], columns=symptom_columns)

    # Ensure model is working properly
    try:
        prediction = model.predict(input_df)[0]
        disease = label_encoder.inverse_transform([prediction])[0]
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

    # Fetch info from all datasets safely
    def safe_fetch(df, column, disease):
        result = df[df['Disease'] == disease][column].values
        return result[0] if len(result) else "Information not available"

    description = safe_fetch(description_df, 'Description', disease)
    precaution = precautions_df[precautions_df['Disease'] == disease].drop('Disease', axis=1).values.flatten().tolist()
    medication = medications_df[medications_df['Disease'] == disease].drop('Disease', axis=1).values.flatten().tolist()
    workout = workout_df[workout_df['Disease'] == disease].drop('Disease', axis=1).values.flatten().tolist()
    diet = diets_df[diets_df['Disease'] == disease].drop('Disease', axis=1).values.flatten().tolist()

    return {
        'disease': disease,
        'description': description,
        'precaution': precaution,
        'medications': medication,
        'workouts': workout,
        'diet': diet
    }


@app.route('/')
def index():
    """Renders the main page with the list of symptoms."""
    return render_template('index.html', symptom_list=symptom_columns)


@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction requests from the frontend."""
    try:
        # ✅ Decode JSON symptoms correctly
        symptoms_raw = request.form.get("symptoms")
        if not symptoms_raw:
            return render_template('result.html', error="No symptoms provided. Please select at least one symptom.")

        selected_symptoms = json.loads(symptoms_raw)  # Convert JSON string to Python list


        valid_symptoms = [sym for sym in selected_symptoms if sym in symptom_columns]
        if not valid_symptoms:
            return render_template('result.html', error="Invalid symptoms selected. Please choose valid symptoms.")

        result = predict_disease(valid_symptoms)


        return render_template('result.html', result=result)

    except Exception as e:
        return render_template('result.html', error=f"Server error: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)
