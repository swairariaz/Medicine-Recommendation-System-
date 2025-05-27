# Medicine Recommendation System using Machine Learning

Welcome to the **Medicine Recommendation System** — an intelligent web application built using **Python (Flask)** for the backend and **HTML, CSS, JavaScript** for the frontend. It allows users to input medical symptoms and receive a predicted disease along with detailed descriptions, precautions, medications, workouts, and diet recommendations.
## 📺 Demo

<p align="center">
  <a href="https://youtu.be/4rA8b2Zjwsk?si=l1SWaWOf9Aq0nvqY">
    <img src="https://img.youtube.com/vi/4rA8b2Zjwsk/0.jpg" alt="Watch on YouTube" width="700"/>
  </a>
</p>
> Click the image to watch the full demo on YouTube.


---
## 📌 Features

- Multi-symptom input via user interface
- Disease prediction using Support Vector Machine (SVC)
- Dynamic result rendering with:
  - Disease description
  - Medications
  - Precautions
  - Recommended workouts
  - Diet suggestions

---

## 📦 Tech Stack

### 👨‍💻 Frontend:
- HTML5
- CSS3 (with custom styling)
- JavaScript
- (via Flask templates)

### 🧪 Backend:
- Python 3
- Flask Web Framework
- Pandas for data manipulation
- scikit-learn for model training
- joblib for model persistence

---

## 🤖 Machine Learning Model

- Model: Support Vector Classifier (SVC)
- Input: List of symptoms selected by the user
- Output: Predicted disease label and useful insights

### 🔬 Trained On:
- Dataset: `Training.csv`
- Labels encoded using LabelEncoder
- Model saved as `svc.pkl`
- Label encoder saved as `label_encoder.pkl`

---

## 🗂️ Dataset Files Used:

- `Training.csv` – Main dataset used for training the model  
- `symtoms_df.csv` – Full list of symptoms  
- `precautions_df.csv` – Preventive tips per disease  
- `workout_df.csv` – Suggested physical activities  
- `description.csv` – Short disease descriptions  
- `medications.csv` – Common medications  
- `diets.csv` – Dietary advice  

---

## 🧠 Features

- 🔍 Predicts diseases based on user-selected symptoms
- 📃 Provides description of the disease
- 💊 Lists possible medications
- 🥗 Recommends diets
- 🏃 Suggests workouts
- 🛡️ Gives prevention tips

---

## 📚 Python Libraries Used

```txt
Flask
scikit-learn
pandas
numpy
joblib
notebook
```
## 🧰 Tools & Platforms Used

PyCharm – `Project development environment`

Git – `Version control`

GitHub – `Project hosting`

Kaggle – `Dataset source`

Jupyter Notebook – `Model training and experimentation`

VS Code – `Train the model`

Google Chrome – `Testing & debugging`


⭐ Final Note
This project is a demonstration of ML integration in web applications. It showcases how machine learning models can be deployed in real-world applications with practical use cases.
