from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder


# Initialize app
app = Flask(__name__, template_folder = 'template')

# Load model
model = pickle.load(open("model/loan_pred_xgboost.pkl", "rb"))
# Load Label Encoder
#label_encoders = joblib.load('model/label_encoders.pkl')
# encoder = LabelEncoder()

"""
# All columns in model order
feature_cols = [
    'person_age', 'person_income', 'person_home_ownership', 'person_emp_length',
    'loan_intent', 'loan_grade', 'loan_amnt', 'loan_int_rate',
    'loan_percent_income', 'cb_person_default_on_file', 'cb_person_cred_hist_length'
]
"""

# Homepage
@app.route("/")
def index():
    return render_template("index.html")


# Predict
@app.route("/predict", methods=["POST"])
def predict():
    # Extract inputs from form
    inputs = [
        int(request.form["person_age"]),
        int(request.form["person_income"]),
        int(request.form["person_home_ownership"]),
        float(request.form["person_emp_length"]),
        int(request.form["loan_intent"]),
        int(request.form["loan_grade"]),
        int(request.form["loan_amnt"]),
        float(request.form["loan_int_rate"]),
        float(request.form["loan_percent_income"]),
        int(request.form["cb_person_default_on_file"]),
        int(request.form["cb_person_cred_hist_length"])
    ]
    print(inputs)
    prediction = model.predict([np.array(inputs)])

    if prediction[0] == 1:
        output = "Approved ✅"
    elif prediction[0] == 0:
        output = "Rejected ❌"
    else:
        output = ""
    return render_template("index.html", result=output)
    


if __name__ == "__main__":
    app.run(debug=True) 

    
