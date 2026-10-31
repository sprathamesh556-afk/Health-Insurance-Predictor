import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 1. LOAD MODEL AND SCALER ARTIFACTS
# ==========================================
model_young = joblib.load(os.path.join(BASE_DIR, "Artifacts", "model_young.joblib"))
model_rest = joblib.load(os.path.join(BASE_DIR, "Artifacts", "model_rest.joblib"))

scaler_young_data = joblib.load(os.path.join(BASE_DIR, "Artifacts", "scaler_young.joblib"))
scaler_rest_data = joblib.load(os.path.join(BASE_DIR, "Artifacts", "scaler_rest.joblib"))

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def calculate_normalized_risk(medical_history):
    if not isinstance(medical_history, str):
        return 0.0
    risk_scores = {"diabetes": 6, "heart disease": 8, "high blood pressure": 6, "thyroid": 5, "no disease": 0, "none": 0}
    diseases = medical_history.lower().split(" & ")
    total_risk_score = sum(risk_scores.get(disease.strip(), 0) for disease in diseases)
    return (total_risk_score - 0) / (14 - 0)

def preprocess_input(input_dict):
    features = {
        'age': input_dict.get('Age', 0),
        'number_of_dependants': input_dict.get('Number of Dependants', 0),
        'income_lakhs': input_dict.get('Income in Lakhs', 0),
        'insurance_plan': {'Bronze': 1, 'Silver': 2, 'Gold': 3}.get(input_dict.get('Insurance Plan'), 0),
        'normalized_risk_score': calculate_normalized_risk(input_dict.get('Medical History', '')),
        
        # 🚨 NEW: Pull the actual slider value instead of hardcoding 0
        'genetical_risk': input_dict.get('Genetical Risk', 0), 
        
        'gender_Male': 1 if input_dict.get('Gender') == 'Male' else 0,
        'region_Northwest': 1 if input_dict.get('Region') == 'Northwest' else 0,
        'region_Southeast': 1 if input_dict.get('Region') == 'Southeast' else 0,
        'region_Southwest': 1 if input_dict.get('Region') == 'Southwest' else 0,
        'marital_status_Unmarried': 1 if input_dict.get('Marital Status') == 'Unmarried' else 0,
        'bmi_category_Obesity': 1 if input_dict.get('BMI Category') == 'Obesity' else 0,
        'bmi_category_Overweight': 1 if input_dict.get('BMI Category') == 'Overweight' else 0,
        'bmi_category_Underweight': 1 if input_dict.get('BMI Category') == 'Underweight' else 0,
        'smoking_status_Occasional': 1 if input_dict.get('Smoking Status') == 'Occasional' else 0,
        'smoking_status_Regular': 1 if input_dict.get('Smoking Status') == 'Regular' else 0,
        'employment_status_Salaried': 1 if input_dict.get('Employment Status') == 'Salaried' else 0,
        'employment_status_Self-Employed': 1 if input_dict.get('Employment Status') == 'Self-Employed' else 0,
    }
    return pd.DataFrame([features])

def handle_scaling(age, df):
    scaler_object = scaler_young_data if age <= 25 else scaler_rest_data
    cols_to_scale = scaler_object['cols_to_scale']
    scaler = scaler_object['scaler']
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    return df

# ==========================================
# 3. MAIN PREDICTION FUNCTION
# ==========================================
def predict(input_dict):
    input_df = preprocess_input(input_dict)
    input_df = handle_scaling(input_dict['Age'], input_df)

    if input_dict['Age'] <= 25:
        prediction = model_young.predict(input_df)
    else:
        prediction = model_rest.predict(input_df)

    return int(prediction[0])