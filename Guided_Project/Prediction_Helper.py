import pandas as pd
import joblib
import os

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 1. LOAD MODEL AND SCALER ARTIFACTS
# ==========================================
# Using os.path.join makes it work perfectly on both Windows and Linux (Streamlit)
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
        
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }
    
    diseases = medical_history.lower().split(" & ")
    total_risk_score = sum(risk_scores.get(disease.strip(), 0) for disease in diseases)
    
    max_score = 14 
    min_score = 0  
    
    normalized_risk_score = (total_risk_score - min_score) / (max_score - min_score)
    return normalized_risk_score


def preprocess_input(input_dict):
    expected_cols = [
        'age', 'number_of_dependants', 'income_lakhs', 'insurance_plan', 
        'genetical_risk', 'normalized_risk_score', 'gender_Male', 
        'region_Northwest', 'region_Southeast', 'region_Southwest', 
        'marital_status_Unmarried', 'bmi_category_Obesity', 
        'bmi_category_Overweight', 'bmi_category_Underweight', 
        'smoking_status_Occasional', 'smoking_status_Regular', 
        'employment_status_Salaried', 'employment_status_Self-Employed'
    ]

    insurance_plan_encoding = {'Bronze': 1, 'Silver': 2, 'Gold': 3}
    features = {col: 0 for col in expected_cols}

    if input_dict.get('Gender') == 'Male':
        features['gender_Male'] = 1
        
    region = input_dict.get('Region')
    if region == 'Northwest':
        features['region_Northwest'] = 1
    elif region == 'Southeast':
        features['region_Southeast'] = 1
    elif region == 'Southwest':
        features['region_Southwest'] = 1
        
    if input_dict.get('Marital Status') == 'Unmarried':
        features['marital_status_Unmarried'] = 1
        
    bmi = input_dict.get('BMI Category')
    if bmi == 'Obesity':
        features['bmi_category_Obesity'] = 1
    elif bmi == 'Overweight':
        features['bmi_category_Overweight'] = 1
    elif bmi == 'Underweight':
        features['bmi_category_Underweight'] = 1
        
    smoking = input_dict.get('Smoking Status')
    if smoking == 'Regular':
        features['smoking_status_Regular'] = 1
    elif smoking == 'Occasional':
        features['smoking_status_Occasional'] = 1
        
    employment = input_dict.get('Employment Status')
    if employment == 'Salaried':
        features['employment_status_Salaried'] = 1
    elif employment == 'Self-Employed':
        features['employment_status_Self-Employed'] = 1

    features['age'] = input_dict.get('Age', 0)
    features['number_of_dependants'] = input_dict.get('Number of Dependants', 0)
    features['income_lakhs'] = input_dict.get('Income in Lakhs', 0) 
    features['insurance_plan'] = insurance_plan_encoding.get(input_dict.get('Insurance Plan'), 0)
    features['genetical_risk'] = input_dict.get('Genetical Risk', 0)
    features['normalized_risk_score'] = calculate_normalized_risk(input_dict.get('Medical History', ''))

    return pd.DataFrame([features])


def handle_scaling(age, df):
    if age <= 25:
        scaler_object = scaler_young_data
    else:
        scaler_object = scaler_rest_data

    cols_to_scale = scaler_object['cols_to_scale']
    scaler = scaler_object['scaler']

    df['income_level'] = None 
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    df.drop('income_level', axis='columns', inplace=True)

    return df


# ==========================================
# 3. MAIN PREDICTION FUNCTION
# ==========================================
def predict(input_dict):
    input_df = preprocess_input(input_dict)
    input_df = handle_scaling(input_dict['Age'], input_df)

    if input_dict['Age'] <= 25:
        if 'genetical_risk' in input_df.columns:
            input_df = input_df.drop(columns=['genetical_risk'])
        prediction = model_young.predict(input_df)
    else:
        prediction = model_rest.predict(input_df)

    # Get the raw prediction value
    final_prediction = int(prediction[0])
    
    # ML models can sometimes predict negative numbers for low-risk profiles. 
    # This ensures the app never shows a negative premium.
    if final_prediction < 0:
        final_prediction = 0

    return final_prediction