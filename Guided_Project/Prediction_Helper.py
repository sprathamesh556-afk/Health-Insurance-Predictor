import pandas as pd
import joblib

# ==========================================
# 1. LOAD MODEL AND SCALER ARTIFACTS
# ==========================================
# Make sure these file paths match where your .joblib files are saved
model_young = joblib.load("artifacts/model_young.joblib")
model_rest = joblib.load("artifacts/model_rest.joblib")

scaler_young_data = joblib.load("artifacts/scaler_young.joblib")
scaler_rest_data = joblib.load("artifacts/scaler_rest.joblib")


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
    # The exact columns the models expect
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

    # Initialize all features to 0
    features = {col: 0 for col in expected_cols}

    # Set categorical features to 1 based on input
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

    # Handle numerical features
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

    # Create dummy column if scaler expects it
    df['income_level'] = None 
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    df.drop('income_level', axis='columns', inplace=True)

    return df


# ==========================================
# 3. MAIN PREDICTION FUNCTION (FIXED)
# ==========================================
def predict(input_dict):
    input_df = preprocess_input(input_dict)
    input_df = handle_scaling(input_dict['Age'], input_df)

    # Route based on age
    if input_dict['Age'] <= 25:
        # The young model was trained WITHOUT genetical_risk. Drop it here.
        if 'genetical_risk' in input_df.columns:
            input_df = input_df.drop(columns=['genetical_risk'])
        prediction = model_young.predict(input_df)
    else:
        # The rest model was trained WITH genetical_risk. Keep it.
        prediction = model_rest.predict(input_df)

    return int(prediction[0])