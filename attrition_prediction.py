# Import libraries
import pandas as pd
import numpy as np
import joblib

EXPECTED_COLUMNS = [
    'Age_Group_Encoded', 'BusinessTravel_Encoded', 'Department', 'DistanceFromHome_Group_Encoded', 'Education',
    'EducationField', 'EnvironmentSatisfaction', 'Gender',
    'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus',
    'MonthlyIncome_Group_Encoded', 'NumCompaniesWorked_Group_Encoded', 'OverTime_Encoded', 'PercentSalaryHike_Group_Encoded',
    'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel',
    'TotalWorkingYears_Group_Encoded', 'TrainingTimesLastYear', 'WorkLifeBalance',
    'YearsAtCompany_Group_Encoded', 'YearsInCurrentRole_Group_Encoded', 'YearsSinceLastPromotion_Group_Encoded',
    'YearsWithCurrManager_Group_Encoded'
]

# Define the unique values for certain columns
UNIQUE_VALUES = {
    'Department': ['Human Resources', 'Research & Development', 'Sales'],
    'EducationField': ['Other', 'Medical', 'Life Sciences', 'Marketing', 'Technical Degree', 'Human Resources'],
    'Gender': ['Male', 'Female'],
    'JobRole': ['Human Resources', 'Healthcare Representative', 'Research Scientist', 'Sales Executive', 'Manager',
                'Laboratory Technician', 'Research Director', 'Manufacturing Director', 'Sales Representative'],
    'MaritalStatus': ['Married', 'Single', 'Divorced'],
}

# Define the categories for specific columns
CATEGORIES = {
    'Age_Group_Encoded': {'18-22': 0, '23-27': 1, '28-32': 2, '33-37': 3, '38-42': 4, '43-47': 5, '48-52': 6, '53-57': 7, '58-60': 8},
    'BusinessTravel_Encoded': {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2},
    'DistanceFromHome_Group_Encoded': {'0-9': 0, '10-19': 1, '20-30': 2},
    'MonthlyIncome_Group_Encoded': {'1K-1,999': 0, '2K-2,999': 1, '3K-3,999': 2, '4K-4,999': 3, '5K-5,999': 4, '6K-6,999': 5, '7K-7,999': 6, '8K-8,999': 7, '9K-9,999': 8, '10K-14,999': 9, '15K-19,999': 10},
    'NumCompaniesWorked_Group_Encoded': {'0': 0, '1': 1, '2-3': 2, '4-5': 3, '6-7': 4, '8-9': 5},
    'OverTime_Encoded': {'No': 0, 'Yes': 1},
    'PercentSalaryHike_Group_Encoded': {'11-15': 0, '16-20': 1, '21-25': 2},
    'TotalWorkingYears_Group_Encoded': {'0-1': 0, '2-3': 1, '4-5': 2, '6-7': 3, '8-9': 4, '10': 5, '11-15': 6, '16-20': 7, '21-25': 8, '26-30': 9, '30-40': 10},
    'YearsInCurrentRole_Group_Encoded': {'0-2': 0, '3-5': 1, '6-8': 2, '9-11': 3, '12-14': 4, '15-18': 5},
    'YearsSinceLastPromotion_Group_Encoded': {'0-2': 0, '3-5': 1, '6-8': 2, '9-11': 3, '12-15': 4},
    'YearsWithCurrManager_Group_Encoded': {'0-2': 0, '3-5': 1, '6-8': 2, '9-11': 3, '12-14': 4, '15-17': 5},
    'YearsAtCompany_Group_Encoded': {'0-1': 0, '2-3': 1, '4-6': 2, '7-10': 3, '11-15': 4, '16-20': 5, '21-30': 6, '31-40': 7},
    'Education': {1: 'Below College', 2: 'College', 3: 'Bachelor', 4: 'Master', 5: 'Doctor'},
    'EnvironmentSatisfaction': {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'},
    'JobLevel': {1: 'Level 1', 2: 'Level 2', 3: 'Level 3', 4: 'Level 4', 5: 'Level 5'},
    'JobInvolvement': {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'},
    'JobSatisfaction': {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'},
    'RelationshipSatisfaction': {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'},
    'WorkLifeBalance': {1: 'Low', 2: 'Good', 3: 'Excellent', 4: 'Outstanding'},
    'StockOptionLevel': {0: 'Level 0', 1: 'Level 1', 2: 'Level 2', 3: 'Level 3'},
    'PerformanceRating': {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}
}

# Function to prompt user for input
def get_user_input():
    user_data = {}
    for column in EXPECTED_COLUMNS:
        if column in UNIQUE_VALUES:
            valid_values = UNIQUE_VALUES[column]
            prompt = f"Enter value for {column} (choose one from {valid_values}): "
            user_input = input(prompt)
            user_data[column] = user_input
        elif column in CATEGORIES:
            category_values = CATEGORIES[column]
            column_name = f"{column}_Group_Encoded"
            prompt = f"Enter value for {column} ({', '.join([f'{v}->{k}' for k, v in category_values.items()])}): "
            user_input = input(prompt)
            user_data[column] = category_values[user_input] if user_input in category_values else None
        else:
            prompt = f"Enter value for {column}: "
            user_input = input(prompt)
            user_data[column] = user_input
    return user_data

# Encoding function
def encode_data(user_data):
    one_hot_encoded_cols = ['Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']
    df = pd.DataFrame([user_data])
    df_encoded = pd.get_dummies(df, columns=one_hot_encoded_cols)
    return df_encoded

# Collect user input
user_data = get_user_input()

# Get dataframes needed
df_encoded = encode_data(user_data)

# MODEL LOAD
rf_trained_model = joblib.load('trained_rf_model.joblib')

# Ensure the input data has the same columns as the training data
input_encoded = df_encoded.reindex(columns=rf_trained_model.feature_names_in_, fill_value=0)

# Use the trained model to predict
y_pred_user = rf_trained_model.predict(input_encoded)
attrition_probability = rf_trained_model.predict_proba(input_encoded)[0][1]

# Display prediction results
print(f"Predicted Attrition: {'Yes' if y_pred_user[0] else 'No'}")
print(f"Attrition Probability: {attrition_probability * 100:.2f}%")