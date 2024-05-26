#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import cufflinks as cf
from streamlit_extras.metric_cards import style_metric_cards 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib

#######################
# Page configuration
st.set_page_config(
    page_title="Attrition Analytical Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################
# PREPARE DATA
# Load the dataset
data_url = "https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/main/employee/employee_data.csv"
df = pd.read_csv(data_url)

# Prepare Functions
# Cleaning function
def clean_data(df):
    df = df.dropna(subset=['Attrition'])
    df['Attrition'] = df['Attrition'].astype(int)
    df = df.drop(columns=['EmployeeId']).copy() 
    single_unique_cols_before = [col for col in df.columns if df[col].nunique() == 1]
    df = df.drop(columns=single_unique_cols_before).copy()
    df_cleaned = df.drop(columns=['DailyRate', 'HourlyRate', 'MonthlyRate'])
    return df_cleaned

# Grouping function
def group_data(df_cleaned):
    bins_age = [18, 23, 28, 33, 38, 43, 48, 53, 58, 61]
    labels_age = ['18-22', '23-27', '28-32', '33-37', '38-42', '43-47', '48-52', '53-57', '58-60']
    df_cleaned['Age_Group'] = pd.cut(df_cleaned['Age'], bins=bins_age, labels=labels_age, right=False)
    

    bin_distance = [0, 9, 19, 31]
    labels_distance = ['0-9', '10-19', '20-30']
    df_cleaned['DistanceFromHome_Group'] = pd.cut(df_cleaned['DistanceFromHome'], bins=bin_distance, labels=labels_distance)

    bin_income = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 15000, 20000]
    labels_income = ['1K-1,999', '2K-2,999', '3K-3,999', '4K-4,999', '5K-5,999', '6K-6,999', '7K-7,999', '8K-8,999', '9K-9,999', '10K-14,999', '15K-19,999']
    df_cleaned['MonthlyIncome_Group'] = pd.cut(df_cleaned['MonthlyIncome'], bins=bin_income, labels=labels_income)

    bins_numcomp = [-1, 1, 3, 5, 7, 9, 10]
    labels_numcomp = ['0', '1', '2-3', '4-5', '6-7', '8-9']
    df_cleaned['NumCompaniesWorked_Group'] = pd.cut(df_cleaned['NumCompaniesWorked'], bins=bins_numcomp, labels=labels_numcomp, right=False)

    bins_salaryhike = [10, 15, 20, 26]
    labels_salaryhike = ['11-15', '16-20', '21-25']
    df_cleaned['PercentSalaryHike_Group'] = pd.cut(df_cleaned['PercentSalaryHike'], bins=bins_salaryhike, labels=labels_salaryhike, right=False)

    bins_totalworking = [0, 2, 4, 6, 8, 10, 11, 16, 21, 26, 31, 41]
    labels_totalworking = ['0-1', '2-3', '4-5', '6-7', '8-9', '10', '11-15', '16-20', '21-25', '26-30', '30-40']
    df_cleaned['TotalWorkingYears_Group'] = pd.cut(df_cleaned['TotalWorkingYears'], bins=bins_totalworking, labels=labels_totalworking, right=False)

    bins_years_in_current_role = [-1, 2, 5, 8, 11, 14, 19]
    labels_years_in_current_role = ['0-2', '3-5', '6-8', '9-11', '12-14', '15-18']
    df_cleaned['YearsInCurrentRole_Group'] = pd.cut(df_cleaned['YearsInCurrentRole'], bins=bins_years_in_current_role, labels=labels_years_in_current_role, right=False)

    bins_years_since_last_promotion = [-1, 2, 5, 8, 11, 16]
    labels_years_since_last_promotion = ['0-2', '3-5', '6-8', '9-11', '12-15']
    df_cleaned['YearsSinceLastPromotion_Group'] = pd.cut(df_cleaned['YearsSinceLastPromotion'], bins=bins_years_since_last_promotion, labels=labels_years_since_last_promotion, right=False)

    bins_years_with_curr_manager = [-1, 2, 5, 8, 11, 14, 18]
    labels_years_with_curr_manager = ['0-2', '3-5', '6-8', '9-11', '12-14', '15-17']
    df_cleaned['YearsWithCurrManager_Group'] = pd.cut(df_cleaned['YearsWithCurrManager'], bins=bins_years_with_curr_manager, labels=labels_years_with_curr_manager, right=False)

    bins_yearsatcompany = [0, 2, 4, 7, 11, 16, 21, 31, 41]
    labels_yearsatcompany = ['0-1', '2-3', '4-6', '7-10', '11-15', '16-20', '21-30', '31-40']
    df_cleaned['YearsAtCompany_Group'] = pd.cut(df_cleaned['YearsAtCompany'], bins=bins_yearsatcompany, labels=labels_yearsatcompany, right=False)

    columns_to_drop = ['Age', 'DistanceFromHome', 'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
    df_grouped = df_cleaned.drop(columns=columns_to_drop)
    return df_grouped

# Encoding function
def encode_data(df_grouped):
    one_hot_encoded_cols = ['Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']
    df_encoded = pd.get_dummies(df_grouped, columns=one_hot_encoded_cols)
    
    # Initialize LabelEncoder
    label_encoder = LabelEncoder()
    label_encoded_cols = ['BusinessTravel', 'OverTime']
    business_travel_mapping = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}
    overtime_mapping = {'No': 0, 'Yes': 1}
    df_encoded['BusinessTravel_Encoded'] = df_encoded['BusinessTravel'].map(business_travel_mapping)
    df_encoded['OverTime_Encoded'] = df_encoded['OverTime'].map(overtime_mapping)
    df_encoded.drop(columns=['BusinessTravel','OverTime'],inplace=True)

    # Encoding category columns
    label_mappings = {}
    for column in df_encoded.select_dtypes(include='category'):
        df_encoded[column + '_Encoded'] = label_encoder.fit_transform(df_encoded[column])
        label_mappings[column] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        if column in df_encoded.columns:
            df_encoded.drop(column, axis=1, inplace=True)
    
    return df_encoded

# Data balancing function
def balance_data(df_encoded):
    X = df_encoded.drop('Attrition', axis=1)
    y = df_encoded['Attrition']
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    df_balanced = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name='Attrition')], axis=1)
    return df_balanced

# Function to extract highest and lowest percentages of attrition
def calculate_attrition_percentages(df_grouped):
    attrition_percentages_by_column = {}
    for column in df_grouped.columns:
        if column == 'Attrition':
            continue
        counts = df_grouped.groupby(column)['Attrition'].mean() * 100
        attrition_percentages_by_column[column] = counts
    return attrition_percentages_by_column

def extract_min_max(percentages):
    min_value = percentages.min()
    max_value = percentages.max()
    min_category = percentages.idxmin()
    max_category = percentages.idxmax()
    return min_category, min_value, max_category, max_value

# Get dataframes needed
df_cleaned = clean_data(df)
df_grouped = group_data(df_cleaned)
df_encoded = encode_data(df_grouped)
df_balanced = balance_data(df_encoded)

######################

# MODEL LOAD
rf_trained_model = joblib.load('trained_rf_model.joblib')

######################

# Navigation Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Attrition Dashboard", "About", "Attrition Analysis", "Attrition Predictor"])


#######################

# Render functions
# Justify text function
def render_justified_text(text):
    st.markdown(
        f"""
        <style>
        .justified-text {{
            text-align: justify;
        }}
        </style>
        <div class="justified-text">
        {text}
        </div>
        """, 
        unsafe_allow_html=True
    )
##############    
# 'About' page render function
def render_about_page():
    st.title('About Attrition Dashboard')
    render_justified_text("""
        This Attrition Analytics Dashboard helps the Human Resource Department of Jaya Jaya Maju identify factors contributing to employee attrition.
        The dashboard comprises four pages, accessible via the sidebar menu:
        """)
    st.write("""
            - :blue-background[**About:**] Presents the fundamental information about this analytical dashboard and the data. 
            - :blue-background[**Attrition Dashboard:**] Presents visualizations of general attrition cases for various categories.
            - :blue-background[**Attrition Analysis:**] Displays the correlation of each variable with attrition, identifying key contributing factors.
            - :blue-background[**Attrition Predictor:**] Predicts the likelihood of attrition based on user input.
        """)
    
    st.header('Dataset Information')
    render_justified_text(
        """
        The dataset contains various features about employees including their age, department, job role, and more.
        The dataset contains demographic details, work-related metrics, and attrition flags. 
        1058 complete records and 28 features were used for the analysis after excluding 7 features with unique values or redundant information: 'EmployeeID', 'EmployeeCount', 'Over18', 'StandardHours', 'DailyRate', 'HourlyRate', 'MonthlyRate'.
        All employees are over 18 years old and work standard hours.
        The target variable is `Attrition` which indicates whether an employee has left the company (1) or not (0).
        """
    )
    
    st.subheader('Raw Data')
    st.write(df.head())

    st.subheader("Preprocessing Data")
    st.write("""
    Numerical data were grouped to enhance visualization and understanding. The categorical text data were then encoded or transformed into numerical form for analysis and modeling stages.
    """)
    st.write("Grouped dataset")
    st.write(df_grouped.head())

    st.write("Encoded dataset")
    st.write(df_encoded.head())


    st.write("""
    **Grouping Includes:**
    - **Age:** Grouped into ['18-22', '23-27', '28-32', '33-37', '38-42', '43-47', '48-52', '53-57', '58-60']
    - **DistanceFromHome:** Grouped into ['0-9', '10-19', '20-30']
    - **MonthlyIncome:** Grouped into ['1K-1,999', '2K-2,999', '3K-3,999', '4K-4,999','5K-5,999', '6K-6,999', '7K-7,999', '8K-8,999', '9K-9,999', '10K-14,999', '15K-19,999']
    - **NumCompaniesWorked:** Grouped into ['0', '1', '2-3', '4-5', '6-7', '8-9']
    - **PercentSalaryHike:** Grouped into ['11-15', '16-20', '21-25']
    - **TotalWorkingYears:** Grouped into ['0-1', '2-3', '4-5', '6-7', '8-9', '10', '11-15', '16-20', '21-25', '26-30', '30-40']
    - **YearsInCurrentRole:** Grouped into ['0-2', '3-5', '6-8', '9-11', '12-14', '15-18']
    - **YearsSinceLastPromotion:** Grouped into ['0-2', '3-5', '6-8', '9-11', '12-15']
    - **YearsWithCurrManager:** Grouped into ['0-2', '3-5', '6-8', '9-11', '12-14', '15-17']
    - **YearsAtCompany:** Grouped into ['0-1', '2-3', '4-6', '7-10', '11-15', '16-20', '21-30', '31-40']

    **Number Transformations Include:**
    - **BusinessTravel:** 'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2
    - **Overtime:** 'No': 0, 'Yes': 1
    - **Attrition:** 'No': 0, 'Yes': 1
    - **Education:** 1-Below College, 2-College, 3-Bachelor, 4-Master, 5-Doctor
    - **EnvironmentSatisfaction:** 1-Low, 2-Medium, 3-High, 4-Very High
    - **JobInvolvement:** 1-Low, 2-Medium, 3-High, 4-Very High
    - **JobLevel:** 1 to 5
    - **JobSatisfaction:** 1-Low, 2-Medium, 3-High, 4-Very High
    - **PerformanceRating:** 1-Low, 2-Good, 3-Excellent, 4-Outstanding
    - **RelationshipSatisfaction:** 1-Low, 2-Medium, 3-High, 4-Very High
    - **StockOptionLevel:** 0 to 3
    - **TrainingTimesLastYear:** 0 to 6
    - **WorkLifeBalance:** 1-Low, 2-Good, 3-Excellent, 4-Outstanding

    **Categorical Transformations Include:**
    - **Department:** ['Human Resources', 'Research & Development', 'Sales']
    - **EducationField:** ['Other', 'Medical', 'Life Sciences', 'Marketing', 'Technical Degree', 'Human Resources']
    - **Gender:** ['Male', 'Female']
    - **JobRole:** ['Human Resources', 'Healthcare Representative', 'Research Scientist', 'Sales Executive', 'Manager', 'Laboratory Technician', 'Research Director', 'Manufacturing Director', 'Sales Representative']
    - **MaritalStatus:** ['Married', 'Single', 'Divorced']
    """)

################
# 'Dashboard' render function
def render_dashboard_page(df_grouped, df_cleaned):
      # Sidebar with Attrition filter
    st.title('📊 JJM Attrition Dashboard')

    # Filter
    selected_attrition = st.selectbox('Select Attrition', ['All Data', 'Yes', 'No'])

        # Filter data based on attrition
    if selected_attrition != 'All Data':
        attrition_value = 1 if selected_attrition == 'Yes' else 0
        df_chart = df_grouped[df_grouped['Attrition'] == attrition_value]
        df_metrics = df_cleaned[df_cleaned['Attrition'] == attrition_value]

    else:
        df_chart = df_grouped
        df_metrics = df_cleaned
    
    # Custom CSS to change text color and background transparency
    st.markdown("""
        <style>
        div[data-testid="stMetricLabel"] {
            color: white !important;
        }
        div[data-testid="stMetricValue"] {
            color: white !important;
        }
        .stMetric {
            background-color: rgba(255, 255, 255, 0) !important;
        }
        </style>
        """, unsafe_allow_html=True)

    ############ Charts and Visualizations###############
    ############# METRICS ################
    # OVERALL METRICS FUNCTIONS
    overall_median_income = df_cleaned['MonthlyIncome'].median()
    overall_mode_stock = df_cleaned['StockOptionLevel'].mode()[0]
    all_gender_female = df_cleaned[df_cleaned['Gender'] == 'Female']['Gender'].count()
    all_gender_male = df_cleaned[df_cleaned['Gender'] == 'Male']['Gender'].count()
    all_marital_status_single = df_cleaned[df_cleaned['MaritalStatus'] == 'Single']['MaritalStatus'].count()
    all_marital_status_married = df_cleaned[df_cleaned['MaritalStatus'] == 'Married']['MaritalStatus'].count()
    all_marital_status_divorced = df_cleaned[df_cleaned['MaritalStatus'] == 'Divorced']['MaritalStatus'].count()
    overall_median_job_satisfaction = df_cleaned['JobSatisfaction'].median()
    overall_median_wlb = df_cleaned['WorkLifeBalance'].median()
            

    # 1. Attrition rate
    def overall_attrition_metric(df_cleaned):
        overall_attrition_rate = df_cleaned['Attrition'].mean() * 100
        st.metric(label="Overall Attrition Rate", value=f"{overall_attrition_rate:.2f}%")

    # 2. Employee count
    def total_employee_metric(df_metrics):
        total_employees = len(df_metrics)
        st.metric(label="Employees Count", value=f"{total_employees}")

    # 3. Monthly Income
    def median_income():
        median_income = df_metrics['MonthlyIncome'].median()
        delta = median_income - overall_median_income
        st.metric(label="Median Monthly Income", value=f"${median_income:,.2f}", delta=delta)
        
    # 4. Job Satisfaction
    def median_job_satisfaction():
        med_job_satisfaction = df_metrics['JobSatisfaction'].median()
        delta = med_job_satisfaction - overall_median_job_satisfaction
        st.markdown('<h4 style="text-align:left; font-size:18px;">Median Job Satisfaction</h4>', unsafe_allow_html=True)
        st.metric(label="", value=f"{med_job_satisfaction:.2f}", delta=f"{delta:.2f}")
    
  
     # 5. Age
    overall_median_age = df_cleaned['Age'].median()
    # Metric function for median age based on attrition
    def median_age_metric():
        median_age = df_metrics['Age'].median()
        delta = median_age - overall_median_age
        st.metric(label="Age Median", value=f"{median_age:.2f}", delta=f"{delta:.2f}")
     
      # 6. Work-life Balance
    def median_wlb():
        median_worklifebalance = df_metrics['WorkLifeBalance'].median()
        delta = median_worklifebalance - overall_median_wlb
        st.metric(label="Median Work-Life Balance", value=f"{median_worklifebalance:.2f}", delta=f"{delta:.2f}")

        # 7. JobInvolvement
    def median_jobinvolvement():
        overall_median_jobinvolvement = df_cleaned['JobInvolvement'].median()
        median_jobinvolvement = df_metrics['JobInvolvement'].median()
        delta = median_jobinvolvement - overall_median_jobinvolvement
        st.metric(label="Median Job Involvement", value=f"{median_jobinvolvement:.2f}", delta=f"{delta:.2f}")
 
    overall_median_env = df_cleaned['EnvironmentSatisfaction'].median()
        # 8. EnvironmentSatisfaction
    def median_env():
        median_environment_satisfaction = df_metrics['EnvironmentSatisfaction'].median()
        delta = median_environment_satisfaction - overall_median_env
        st.metric(label="Median Environment Satisfaction", value=f"{median_environment_satisfaction:.2f}", delta=f"{delta:.2f}")

    # 9. Stock Option Level
    def mode_stock():
        mode_stocklevel = df_metrics['StockOptionLevel'].mode()[0]
        delta = mode_stocklevel - overall_mode_stock
        st.metric(label="Mode Stock Option Level", value=f"{mode_stocklevel}", delta=f"{delta}")

   
    # 11. Gender Female
    def percent_gender_female():
        percent_female = (df_metrics[df_metrics['Gender'] == 'Female']['Gender'].count() / all_gender_female) * 100
        st.metric(label="Percentage of Selected Attrition on Female", value=f"{percent_female:.2f}%")

    # 12. Gender Male
    def percent_gender_male():
        percent_male = (df_metrics[df_metrics['Gender'] == 'Male']['Gender'].count() / all_gender_male) * 100
        st.metric(label="Percentage of Selected Attrition on Male", value=f"{percent_male:.2f}%")

    # 13. Marital Status: Single
    def percent_single():
        single_count = df_metrics[df_metrics['MaritalStatus'] == 'Single'].shape[0]
        percent_single = (df_metrics[df_metrics['MaritalStatus'] == 'Single']['MaritalStatus'].count() / all_marital_status_single) * 100
        st.metric(label="Percentage of Selected Attrition on Single", value=f"{percent_single:.2f}%")

    # 14. Marital Status: Married
    def percent_married():
        married_count = df_metrics[df_metrics['MaritalStatus'] == 'Married'].shape[0]
        percent_married = (df_metrics[df_metrics['MaritalStatus'] == 'Married']['MaritalStatus'].count() / all_marital_status_married) * 100
        st.metric(label="Percentage of Selected Attrition on Married", value=f"{percent_married:.2f}%")

    # 15. Marital Status: Divorced
    def percent_divorced():
        divorced_count = df_metrics[df_metrics['MaritalStatus'] == 'Divorced'].shape[0]
        percent_divorced = (df_metrics[df_metrics['MaritalStatus'] == 'Divorced']['MaritalStatus'].count() / all_marital_status_divorced) * 100
        st.metric(label="Percentage of Selected Attrition on Divorced", value=f"{percent_divorced:.2f}%")
    
    # 16. Age
    overall_median_age = df_cleaned['Age'].median()
    # Metric function for median age based on attrition
    def median_age_metric(df_metrics, overall_median_age):
        median_age = df_metrics['Age'].median()
        delta = median_age - overall_median_age
        st.metric(label="Age Median", value=f"{median_age:.2f}", delta=f"{delta:.2f}")
    
    # 17. Income per JobRole
    overall_healthcare_median_income = df_cleaned[df_cleaned['JobRole'] == 'Healthcare Representative']['MonthlyIncome'].median()
    overall_researchscientist_median_income = df_cleaned[df_cleaned['JobRole'] == 'Research Scientist']['MonthlyIncome'].median()
    overall_salesexecutive_median_income = df_cleaned[df_cleaned['JobRole'] == 'Sales Executive']['MonthlyIncome'].median()
    overall_manager_median_income = df_cleaned[df_cleaned['JobRole'] == 'Manager']['MonthlyIncome'].median()

    ## Healthcare Representative
    def median_income_healthcare_representative():
        jobrole_median_income = df_metrics[df_metrics['JobRole'] == 'Healthcare Representative']['MonthlyIncome'].median()
        delta = jobrole_median_income - overall_healthcare_median_income
        st.markdown('<h4 style="text-align:left; font-size:18px;">Healthcare Representative</h4>', unsafe_allow_html=True)
        st.metric(label="Median Monthly Income", value=f"${jobrole_median_income:,.2f}", delta=delta)

    #  Research Scientist
    def median_income_research_scientist():
        jobrole_median_income = df_metrics[df_metrics['JobRole'] == 'Research Scientist']['MonthlyIncome'].median()
        delta = jobrole_median_income - overall_researchscientist_median_income
        st.markdown('<h4 style="text-align:left; font-size:18px;">Research Scientist</h4>', unsafe_allow_html=True)
        st.metric(label="Median Monthly Income", value=f"${jobrole_median_income:,.2f}", delta=delta)

    #  Sales Executive
    def median_income_sales_executive():
        jobrole_median_income = df_metrics[df_metrics['JobRole'] == 'Sales Executive']['MonthlyIncome'].median()
        delta = jobrole_median_income - overall_salesexecutive_median_income
        st.markdown('<h4 style="text-align:left; font-size:18px;">Sales Executive</h4>', unsafe_allow_html=True)
        st.metric(label="Median Monthly Income", value=f"${jobrole_median_income:,.2f}", delta=delta)

    # Manager
    def median_income_manager():
        jobrole_median_income = df_metrics[df_metrics['JobRole'] == 'Manager']['MonthlyIncome'].median()
        delta = jobrole_median_income - overall_manager_median_income
        st.markdown('<h4 style="text-align:left; font-size:18px;">Manager</h4>', unsafe_allow_html=True)
        st.metric(label="Median Monthly Income", value=f"${jobrole_median_income:,.2f}", delta=delta)
    

    overall_laboratory_median_income = df_cleaned[df_cleaned['JobRole'] == 'Laboratory Technician']['MonthlyIncome'].median()
    overall_researchdirector_median_income = df_cleaned[df_cleaned['JobRole'] == 'Research Director']['MonthlyIncome'].median()
    # Laboratory Technician
    def median_income_laboratory_technician():
        jobrole_median_income = df_metrics[df_metrics['JobRole'] == 'Laboratory Technician']['MonthlyIncome'].median()
        delta = jobrole_median_income - overall_laboratory_median_income
        st.markdown('<h4 style="text-align:left; font-size:18px;">Laboratory Technician</h4>', unsafe_allow_html=True)
        st.metric(label="Median Monthly Income", value=f"${jobrole_median_income:,.2f}", delta=delta)

    # Research Director
    def median_income_research_director():
        jobrole_median_income = df_metrics[df_metrics['JobRole'] == 'Research Director']['MonthlyIncome'].median()
        delta = jobrole_median_income - overall_researchdirector_median_income
        st.markdown('<h4 style="text-align:left; font-size:18px;">Research Director</h4>', unsafe_allow_html=True)
        st.metric(label="Median Monthly Income", value=f"${jobrole_median_income:,.2f}", delta=delta)
    
    overall_manufacturingdirector_median_income = df_cleaned[df_cleaned['JobRole'] == 'Manufacturing Director']['MonthlyIncome'].median()
    overall_hr_median_income = df_cleaned[df_cleaned['JobRole'] == 'Human Resources']['MonthlyIncome'].median()
    # Manufacturing Director
    def median_income_manufacturing_director():
        jobrole_median_income = df_metrics[df_metrics['JobRole'] == 'Manufacturing Director']['MonthlyIncome'].median()
        delta = jobrole_median_income - overall_manufacturingdirector_median_income
        st.markdown('<h4 style="text-align:left; font-size:18px;">Manufacturing Director</h4>', unsafe_allow_html=True)
        st.metric(label="Median Monthly Income", value=f"${jobrole_median_income:,.2f}", delta=delta)
    
    
    # Human Resources
    def median_income_human_resources():
        jobrole_median_income = df_metrics[df_metrics['JobRole'] == 'Human Resources']['MonthlyIncome'].median()
        delta = jobrole_median_income - overall_hr_median_income
        st.markdown('<h4 style="text-align:left; font-size:18px;">Human Resources</h4>', unsafe_allow_html=True)
        st.metric(label="Median Monthly Income", value=f"${jobrole_median_income:,.2f}", delta=delta)
    
    # Sales Representative
    overall_salesrepresentative_median_income = df_cleaned[df_cleaned['JobRole'] == 'Sales Representative']['MonthlyIncome'].median()
    def median_income_sales_representative():
        jobrole_median_income = df_metrics[df_metrics['JobRole'] == 'Sales Representative']['MonthlyIncome'].median()
        delta = jobrole_median_income - overall_salesrepresentative_median_income
        st.markdown('<h4 style="text-align:left; font-size:18px;">Sales Representative</h4>', unsafe_allow_html=True)
        st.metric(label="Median Monthly Income", value=f"${jobrole_median_income:,.2f}", delta=delta)


       # Apply consistent styling to metric cards
    ############ CHARTS #################
    # 1. Gender
    def gender_chart(df_chart):
        gender_cnt = df_chart['Gender'].value_counts().reset_index()
        gender_cnt.columns = ['Gender', 'Count']
        pie_fig_gender = px.pie(gender_cnt, names='Gender', values='Count', title='Attrition by Gender', hole=0.4)
        pie_fig_gender.update_layout(
            width=400,
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title={
                'text': 'Gender Proportion',
                'font': {
                    'color': 'white',
                    'size': 27
                },
                'x': 0.5,
                'xanchor': 'center'
            },
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.3,
                xanchor='center',
                x=0.5
            )
        )
        chart_gender = st.plotly_chart(pie_fig_gender)
        return chart_gender


    # 2. Jobrole
    def jobrole_chart(df_chart):
        jobrole_cnt = df_chart['JobRole'].value_counts().reset_index()
        jobrole_cnt.columns = ['JobRole', 'Count']
        pie_fig_jobrole = px.pie(jobrole_cnt, names='JobRole', values='Count', title='Attrition by JobRole', hole=0.4)
        pie_fig_jobrole.update_layout(
            width=400,
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title={
                'text': 'Attrition Proportion on JobRole',
                'font': {
                    'color': 'white',
                    'size': 27
                },
                'x': 0.5,
                'xanchor': 'center'
            },
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.3,
                xanchor='center',
                x=0.5
            )
        )
        chart_jobrole = st.plotly_chart(pie_fig_jobrole)
        return chart_jobrole

      
    # 3. Age distribusion
    def age_chart(df_chart):
        st.markdown('### Age Distribution')
        age_chart = alt.Chart(df_chart).mark_bar().encode(
        x=alt.X('Age_Group:N', title='Age Group'),
        y=alt.Y('count():Q', title='Count'),
        color=alt.Color('Attrition:N', title='Attrition')).properties(width=400, height=300)
        chart_age = st.altair_chart(age_chart, use_container_width=True)
        return chart_age


        
    # 4. JobRole 
    total_jobrole_level_counts = df_cleaned.groupby(['JobRole', 'JobLevel']).size().reset_index(name='Total')
    attrition_jobrole_level_counts = df_metrics.groupby(['JobRole', 'JobLevel']).size().reset_index(name='Attrition_Count')
    df_attrition_jobrolelevel = attrition_jobrole_level_counts.merge(total_jobrole_level_counts, on=['JobRole', 'JobLevel'])
    df_attrition_jobrolelevel['Attrition_Percentage'] = (df_attrition_jobrolelevel['Attrition_Count'] / df_attrition_jobrolelevel['Total']) * 100

    def jobrole_level_chart(df_attrition_jobrolelevel):
        st.markdown('### Attrition Percentage by Job Role and Job Level')
        jobrole_level = px.bar(df_attrition_jobrolelevel, x='JobRole', y='Attrition_Percentage', color='Attrition_Count', facet_col='JobLevel')
        st.plotly_chart(jobrole_level)

    
    # 5. Job satisfaction
    median_job_satisfaction = df_metrics.groupby(['JobRole', 'JobLevel'])['JobSatisfaction'].median().reset_index()

    def jobrole_satisfaction_chart(df_job_satisfaction):
        st.markdown('### Median Job Satisfaction by Job Role and Job Level')
        jobrole_level = px.bar(df_job_satisfaction, x='JobRole', y='JobSatisfaction', color='JobLevel', 
                            barmode='group')
        st.plotly_chart(jobrole_level)


    # 6. Job Involvement chart
    def jobinvolvement_chart(df_cleaned, df_metrics):
        total_counts = df_cleaned.groupby(['JobRole', 'JobInvolvement', 'JobLevel']).size().reset_index(name='Total')
        attrition_counts = df_metrics.groupby(['JobRole', 'JobInvolvement', 'JobLevel']).size().reset_index(name='Attrition_Count')
        df_combined = total_counts.merge(attrition_counts, on=['JobRole', 'JobInvolvement', 'JobLevel'], how='left')
        df_combined['Attrition_Count'] = df_combined['Attrition_Count'].fillna(0)
        df_combined['Attrition_Percentage'] = (df_combined['Attrition_Count'] / df_combined['Total']) * 100

        # Area Chart for Job Involvement and Attrition Percentage
        st.markdown('### Job Involvement by Job Role, Job Level, and Attrition Percentage')
        fig_job_involvement_area = px.area(
            df_combined, 
            x='JobInvolvement', 
            y='Attrition_Percentage', 
            color='JobLevel', 
            facet_col='JobRole', 
            facet_col_wrap=3,
            labels={'JobInvolvement': 'Job Involvement', 'Attrition_Percentage': 'Attrition Percentage', 'JobRole': 'Job Role'},
            hover_data={'Total': True, 'Attrition_Count': True}
        )

        # Update hover template to include more details
        fig_job_involvement_area.update_traces(hovertemplate=(
            'Job Involvement: %{x}<br>' +
            'Attrition Percentage: %{y:.2f}%<br>' +
            'Total Employees: %{customdata[0]}<br>' +
            'Attrition Count: %{customdata[1]}'
        ))

        # Add custom data for hover template
        fig_job_involvement_area.update_traces(customdata=df_combined[['Total', 'Attrition_Count']])

        st.plotly_chart(fig_job_involvement_area)


    # 7. Work-life Balance
    def work_life_balance_area_chart_faceted(df):
        # Ensure the JobRole and JobLevel columns are categories
        df['JobRole'] = df['JobRole'].astype('category')
        df['JobLevel'] = df['JobLevel'].astype('category')

        st.markdown('### Work-Life Balance by Job Role and Job Level')
        fig = px.area(df, x='JobRole', y='WorkLifeBalance', color='JobLevel',
                        facet_col='JobLevel', facet_col_wrap=3,
                        labels={'WorkLifeBalance': 'Work-Life Balance', 'JobRole': 'Job Role', 'JobLevel': 'Job Level'})

        st.plotly_chart(fig)

    
    # 8. Combined Satisfaction Bar Chart
    def combined_satisfaction_barchart(df_cleaned, df_metrics):
        total_counts = df_cleaned.groupby(['JobRole', 'EnvironmentSatisfaction', 'RelationshipSatisfaction']).size().reset_index(name='Total')
        attrition_counts = df_metrics.groupby(['JobRole', 'EnvironmentSatisfaction', 'RelationshipSatisfaction']).size().reset_index(name='Attrition_Count')
        df_combined = total_counts.merge(attrition_counts, on=['JobRole', 'EnvironmentSatisfaction', 'RelationshipSatisfaction'], how='left')
        df_combined['Attrition_Count'] = df_combined['Attrition_Count'].fillna(0)
        df_combined['Attrition_Percentage'] = (df_combined['Attrition_Count'] / df_combined['Total']) * 100

        # Area Chart for Environment Satisfaction and Relationship Satisfaction
        st.markdown('### Attrition Percentage by Environment Satisfaction and Relationship Satisfaction')
        fig_satisfaction_area = px.area(
            df_combined, 
            x='EnvironmentSatisfaction', 
            y='Attrition_Percentage', 
            color='RelationshipSatisfaction',
            facet_col='JobRole', 
            facet_col_wrap=3,
            labels={
                'EnvironmentSatisfaction': 'Environment Satisfaction', 
                'Attrition_Percentage': 'Attrition Percentage', 
                'JobRole': 'Job Role', 
                'RelationshipSatisfaction': 'Relationship Satisfaction'
            },
            hover_data={'Total': True, 'Attrition_Count': True}
        )

        # Update hover template to include more details
        fig_satisfaction_area.update_traces(hovertemplate=(
            'Environment Satisfaction: %{x}<br>' +
            'Attrition Percentage: %{y:.2f}%<br>' +
            'Total Employees: %{customdata[0]}<br>' +
            'Attrition Count: %{customdata[1]}'
        ))

        # Add custom data for hover template
        fig_satisfaction_area.update_traces(customdata=df_combined[['Total', 'Attrition_Count']])

        st.plotly_chart(fig_satisfaction_area)
  
    # 9. JobRole Tree Map
    def jobrole_treemap(df_cleaned, selected_attrition):
        st.markdown("### Monthly Income based on Jobrole and Gender")
        fig = px.treemap(filtered_data, 
                        path=['Gender', 'Department', 'JobRole'],
                        color_continuous_scale='RdBu',
                        color='MonthlyIncome',  
                        hover_data={'Gender': True, 'Department': True, 'JobRole': True, 'MonthlyIncome': True}
                        )

        st.plotly_chart(fig)
    
    # 10. WLB treemap
    def wlb_treemap(df_grouped, selected_attrition):
        if selected_attrition == 'All Data':
            df_filtered = df_grouped.copy()  
        else:
            attrition_value = 1 if selected_attrition == 'Yes' else 0  
            df_filtered = df_grouped[df_grouped['Attrition'] == attrition_value]

        # print("Filtered DataFrame:")
        # print(df_filtered.head()) 

        # Create the treemap
        st.markdown('### Employee Distribution by Gender, OverTime, Marital Status, WorkLifeBalance')
        fig = px.treemap(df_filtered, 
                        path=['Gender', 'OverTime', 'MaritalStatus', 'WorkLifeBalance'], 
                        color='WorkLifeBalance', 
                        color_continuous_scale='blues')
        fig.update_layout(margin=dict(t=50, l=0, r=0, b=0))  # Adjust margin for better layout

        print("Figure Object:")
        print(fig) 
        st.plotly_chart(fig)


    # 11. Job involvement on jobrole
    def jobrole_inv(df_metrics):
        df_job_involvement_counts = df_metrics.groupby(['JobRole', 'JobInvolvement', 'Attrition']).size().reset_index(name='Count')

        # Area Chart for Job Involvement and Attrition
        st.markdown('### Job Involvement by Job Role and Attrition')
        fig_job_involvement_area = px.area(df_job_involvement_counts, x='JobInvolvement', y='Count', color='Attrition', line_group='JobRole',
                                        facet_col='JobRole', facet_col_wrap=3,
                                        labels={'JobInvolvement': 'Job Involvement', 'Count': 'Count of Employees', 'JobRole': 'Job Role'})
        fig_job_involvement_area.update_traces(mode='lines+markers')
        st.plotly_chart(fig_job_involvement_area)


    # 12. Gender and distance
    def attrition_percentage_by_gender_and_distance(df_grouped, df_chart, selected_attrition):
        total_counts = df_grouped.groupby(['Gender', 'DistanceFromHome_Group']).size().reset_index(name='Total')

        if selected_attrition != 'All Data':
            attrition_value = 1 if selected_attrition == 'Yes' else 0
            attrition_counts = df_chart.groupby(['Gender', 'DistanceFromHome_Group']).size().reset_index(name='Attrition_Count')
        else:
            attrition_counts = df_chart[df_chart['Attrition'] == 1].groupby(['Gender', 'DistanceFromHome_Group']).size().reset_index(name='Attrition_Count')

        # Merge the two DataFrames to get the total and attrition count together
        df_combined = total_counts.merge(attrition_counts, on=['Gender', 'DistanceFromHome_Group'], how='left')

        # Calculate the attrition percentage
        df_combined['Attrition_Percentage'] = (df_combined['Attrition_Count'] / df_combined['Total']) * 100

        # Bar Chart for Attrition Percentage by Gender and DistanceFromHome_Group
        st.markdown('### Attrition Percentage by Gender and Distance From Home')
        fig_attrition_bar = px.bar(
            df_combined,
            x='DistanceFromHome_Group',
            y='Attrition_Percentage',
            color='Gender',
            barmode='group',
            labels={
                'DistanceFromHome_Group': 'Distance From Home',
                'Attrition_Percentage': 'Attrition Percentage',
                'Gender': 'Gender'
            },
            hover_data={'Total': True, 'Attrition_Count': True}
        )

        # Update hover template to include more details
        fig_attrition_bar.update_traces(hovertemplate=(
            'Distance From Home: %{x}<br>' +
            'Attrition Percentage: %{y:.2f}%<br>' +
            'Total Employees: %{customdata[0]}<br>' +
            'Attrition Count: %{customdata[1]}'
        ))

        # Add custom data for hover template
        fig_attrition_bar.update_traces(customdata=df_combined[['Total', 'Attrition_Count']])

        st.plotly_chart(fig_attrition_bar)
    
    # Training treeemap
    def training_treemap(df_metrics):
        fig = px.treemap(df_metrics, 
                        path=['TrainingTimesLastYear', 'JobSatisfaction', 'Department'], 
                        values='JobSatisfaction',  
                        color='JobSatisfaction',  
                        color_continuous_scale='Blues',
                        title='Employee Distribution by Training Times, Job Satisfaction, and Department',
                        labels={'JobSatisfaction': 'Job Satisfaction', 'TrainingTimesLastYear': 'Training Times Last Year'})
        fig.update_layout(margin=dict(t=50, l=0, r=0, b=0), coloraxis_colorbar=dict(title='Job Satisfaction'))
        st.plotly_chart(fig)

#########################

# Streamlit application

#########################
    # Display metrics
    st.markdown("<h2 style='text-align: center;'>Selected Attrition on Demographic Features</h2>", unsafe_allow_html=True)
    if selected_attrition == 'All Data':
        filtered_data = df_cleaned  
        title = 'Selected Attrition: All Data'
    else:
        attrition_value = 1 if selected_attrition == 'Yes' else 0
        filtered_data = df_cleaned[df_cleaned['Attrition'] == attrition_value]
        title = f'Attrition: {"Yes" if attrition_value == 1 else "No"}'
    st.markdown(f"<h4 style='text-align: center;'>{title}</h4>", unsafe_allow_html=True)

    st.write("")
    
    # Set the demographic columns
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        overall_attrition_metric(df_cleaned)
    with col2:
        total_employee_metric(df_metrics)
    with col3:
        percent_gender_male()        
    with col4:
        percent_gender_female()
      
        
    col6, col7, col8, col5 = st.columns(4) 
    with col6:
        percent_single()
    with col7:
        percent_married()
    with col8:
        percent_divorced()
    with col5:
        median_age_metric(df_metrics, overall_median_age)
    
    
    cola, colb, colc1 = st.columns(3)
    with cola:
        gender_chart(df_metrics)
    with colb:
        age_chart(df_chart)
    with colc1:
        jobrole_chart(df_chart)

    colc, cold = st.columns(2)
    with colc:
        attrition_percentage_by_gender_and_distance(df_grouped, df_chart, selected_attrition)
    with cold:
        jobrole_level_chart(df_attrition_jobrolelevel) 

    st.write("")

    st.markdown("<h2 style='text-align: center;'>Features Value based on Selected Attrition</h2>", unsafe_allow_html=True)
    if selected_attrition == 'All Data':
        filtered_data = df_cleaned  
        title = 'Selected Attrition: All Data'
    else:
        attrition_value = 1 if selected_attrition == 'Yes' else 0
        filtered_data = df_cleaned[df_cleaned['Attrition'] == attrition_value]
        title = f'Attrition: {"Yes" if attrition_value == 1 else "No"}'
    st.markdown(f"<h4 style='text-align: center;'>{title}</h4>", unsafe_allow_html=True)

    st.write("")
    col9, col10, col43, col44, col45 = st.columns(5)
    with col9:
        median_income()
    with col10:        
        mode_stock()
    with col43:
        median_env()    
    with col44:
        median_wlb()
    with col45:
        median_jobinvolvement()
    st.write("")    
        
    st.write("")   
    cole, colf = st.columns(2)
    with cole:
        training_treemap(df_metrics)
    with colf:
        jobrole_treemap(df_cleaned, selected_attrition)
  
    st.markdown("<h2 style='text-align: center;'>Income per Jobrole</h2>", unsafe_allow_html=True)
    if selected_attrition == 'All Data':
        filtered_data = df_cleaned  
        title = 'Selected Attrition: All Data'
    else:
        attrition_value = 1 if selected_attrition == 'Yes' else 0
        filtered_data = df_cleaned[df_cleaned['Attrition'] == attrition_value]
        title = f'Attrition: {"Yes" if attrition_value == 1 else "No"}'
    st.markdown(f"<h4 style='text-align: center;'>{title}</h4>", unsafe_allow_html=True)

    st.write("")
    
    col11, col12, col13, col14, col15 = st.columns(5)
    with col11:
        median_income_research_director()
    with col12:
        median_income_manager()
    with col13:
        median_income_healthcare_representative()
    with col14:
        median_income_manufacturing_director()
    with col15:
        median_income_sales_executive()
        
    col16, col17, col18, col19 = st.columns(4)
    with col16:
        median_income_human_resources()
    with col17:
        median_income_laboratory_technician()
    with col18:
        median_income_research_scientist()
    with col19:
        median_income_sales_representative() 
        
            

    st.markdown("<h2 style='text-align: center;'>Attrition by Jobrole</h2>", unsafe_allow_html=True)
    if selected_attrition == 'All Data':
        filtered_data = df_cleaned  
        title = 'Selected Attrition: All Data'
    else:
        attrition_value = 1 if selected_attrition == 'Yes' else 0
        filtered_data = df_cleaned[df_cleaned['Attrition'] == attrition_value]
        title = f'Attrition: {"Yes" if attrition_value == 1 else "No"}'
    st.markdown(f"<h4 style='text-align: center;'>{title}</h4>", unsafe_allow_html=True)

    st.write("")
    col31, col32 = st.columns(2)
    with col31:
       jobrole_inv(df_metrics)
    with col32:
        work_life_balance_area_chart_faceted(df_metrics)


    col20, col21 = st.columns(2)
    with col20:
        wlb_treemap(df_grouped, selected_attrition)           

    with col21:
        jobrole_satisfaction_chart(median_job_satisfaction)
      
    
    col22, col23 = st.columns(2)
    with col22:
        jobinvolvement_chart(df_cleaned, df_metrics)
    with col23:
          combined_satisfaction_barchart(df_cleaned, df_metrics)

    
##############################################################

# Correlation render function
def render_analysis_page(df_balanced, df_grouped):
    st.title("Attrition Analysis")
    st.write("This section provides analyses to identify factors contributing to attrition.")
    st.write("Use this section to understand which factors have the strongest relationships with attrition.")

    # Correlation Metrix
    st.write("### Correlation Table")
    st.write("This table shows the correlations between data features.")
    # Calculate the correlation matrix
    correlation_matrix = df_balanced.corr()

    # Initialize an empty DataFrame to store highlighted correlations
    highlighted_matrix = correlation_matrix.copy()

    # Iterate through each variable
    for column in correlation_matrix.columns:
        # Find the highest and lowest correlation values for the variable
        max_corr = correlation_matrix[column].drop(column).max()
        min_corr = correlation_matrix[column].drop(column).min()

        # Highlight the highest correlation value
        highlighted_matrix.loc[highlighted_matrix[column] == max_corr, column] = f"<span style='color: red;'>{max_corr:.2f}</span>"
        # Highlight the lowest correlation value
        highlighted_matrix.loc[highlighted_matrix[column] == min_corr, column] = f"<span style='color: blue;'>{min_corr:.2f}</span>"

    # Display the highlighted correlation matrix with HTML
    correlation_html = highlighted_matrix.to_html(escape=False)
    st.write(f"<div style='overflow-x: auto; overflow-y: auto; max-width: 1000px; max-height: 500px;'>{correlation_html}</div>", unsafe_allow_html=True)


    # Calculate attrition percentages and extract min/max
    attrition_percentages_by_column = calculate_attrition_percentages(df_grouped)
    attrition_summary = {}

    for feature, percentages in attrition_percentages_by_column.items():
        min_category, min_value, max_category, max_value = extract_min_max(percentages)
        attrition_summary[feature] = {
            'Lowest attrition percentage': f"{min_value:.2f}% for category '{min_category}'",
            'Highest attrition percentage': f"{max_value:.2f}% for category '{max_category}'"
        }

    # Convert the summary dictionary to a DataFrame and transpose it
    summary_df = pd.DataFrame(attrition_summary).transpose()

    # Display the summary as a table
    st.write("### Highest and Lowest Attrition Percentages by Feature")
    st.table(summary_df)

    # Correlation with Attrition
    st.markdown("### Correlation with Attrition")
    correlations = df_balanced.corr()['Attrition'].drop('Attrition')
    st.write("**Positive Correlations**")
    st.write("""
    The table below shows the variables that has possitive correlation with attrition. 
    It means that the value contributing to attrition in linear way: when the value is higher, the more attrition risk it get.
    For example, we have 'Overtime' there. So, if the employee is working overtime (value=1) they will tend to increase the attrition rate, comparing to not overtime (value=0).
    The value indicating the correlation power.""")
    
    positive_corr = correlations[correlations > 0].sort_values(ascending=False)
    st.dataframe(positive_corr, width=800)

    st.write("**Negative Correlations**")
    st.write("""
    The table below shows the variables that has negative correlation with attrition. 
    It means that the value contributing to attrition in reverse way: when the value is higher, the less attrition risk it get.
    For instance, we have StockOptionLevel on the list. So, when the employee who has StockOptionLevel 4 tends to have less attrition risk compare to who has StockOptionLevel 0
    """)
    
    negative_corr = correlations[correlations < 0].sort_values(ascending=True)
    st.dataframe(negative_corr, width=800)

    st.markdown("## Conclusion")
    st.write("""
    Based on the analysis performed on employee attrition data, several key factors contributing to higher attrition rates have been identified:

    ##### Demographic Factors:
    - Male employees, single, aged between 28-32 years old, and those living more than 20 km from the office exhibit higher attrition rates.             
    - Male employees who frequently work overtime, hold positions as Sales Representatives or Laboratory Technicians have significantly higher attrition rates.

    ##### Income and Satisfaction:
    - Monthly income and median environment satisfaction are significantly different between employees who leave (attrition) and those who stay.

    ##### Job Roles with High Attrition:
    - The job roles with the highest attrition rates are Laboratory Technicians, Sales Representatives, Research Scientists, and Sales Executives. Among these, all except Sales Executives have lower monthly incomes.

    ###### Laboratory Technicians:
    - They experience the lowest work-life balance.
    - Higher attrition is observed at job level 1, with job involvement level 2, environment satisfaction level 1, and relationship satisfaction levels 2 and 3.
    - This suggests that Laboratory Technicians feel a heavy work burden at entry levels, low work-life balance, unsatisfactory environmental support, and higher income despite relatively good relationships with colleagues.

   ###### Research Scientists:
    - They face the lowest work-life balance and lower salaries.
    - Higher attrition is seen at job level 1 and job involvement level 1, indicating they may not feel their roles are important or valued.

    ###### Human Resources:
    - Job level 1 with involvement level 2 has the highest attrition.
    - No job involvement level is less than 1, indicating sufficient role participation.
    - Job level 3 employees have higher job involvement but lower job satisfaction.
    - Higher attrition is associated with environment satisfaction level 2 and relationship satisfaction levels 2 and 3.

    ###### Training and Job Satisfaction:
    - Sales Representatives and Research and Development roles with only one training session in the last year show the lowest job satisfaction.

    ###### Sales Executives:
    - They have higher job involvement at level 4 and higher salaries.
    - However, job level 4 also exhibits higher attrition.

    ## Business Action Recommendations
    To address these issues and reduce attrition rates, the following actions are recommended:

    ##### 1. Improve Work-Life Balance:
    - Implement flexible working hours and remote work options, especially for roles with low work-life balance like Laboratory Technicians and Research Scientists.

    ##### 2. Enhance Compensation and Benefits:
    - Review and potentially increase the salaries for roles with high attrition but lower incomes, such as Laboratory Technicians, Sales Representatives, and Research Scientists.
    - Offer additional benefits like transportation allowances for employees living far from the office.

    ##### 3. Increase Training and Development:
    - Provide more frequent and comprehensive training programs for all employees, particularly those in Sales Representative and Research and Development roles, to improve job satisfaction and skill development.

    ##### 4. Strengthen Employee Engagement and Support:
    - Develop programs to enhance job involvement and recognition, especially for entry-level positions and roles where employees feel undervalued.
    - Foster a supportive work environment by improving workplace conditions and resources.

    ##### 5. Focus on Career Development:
    - Implement clear career progression paths and opportunities for promotion, especially for job levels with higher attrition.
    - Offer mentorship and coaching programs to help employees at all levels feel more engaged and supported.

    ##### 6. Targeted Interventions for High Attrition Roles:
    - For Laboratory Technicians: Focus on improving work-life balance, environmental support, and job involvement.
    - For Research Scientists: Address salary issues and increase job involvement to help them feel more valued.
    - For Human Resources: Improve job satisfaction by addressing environmental and relationship factors.

    ##### 7. Monitor and Adapt:
    - Continuously monitor employee satisfaction and attrition rates.
    - Adapt strategies based on feedback and changing employee needs to ensure ongoing improvement in retention efforts.

    By implementing these recommendations, the organization can work towards reducing attrition rates, improving employee satisfaction, and fostering a more supportive and engaging work environment.
    """)

#########   
# 'Prediction' page render function
def render_prediction_page(df_grouped, rf_trained_model):
    st.title("Attrition Predictor")
    st.write("Predict employee attrition based on employee's profile.")
    st.write("Provide the employee details below and click 'Predict' to determine the likelihood of attrition.")


    input_data = {}
    columns = st.columns(3)  # Create 3 columns for layout

    # Distribute the select boxes across the three columns
    for idx, col in enumerate(df_grouped.columns):
        if col != 'Attrition':
            column_name = col.replace('_', ' ')
            unique_values = df_grouped[col].unique()
            selected_value = columns[idx % 3].selectbox(f"Select value for {column_name}:", unique_values)
            input_data[col] = selected_value

    if st.button("Predict"):
        # Encode the input data
        input_df = pd.DataFrame([input_data])
        input_encoded = encode_data(input_df)

        # Ensure the input data has the same columns as the training data
        missing_cols = set(rf_trained_model.feature_names_in_) - set(input_encoded.columns)
        for col in missing_cols:
            input_encoded[col] = 0

        input_encoded = input_encoded[rf_trained_model.feature_names_in_]

        # Use the trained model to predict
        y_pred_user = rf_trained_model.predict(input_encoded)
        attrition_probability = rf_trained_model.predict_proba(input_encoded)[0][1]

        # Display prediction results
        st.write(f"Predicted Attrition: {'Yes' if y_pred_user[0] else 'No'}")
        st.write(f"Attrition Probability: {attrition_probability * 100:.2f}%")

# Add the simple text footer at the bottom of the sidebar
st.sidebar.markdown(
    """
    <div style="height: 100px;"></div> <!-- Empty div with specified height -->
    <div style="text-align: center; font-size: small; color: grey;">
        &copy; 2024 Anggi Novitasari. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)
######################################################################

# STREAMLIT DEPLOYMENT
if page == "About":
    render_about_page()
elif page == "Attrition Dashboard":
    render_dashboard_page(df_grouped, df_cleaned)
elif page == "Attrition Analysis":
    render_analysis_page(df_balanced, df_grouped)
elif page == "Attrition Predictor":
    render_prediction_page(df_grouped, rf_trained_model)


######################################################################