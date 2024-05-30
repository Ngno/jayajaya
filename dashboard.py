#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
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
    st.markdown("<h2 style='text-align: center;'>Selected Attrition on Demographic Features</h2>", unsafe_allow_html=True)

    selected_attrition = st.selectbox('Select Attrition (Demographic)', ['All Data', 'Yes', 'No'])

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
    def median_income(df_metrics):
        median_income = df_metrics['MonthlyIncome'].median()
        delta = median_income - overall_median_income
        st.metric(label="Median Monthly Income", value=f"${median_income:,.2f}", delta=delta)
        
     # 4. Age
    overall_median_age = df_cleaned['Age'].median()
    # Metric function for median age based on attrition
    def median_age_metric(df_metrics):
        median_age = df_metrics['Age'].median()
        delta = median_age - overall_median_age
        st.metric(label="Age Median", value=f"{median_age:.2f}", delta=f"{delta:.2f}")
     
    # 5. Work-life Balance
    def median_wlb(df_metrics):
        median_worklifebalance = df_metrics['WorkLifeBalance'].median()
        delta = median_worklifebalance - overall_median_wlb
        st.metric(label="Median Work-Life Balance", value=f"{median_worklifebalance:.2f}", delta=f"{delta:.2f}")

    # 6. JobInvolvement
    def median_jobinvolvement(df_metrics):
        overall_median_jobinvolvement = df_cleaned['JobInvolvement'].median()
        median_jobinvolvement = df_metrics['JobInvolvement'].median()
        delta = median_jobinvolvement - overall_median_jobinvolvement
        st.metric(label="Median Job Involvement", value=f"{median_jobinvolvement:.2f}", delta=f"{delta:.2f}")
 
    overall_median_env = df_cleaned['EnvironmentSatisfaction'].median()
    # 7. EnvironmentSatisfaction
    def median_env(df_metrics):
        median_environment_satisfaction = df_metrics['EnvironmentSatisfaction'].median()
        delta = median_environment_satisfaction - overall_median_env
        st.metric(label="Median Environment Satisfaction", value=f"{median_environment_satisfaction:.2f}", delta=f"{delta:.2f}")

    # 8. Stock Option Level
    def mode_stock(df_metrics):
        mode_stocklevel = df_metrics['StockOptionLevel'].mode()[0]
        delta = mode_stocklevel - overall_mode_stock
        st.metric(label="Mode Stock Option Level", value=f"{mode_stocklevel}", delta=f"{delta}")

    # 9. Gender Female
    def percent_gender_female(df_metrics):
        percent_female = (df_metrics[df_metrics['Gender'] == 'Female']['Gender'].count() / all_gender_female) * 100
        st.metric(label="Selected Attrition on Female", value=f"{percent_female:.2f}%")

    # 12. Gender Male
    def percent_gender_male(df_metrics):
        percent_male = (df_metrics[df_metrics['Gender'] == 'Male']['Gender'].count() / all_gender_male) * 100
        st.metric(label="Selected Attrition on Male", value=f"{percent_male:.2f}%")

    # 13. Marital Status: Single   
    def percent_single(df_metrics):
        single_count = df_metrics[df_metrics['MaritalStatus'] == 'Single'].shape[0]
        percent_single = (df_metrics[df_metrics['MaritalStatus'] == 'Single']['MaritalStatus'].count() / all_marital_status_single) * 100
        st.metric(label="Selected Attrition on Single", value=f"{percent_single:.2f}%")

    # 14. Marital Status: Married
    def percent_married(df_metrics):
        married_count = df_metrics[df_metrics['MaritalStatus'] == 'Married'].shape[0]
        percent_married = (df_metrics[df_metrics['MaritalStatus'] == 'Married']['MaritalStatus'].count() / all_marital_status_married) * 100
        st.metric(label="Selected Attrition on Married", value=f"{percent_married:.2f}%")

    # 15. Marital Status: Divorced
    def percent_divorced(df_metrics):
        divorced_count = df_metrics[df_metrics['MaritalStatus'] == 'Divorced'].shape[0]
        percent_divorced = (df_metrics[df_metrics['MaritalStatus'] == 'Divorced']['MaritalStatus'].count() / all_marital_status_divorced) * 100
        st.metric(label="Selected Attrition on Divorced", value=f"{percent_divorced:.2f}%")

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
    def median_income_healthcare_representative(df_metrics):
        jobrole_median_income = df_metrics[df_metrics['JobRole'] == 'Healthcare Representative']['MonthlyIncome'].median()
        delta = jobrole_median_income - overall_healthcare_median_income
        st.markdown('<h4 style="text-align:left; font-size:18px;">Healthcare Representative</h4>', unsafe_allow_html=True)
        st.metric(label="Median Monthly Income", value=f"${jobrole_median_income:,.2f}", delta=delta)

    #  Research Scientist
    def median_income_research_scientist(df_metrics):
        jobrole_median_income = df_metrics[df_metrics['JobRole'] == 'Research Scientist']['MonthlyIncome'].median()
        delta = jobrole_median_income - overall_researchscientist_median_income
        st.markdown('<h4 style="text-align:left; font-size:18px;">Research Scientist</h4>', unsafe_allow_html=True)
        st.metric(label="Median Monthly Income", value=f"${jobrole_median_income:,.2f}", delta=delta)

    #  Sales Executive
    def median_income_sales_executive(df_metrics):
        jobrole_median_income = df_metrics[df_metrics['JobRole'] == 'Sales Executive']['MonthlyIncome'].median()
        delta = jobrole_median_income - overall_salesexecutive_median_income
        st.markdown('<h4 style="text-align:left; font-size:18px;">Sales Executive</h4>', unsafe_allow_html=True)
        st.metric(label="Median Monthly Income", value=f"${jobrole_median_income:,.2f}", delta=delta)

    # Manager
    def median_income_manager(df_metrics):
        jobrole_median_income = df_metrics[df_metrics['JobRole'] == 'Manager']['MonthlyIncome'].median()
        delta = jobrole_median_income - overall_manager_median_income
        st.markdown('<h4 style="text-align:left; font-size:18px;">Manager</h4>', unsafe_allow_html=True)
        st.metric(label="Median Monthly Income", value=f"${jobrole_median_income:,.2f}", delta=delta)
    

    overall_laboratory_median_income = df_cleaned[df_cleaned['JobRole'] == 'Laboratory Technician']['MonthlyIncome'].median()
    overall_researchdirector_median_income = df_cleaned[df_cleaned['JobRole'] == 'Research Director']['MonthlyIncome'].median()
    # Laboratory Technician
    def median_income_laboratory_technician(df_metrics):
        jobrole_median_income = df_metrics[df_metrics['JobRole'] == 'Laboratory Technician']['MonthlyIncome'].median()
        delta = jobrole_median_income - overall_laboratory_median_income
        st.markdown('<h4 style="text-align:left; font-size:18px;">Laboratory Technician</h4>', unsafe_allow_html=True)
        st.metric(label="Median Monthly Income", value=f"${jobrole_median_income:,.2f}", delta=delta)

    # Research Director
    def median_income_research_director(df_metrics):
        jobrole_median_income = df_metrics[df_metrics['JobRole'] == 'Research Director']['MonthlyIncome'].median()
        delta = jobrole_median_income - overall_researchdirector_median_income
        st.markdown('<h4 style="text-align:left; font-size:18px;">Research Director</h4>', unsafe_allow_html=True)
        st.metric(label="Median Monthly Income", value=f"${jobrole_median_income:,.2f}", delta=delta)
    
    overall_manufacturingdirector_median_income = df_cleaned[df_cleaned['JobRole'] == 'Manufacturing Director']['MonthlyIncome'].median()
    overall_hr_median_income = df_cleaned[df_cleaned['JobRole'] == 'Human Resources']['MonthlyIncome'].median()
    # Manufacturing Director
    def median_income_manufacturing_director(df_metrics):
        jobrole_median_income = df_metrics[df_metrics['JobRole'] == 'Manufacturing Director']['MonthlyIncome'].median()
        delta = jobrole_median_income - overall_manufacturingdirector_median_income
        st.markdown('<h4 style="text-align:left; font-size:18px;">Manufacturing Director</h4>', unsafe_allow_html=True)
        st.metric(label="Median Monthly Income", value=f"${jobrole_median_income:,.2f}", delta=delta)
    
    
    # Human Resources
    def median_income_human_resources(df_metrics):
        jobrole_median_income = df_metrics[df_metrics['JobRole'] == 'Human Resources']['MonthlyIncome'].median()
        delta = jobrole_median_income - overall_hr_median_income
        st.markdown('<h4 style="text-align:left; font-size:18px;">Human Resources</h4>', unsafe_allow_html=True)
        st.metric(label="Median Monthly Income", value=f"${jobrole_median_income:,.2f}", delta=delta)
    
    # Sales Representative
    overall_salesrepresentative_median_income = df_cleaned[df_cleaned['JobRole'] == 'Sales Representative']['MonthlyIncome'].median()
    def median_income_sales_representative(df_metrics):
        jobrole_median_income = df_metrics[df_metrics['JobRole'] == 'Sales Representative']['MonthlyIncome'].median()
        delta = jobrole_median_income - overall_salesrepresentative_median_income
        st.markdown('<h4 style="text-align:left; font-size:18px;">Sales Representative</h4>', unsafe_allow_html=True)
        st.metric(label="Median Monthly Income", value=f"${jobrole_median_income:,.2f}", delta=delta)

    # 18. Overtime Yes
    def percent_overtime_yes(df_metrics):
        df_overtime = df_metrics[df_metrics['OverTime'] == 'Yes']
        percent_overtime = (df_overtime.shape[0] / df_metrics.shape[0]) * 100
        st.metric(label="Percentage of Employees with Overtime", value=f"{percent_overtime:.2f}%")

    
    # 19. Jobrole_monthly income metrics
    def jobrole_monthly_income_metrics(df_metrics, df_cleaned, job_role):
        # Convert JobLevel to string
        df_metrics['JobLevel'] = df_metrics['JobLevel'].astype(str)
        df_cleaned['JobLevel'] = df_cleaned['JobLevel'].astype(str)

        # Calculate the overall mean monthly income per job role per job level
        overall_mean_income_per_level = df_cleaned.groupby(['JobRole', 'JobLevel'])['MonthlyIncome'].mean().reset_index()
        overall_mean_income_per_level.set_index(['JobRole', 'JobLevel'], inplace=True)
        
        # Filter the DataFrame for the specific job role
        df_filtered = df_metrics[df_metrics['JobRole'] == job_role]
        
        # Iterate through each job level within the specified job role
        job_levels = df_filtered['JobLevel'].unique()
        for level in job_levels:
            # Filter the DataFrame for the specific job level
            df_level = df_filtered[df_filtered['JobLevel'] == level]

            # Calculate the mean monthly income for the specific job level in the filtered data
            mean_income = df_level['MonthlyIncome'].mean()

            # Check if the fixed mean income for the job role and job level exists
            if (job_role, level) in overall_mean_income_per_level.index:
                fixed_mean_income = overall_mean_income_per_level.loc[(job_role, level), 'MonthlyIncome']

                # Calculate the delta as the difference between the current mean income and the fixed overall mean income
                delta = mean_income - fixed_mean_income

                # Display the metric
                st.metric(label=f"Average Monthly Income for {job_role} (Level {level})",
                        value=f"${mean_income:,.2f}",
                        delta=delta)
            else:
                # Display a warning if no data is available for the specific job role and job level
                st.warning(f"No data available for {job_role} (Level {level}) in the overall dataset.")


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
                'xanchor': 'center',
                'y': 0.96,  
                'yanchor': 'top'
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
                'xanchor': 'center',
                'y': 0.98,  
                'yanchor': 'top'  
            },
            legend=dict(
                orientation='v',  
                yanchor='auto',
                y=1,  
                xanchor='left',
                x=-1  
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
        color=alt.Color('Attrition:N', title='Attrition')).properties(width=400, height=350)
        chart_age = st.altair_chart(age_chart, use_container_width=True)
        return chart_age

        
    # 4. JobRole 
    total_jobrole_level_counts = df_cleaned.groupby(['JobRole', 'JobLevel']).size().reset_index(name='Total')
    attrition_jobrole_level_counts = df_metrics.groupby(['JobRole', 'JobLevel']).size().reset_index(name='Attrition_Count')
    df_attrition_jobrolelevel = attrition_jobrole_level_counts.merge(total_jobrole_level_counts, on=['JobRole', 'JobLevel'])
    df_attrition_jobrolelevel['Attrition_Percentage'] = (df_attrition_jobrolelevel['Attrition_Count'] / df_attrition_jobrolelevel['Total']) * 100
   
  
    # 5. JobRole Tree Map
    def jobrole_treemap(df_cleaned):
        st.markdown("#### Monthly Income based on Jobrole and Gender")
        fig = px.treemap(df_cleaned, 
                        path=['Gender', 'Department', 'JobRole'],
                        color_continuous_scale='RdBu',
                        color='MonthlyIncome',  
                        hover_data={'Gender': True, 'Department': True, 'JobRole': True, 'MonthlyIncome': True}
                        )
        fig.update_layout(width=550, height=400, margin=dict(t=0, b=0, l=0, r=0))  # Adjust margin to reduce distance

        st.plotly_chart(fig)
    
    # 6. Distance Tree Map
    def distance_treemap(df_grouped):
        st.markdown("#### WorkLifeBalance based on Gender, Distance, and Overtime")
        fig = px.treemap(df_grouped, 
                        path=['Gender', 'DistanceFromHome_Group', 'OverTime'],
                        color_continuous_scale='RdBu',
                        color='WorkLifeBalance',  
                        hover_data={'Gender': True, 'DistanceFromHome_Group': True, 'OverTime': True, 'WorkLifeBalance': True}
                        )

        fig.update_layout(width=550, height=400, margin=dict(t=0, b=0, l=0, r=0))  # Adjust margin to reduce distance

        st.plotly_chart(fig)


    # 7. Department gender
    def marital_treemap(df_chart, selected_attrition):
        st.markdown("#### Marital Status based on Gender, and Department")
        fig = px.treemap(df_chart, 
                        path=['MaritalStatus', 'Gender', 'Department'],
                        color='MaritalStatus',
                        color_discrete_map={'Married': 'blues', 'Single': 'greys', 'Divorced': 'greens'},  
                        hover_data={'Gender': True, 'MaritalStatus': True, 'Department': True}
                        )

        fig.update_layout(width=500, height=300, margin=dict(t=0, b=0, l=0, r=0))  # Adjust margin to reduce distance

        st.plotly_chart(fig)


    
    # 8. Environment Tree Map
    def environmentsatisfaction_treemap(df_grouped):
        st.markdown("#### EnvironmentSatisfaction based on Age, Deparment, and Overtime")
        fig = px.treemap(df_grouped, 
                        path=['Department', 'OverTime','Age_Group'],
                        color_continuous_scale='RdBu',
                        color='EnvironmentSatisfaction',  
                        hover_data={'OverTime': True, 'Age_Group': True, 'Department': True, 'EnvironmentSatisfaction': True}
                        )
        fig.update_layout(
            width=600, 
            height=400, 
            margin=dict(t=0, b=0, l=0, r=0),
            coloraxis_colorbar=dict(title='EnvSatisfaction')
       ) 

        st.plotly_chart(fig)

    
      # 9. JobRole Tree Map
    def relationship_treemap(df_grouped):
        st.markdown("#### RelationshipSatisfaction based on Gender, Age, and Deparment")
        fig = px.treemap(df_grouped, 
                        path=['Gender', 'Age_Group', 'Department'],
                        color_continuous_scale='RdBu',
                        color='RelationshipSatisfaction',  
                        hover_data={'Gender': True, 'Age_Group': True, 'Department': True, 'RelationshipSatisfaction': True}
                        )
        fig.update_layout(width=600, height=400, margin=dict(t=0, b=0, l=0, r=0))  # Adjust margin to reduce distance

        st.plotly_chart(fig)
    
    # 10. Job involvement on jobrole
    def jobrole_inv(df_metrics):
        df_job_involvement_counts = df_metrics.groupby(['JobRole', 'JobInvolvement', 'Attrition']).size().reset_index(name='Count')

        # Area Chart for Job Involvement and Attrition
        st.markdown('### Job Involvement by Job Role and Attrition')
        fig_job_involvement_area = px.area(df_job_involvement_counts, x='JobInvolvement', y='Count', color='Attrition', line_group='JobRole',
                                        facet_col='JobRole', facet_col_wrap=3,
                                        labels={'JobInvolvement': 'Job Involvement', 'Count': 'Count of Employees', 'JobRole': 'Job Role'})
        fig_job_involvement_area.update_traces(mode='lines+markers')
        st.plotly_chart(fig_job_involvement_area)

    # 11. Job level on jobrole
    def jobrole_level(df_metrics):
        df_job_level_counts = df_metrics.groupby(['JobRole', 'JobLevel']).size().reset_index(name='Count')

        # Line Chart for Job Level counts by Job Role
        fig_job_level_line = px.line(
            df_job_level_counts,
            x='JobLevel',
            y='Count',
            color='JobRole',
            markers=True, 
            labels={'JobLevel': 'Job Level', 'Count': 'Count of Employees', 'JobRole': 'Job Role'}
        )
        
         
        fig_job_level_line.update_layout(
            width=550, 
            height=370,  
            margin=dict(t=80, b=0, l=0, r=0),
            title={
                'text': 'Job Level Counts by Job Role',
                'font': {
                    'color': 'white',
                    'size': 27
                },
                'x': 0.5,
                'xanchor': 'center',
                'y': 0.9,  
                'yanchor': 'top'
            },
            xaxis=dict(
                tickmode='array', 
                tickvals=sorted(df_job_level_counts['JobLevel'].unique()),
                ticktext=[str(int(x)) for x in sorted(df_job_level_counts['JobLevel'].unique())],
                tickangle=0  
        )
    )
        st.plotly_chart(fig_job_level_line)

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
        st.markdown('#### Attrition Percentage by Gender and Distance From Home')
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

        # Set the size of the figure
        fig_attrition_bar.update_layout(
        width=400,  
        height=300 
        )

        st.plotly_chart(fig_attrition_bar)
    
    
    # 13. Overtime treemap
    def overtime_treemap(df_df_metrics):
        st.markdown("#### EnvSatisfaction based on Department, TotalWorkingYears, and OverTime")
        # Ensure the DataFrame contains the necessary columns
        required_columns = ['Department', 'TotalWorkingYears_Group', 'OverTime', 'JobSatisfaction']
        if not all(column in df_chart.columns for column in required_columns):
            st.error(f"The DataFrame must contain the following columns: {', '.join(required_columns)}")
            return
        
        # Convert TrainingTimesLastYear to string
        df_chart['TotalWorkingYears_Group'] = df_metrics['TotalWorkingYears_Group'].astype(str)
        
        # Add a count column for the values in the treemap
        df_chart['Count'] = 1
        
        # Create the treemap
        fig = px.treemap(
            df_chart, 
            path=['Department','TotalWorkingYears_Group', 'OverTime'], 
            values='Count',  
            color='EnvironmentSatisfaction',  
            color_continuous_scale='Blues',
            labels={
                'TotalWorkingYears_Group': 'Total Working Years',
                'Department': 'Department',
                'EnvironmentSatisfaction': 'Env Satisfaction',
                'Count': 'Count',
                'OverTime': 'Overtime'
            }
        )
        
        fig.update_layout(
            width=550, 
            height=400,
            margin=dict(t=0, l=0, r=0, b=0), 
            coloraxis_colorbar=dict(title='EnvSatisfaction')
        )
        
        # Display the treemap in Streamlit
        st.plotly_chart(fig)

#########################

# Streamlit application

#########################
    
    # Set the demographic columns
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns([1, 1.5, 2, 2])
    with col1:
        overall_attrition_metric(df_cleaned)
        total_employee_metric(df_metrics)
        percent_gender_male(df_metrics)
        percent_gender_female(df_metrics)
        percent_single(df_metrics)
        percent_married(df_metrics)
        percent_divorced(df_metrics)
        median_age_metric(df_metrics, overall_median_age)
    with col2:
        gender_chart(df_metrics)
        attrition_percentage_by_gender_and_distance(df_grouped, df_metrics, selected_attrition)
        
    with col3:
        age_chart(df_chart)
        jobrole_chart(df_chart)
    with col4:
        marital_treemap(df_chart, selected_attrition) 
        jobrole_level(df_metrics) 

      ########### middle #############

     # Second filter bar for middle features
    st.markdown("<h2 style='text-align: center;'>Selected Attrition on Job Metrics</h2>", unsafe_allow_html=True)
    
    selected_attrition_middle = st.selectbox('Select Attrition (Job Metrics)', ['All Data', 'Yes', 'No'])
    # Filter data based on attrition
    if selected_attrition_middle != 'All Data':
        attrition_value_middle = 1 if selected_attrition_middle == 'Yes' else 0
        df_chart_middle = df_grouped[df_grouped['Attrition'] == attrition_value_middle]
        df_metrics_middle = df_cleaned[df_cleaned['Attrition'] == attrition_value_middle]

    else:
        df_chart_middle = df_grouped
        df_metrics_middle = df_cleaned


    # Column set up
    col21, col22, col23, col24, col25, col26 = st.columns(6)
    with col21:
        median_income(df_metrics_middle)
    with col22:        
        mode_stock(df_metrics_middle)
    with col23:
        median_env(df_metrics_middle)    
    with col24:
        median_wlb(df_metrics_middle)
    with col25:
        median_jobinvolvement(df_metrics_middle)
    with col26:
        percent_overtime_yes(df_metrics_middle)

    col27, col28, col29 = st.columns(3)
    with col27:
       jobrole_treemap(df_metrics_middle)
    with col28:
       environmentsatisfaction_treemap(df_chart_middle)
    with col29:
       overtime_treemap(df_chart_middle)
        
    col30, col31, col32 = st.columns([1.5, 1, 1])
    with col30:
        jobrole_inv(df_metrics_middle)
    with col31:
        relationship_treemap(df_chart_middle)
    with col32:
         distance_treemap(df_chart_middle)
        

      ########### bottom #############
    # Third filter bar for bottom features
    st.markdown("<h2 style='text-align: center;'>Selected Attrition on MonthlyIncome Features</h2>", unsafe_allow_html=True)
    
    selected_attrition_bottom = st.selectbox('Select Attrition (Monthly Income)', ['All Data', 'Yes', 'No'])

    # Filter data based on attrition
    if selected_attrition_bottom != 'All Data':
        attrition_value_bottom = 1 if selected_attrition_bottom == 'Yes' else 0
        df_chart_bottom = df_grouped[df_grouped['Attrition'] == attrition_value_bottom]
        df_metrics_bottom = df_cleaned[df_cleaned['Attrition'] == attrition_value_bottom]

    else:
        df_chart_bottom = df_grouped
        df_metrics_bottom = df_cleaned

      
    
    col41, col42, col43, col44, col45 = st.columns(5)
    with col41:
        median_income_research_director(df_metrics_bottom)
    with col42:
        median_income_manager(df_metrics_bottom)
    with col43:
        median_income_healthcare_representative(df_metrics_bottom)
    with col44:
        median_income_manufacturing_director(df_metrics_bottom)
    with col45:
        median_income_sales_executive(df_metrics_bottom)
        
    col46, col47, col48, col49 = st.columns(4)
    with col46:
        median_income_human_resources(df_metrics_bottom)
    with col47:
        median_income_laboratory_technician(df_metrics_bottom)
    with col48:
        median_income_research_scientist(df_metrics_bottom)
    with col49:
        median_income_sales_representative(df_metrics_bottom) 
        
    st.write("")
    
    st.markdown("<h3 style='text-align: center;'>JobRole-JobLevel-specified MonthlyIncome Features based on Selected Attrition</h3>", unsafe_allow_html=True)
    st.write("")
    cola, colb, colc, cold = st.columns(4)
    with cola:
        jobrole_monthly_income_metrics(df_metrics_bottom, df_cleaned, 'Human Resources')
    with colb:
        jobrole_monthly_income_metrics(df_metrics_bottom, df_cleaned, 'Laboratory Technician')
    with colc:
        jobrole_monthly_income_metrics(df_metrics_bottom, df_cleaned, 'Research Scientist')
    with cold:
        jobrole_monthly_income_metrics(df_metrics_bottom, df_cleaned, 'Sales Representative')
    
    
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
  
    **1. Demographic Factors:**
    - Higher attrition rates are observed among male employees, singles, and those aged 28-32 years.
    - Employees living 20-30 km away from the office are more likely to leave.
    
    **2. Job Role and Job Level:**
    - The highest attrition rates are found in job roles such as Laboratory Technician, Sales Executive, Research Scientist, and Sales Representative, especially at Job Level 1.
    
    **3. Income:**
    - Median monthly income is generally lower in roles with high attrition, particularly for Laboratory Technicians, Research Scientists, and Sales Representatives.
    
    **4. Work Environment Satisfaction:**
    - Median environment satisfaction is lower among employees who leave.
    - 54.75% of employees who leave experience overtime, but overtime itself does not significantly impact environment satisfaction.
    - Interestingly, lower environment satisfaction is more common among those not working overtime, possibly because these employees feel less motivated in their work.
    - Lower environment satisfaction is more prevalent among employees aged 43-47 who do not work overtime.
    
    **5.Total Working Years:**
    - Employees with 0-3 years in Human Resources, 26-40 years in Sales, and 11-15 years in Research and Development have the lowest environment satisfaction.
    
    **6. Job Involvement and Relationship Satisfaction:**
    - The highest attrition is associated with job involvement level 3.
    - Male employees generally have higher relationship satisfaction than females.
    - The lowest satisfaction is in males aged 33-37 in Sales and R&D and females aged 18-22 and 33-37 in R&D, and 23-27 in Sales and HR.
    
    **7. Work-Life Balance (WLB):**
    - Women with a commute of 0-9 km have the best work-life balance regardless of overtime status.
    - Men generally have moderate to low work-life balance across all distances unless they work overtime.
    
    **8. Income Discrepancies:**
    - Significant income discrepancies are found in HR Level 1, Laboratory Technicians Level 3, Research Scientist Level 1, and Sales Representative Level 2.
             

    ## Business Action Recommendations
    To address these issues and reduce attrition rates, the following actions are recommended:
    
    **1. Targeted Retention Programs:**
    - For Demographics: Develop retention initiatives specifically for male, single employees aged 28-32, and those living 20-30 km from the office. This can include mentorship programs, community-building activities, and social events to increase engagement and satisfaction.
    - For Commuters: Offer incentives like transportation subsidies, flexible working hours, and remote work options to reduce commute stress and improve work-life balance for employees living further away.
   
    **2. Compensation and Benefits Adjustment:**
    - Address Income Disparities: Conduct regular salary reviews to ensure competitive and fair compensation, particularly for high-attrition roles such as Laboratory Technicians, Research Scientists, and Sales Representatives. Adjust compensation for roles and levels identified with significant income discrepancies (HR Level 1, Laboratory Level 3, Research Scientist Level 1, and Sales Representative Level 2).
    - Enhance Benefits Packages: Improve benefits packages to include comprehensive health care, nutritious lunch and snacks, wellness programs, and performance bonuses.
    
    **3. Improve Work Environment Satisfaction:**
    - Environment Enhancement: Implement initiatives to improve the physical and cultural work environment, such as modernizing office spaces, providing ergonomic furniture, place a mini garden next to the RnD department office, and creating a more inclusive and collaborative culture.
    - Provide mental health consultation center, team-building program, and regular company outing. 
    - Recognition and Reward Programs: Establish programs to recognize and reward employees' contributions regularly, boosting morale and satisfaction. Special focus should be given to employees aged 43-47 who report lower satisfaction levels.
    - Support for Overtime Workers: Develop structured support for employees who work overtime, ensuring they have access to resources and recognition for their efforts.
    
    **4. Career Development and Support Programs:**
    - Training and Development: Provide continuous learning opportunities, skill development programs, and clear career progression paths to increase job involvement and satisfaction, particularly for roles with high attrition.
    - Mentorship and Coaching: Implement mentorship programs to guide employees in their career growth and foster a supportive work environment. This is especially important for employees in their early and mid-career stages in Human Resources, Sales, and Research and Development.
    - Job Involvement Strategies: Increase job involvement by ensuring roles are meaningful and employees understand their impact on the organization. Engage employees through challenging projects and decision-making opportunities.

    **5. Hear Employee's Toughts:**
    - Provide a suggestion box for employees to share their thoughts, and then conduct regular evaluations based on the feedback received.
             
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
