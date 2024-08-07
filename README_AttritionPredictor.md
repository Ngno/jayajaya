# Predicting Attrition with Trained Model

This guide explains how to use the provided Python script (`attrition_predictor.py`) with trained machine learning model (`trained_rf_model.joblib`) to predict attrition.

## Prerequisites

Before getting started, ensure you have the following:

1. Python version 3.11.4 installed on your system. If you haven't:
    a. Visit the official Python website: [Python Downloads](https://www.python.org/downloads/)
    b. Find the version 3.11.4 and click on the link to download the installer for your operating system (Windows, macOS, or Linux).
    c. Run the installer and follow the on-screen instructions.
        - On Windows, make sure to check the box that says "Add Python to PATH" during the installation process.
        - On macOS and Linux, you may need to use `brew` or `apt-get` respectively, if you prefer using a package manager.


2. Downloaded necessary files
    All files are available at https://github.com/Ngno/jayajaya/tree/main:

    a. **Download the Attrition Predictor Requirement Script:**
        - Download the Python script (`requirements_attritionpredictor.txt`).

    b. **Download the Python Script:**
        - Download the Python script (`attrition_predictor.py`).

    c. **Download the Trained Model:**
        - Download the trained model file (`trained_rf_model.joblib`).

    d. **Place the Files in the Same Folder:**
        - Ensure all files are in the same folder.

3. Set up environment and installed requirement libraries.
    This script and model need specific version libraries to be installed. Set up environment to avoid conflict with other apps.
    You can set up environment using Anaconda or pipenv. Here's how you can proceed. You can change the environment name (`env-name`) to your preferred name:
    
    ## Setup Environment - Anaconda
    ```
    conda create --name env-name python=3.11.4
    conda activate env-name
    pip install -r requirements_attritionpredictor.txt
    ```

    ## Setup Environment - Shell/Terminal
    ```
    mkdir env-name
    cd env-name
    pip install pipenv
    pipenv shell
    pip install -r requirements_attritionpredictor.txt
    ```

## Usage

1. **Run the Script:**
    - Open your terminal or command prompt.
    - Navigate to the directory where the Python script `attrition_predictor.py` is located.

        ```
        cd path/to/script/directory
        ```

    - Run the Python script:

        ```
        python attrition_predictor.py
        ```

2. **Follow the Prompts:**
    - Enter the values for employee characteristics based on the prompts.
    - For features with numeric groupings, enter a number as referenced.
        Example: 
        ```
        Enter value for Age_Group_Encoded (0->18-22, 1->23-27, 2->28-32, 3->33-37, 4->38-42, 5->43-47, 6->48-52, 7->53-57, 8->58-60): 0
        ```
        User input to refer age group 18-22: `0`
    - For categorical features, enter the corresponding number.
        Example: 
        ```
        Enter value for BusinessTravel_Encoded (0->Non-Travel, 1->Travel_Rarely, 2->Travel_Frequently): 0
        ```
        User input to refer Non-Travel: `0`
    - For string features, enter the exact string value without quotes.
        Example: 
        ```
        Enter value for Department (choose one from ['Human Resources', 'Research & Development', 'Sales']): Sales
        ```
        User input: `Sales`
    - For categorical features with reverse reference numbers and strings, input the number.
        Example: 
        ```
        Enter value for Education (Below College->1, College->2, Bachelor->3, Master->4, Doctor->5): 3
        ```
        User input: `3`
    - For features without reference information, input a free number.
        Example: 
        ```
        Enter value for TrainingTimesLastYear: 2
        ```
        User input: `2`

    Example prompt sequence:
    ```
    Enter value for Age_Group_Encoded (0->18-22, 1->23-27, 2->28-32, 3->33-37, 4->38-42, 5->43-47, 6->48-52, 7->53-57, 8->58-60): 0
    Enter value for BusinessTravel_Encoded (0->Non-Travel, 1->Travel_Rarely, 2->Travel_Frequently): 0
    Enter value for Department (choose one from ['Human Resources', 'Research & Development', 'Sales']): Sales
    Enter value for DistanceFromHome_Group_Encoded (0->0-9, 1->10-19, 2->20-30): 0
    Enter value for Education (Below College->1, College->2, Bachelor->3, Master->4, Doctor->5): 1
    Enter value for EducationField (choose one from ['Other', 'Medical', 'Life Sciences', 'Marketing', 'Technical Degree', 'Human Resources']): Other
    Enter value for EnvironmentSatisfaction (Low->1, Medium->2, High->3, Very High->4): 1
    Enter value for Gender (choose one from ['Male', 'Female']): Male
    Enter value for JobInvolvement (Low->1, Medium->2, High->3, Very High->4): 1
    Enter value for JobLevel (Level 1->1, Level 2->2, Level 3->3, Level 4->4, Level 5->5): 1
    Enter value for JobRole (choose one from ['Human Resources', 'Healthcare Representative', 'Research Scientist', 'Sales Executive', 'Manager', 'Laboratory Technician', 'Research Director', 'Manufacturing Director', 'Sales Representative']): Manager
    Enter value for JobSatisfaction (Low->1, Medium->2, High->3, Very High->4): 1
    Enter value for MaritalStatus (choose one from ['Married', 'Single', 'Divorced']): Single
    Enter value for MonthlyIncome_Group_Encoded (0->1K-1,999, 1->2K-2,999, 2->3K-3,999, 3->4K-4,999, 4->5K-5,999, 5->6K-6,999, 6->7K-7,999, 7->8K-8,999, 8->9K-9,999, 9->10K-14,999, 10->15K-19,999): 1
    Enter value for NumCompaniesWorked_Group_Encoded (0->0, 1->1, 2->2-3, 3->4-5, 4->6-7, 5->8-9): 0
    Enter value for OverTime_Encoded (0->No, 1->Yes): 1
    Enter value for PercentSalaryHike_Group_Encoded (0->11-15, 1->16-20, 2->21-25): 1
    Enter value for PerformanceRating (Low->1, Medium->2, High->3, Very High->4): 1
    Enter value for RelationshipSatisfaction (Low->1, Medium->2, High->3, Very High->4): 1
    Enter value for StockOptionLevel (Level 0->0, Level 1->1, Level 2->2, Level 3->3): 1
    Enter value for TotalWorkingYears_Group_Encoded (0->0-1, 1->2-3, 2->4-5, 3->6-7, 4->8-9, 5->10, 6->11-15, 7->16-20, 8->21-25, 9->26-30, 10->30-40): 1
    Enter value for TrainingTimesLastYear: 1
    Enter value for WorkLifeBalance (Low->1, Good->2, Excellent->3, Outstanding->4): 1
    Enter value for YearsAtCompany_Group_Encoded (0->0-1, 1->2-3, 2->4-6, 3->7-10, 4->11-15, 5->16-20, 6->21-30, 7->31-40): 1
    Enter value for YearsInCurrentRole_Group_Encoded (0->0-2, 1->3-5, 2->6-8, 3->9-11, 4->12-14, 5->15-18): 1
    Enter value for YearsSinceLastPromotion_Group_Encoded (0->0-2, 1->3-5, 2->6-8, 3->9-11, 4->12-15): 1
    Enter value for YearsWithCurrManager_Group_Encoded (0->0-2, 1->3-5, 2->6-8, 3->9-11, 4->12-14, 5->15-17): 1
    ```

6. **View the Results:**
    - Once the script finishes running, it will display the predicted attrition results based on the input data.
    Example output:
    ```
    Predicted Attrition: Yes
    Attrition Probability: 61.00%
    ```

7. **Explore Further:**
    - Feel free to explore the script further to understand how it works and to customize it according to your needs.

That's it! You have successfully used the Python script to predict attrition using a trained model.

For any questions or issues, please refer to the documentation or contact the script author.
