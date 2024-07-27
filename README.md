# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

### Business Understanding
**Business Background:**
Jaya Jaya Maju is a multinational company established in 2000. With over 1000 employees spread across the country, the company faces challenges in managing its workforce, particularly concerning the high attrition rate, which is currently over 10%.

**Business Problem:**
The main issue is identifying the factors intricately driving the high attrition rate across diverse employee characteristics. Generally, overtime is one of the factors linked to burnout, leading to higher attrition, and should be avoided. However, interestingly, at Jaya Jaya Maju, although overtime is more prevalent in the attrition "yes" group, data shows that employees who work overtime tend to have higher environmental satisfaction. This could indicate either a positive work environment that encourages overtime or those who work overtime have a better perception of work environment and higher motivation to perform. This highlights the importance of understanding the data in order to have better management plan. However, this is just one of several factors, and different individuals may have different circumstances and different approach. If these factors are not addressed properly and promptly, long-term risks may include decreased productivity, lower employee morale and satisfaction, higher attrition, and increased recruitment and training costs for new employees, all of which can negatively impact the overall performance of the company. Analyzing these factors through data analytics based on group and pattern approaches is important for Human Resources (HR) management to observe, learn, and implement effective strategies to reduce attrition. Having a business attrition dashboard that presents the visualisation of data pattern is crucial for that. 

**Project Scope:**
This project involves conducting employee data analysis to identify factors contributing to the high attrition rate at Jaya Jaya Maju through a group approach, and present the visualization through analytical dashboard. The project encompasses identifying demographic or general overviews based on attrition group and then delving deeper into factors with stronger correlations to attrition, such as job involvement, environment satisfaction, work-life balance, and monthly income. Various visualization styles are applied to effectively portray the patterns of characteristic groups, with only features that significantly contribute being included. These visualizations are presented through a Streamlit attrition dashboard, and the data is processed using a machine learning model to produce an attrition predictor that was deployed through the same Streamlit application. These insights can be utilized to identify characteristic patterns within each employee group. By understanding these patterns, the company can formulate targeted approaches to reduce the attrition rate.

### Preparation
**Data Source:**
The data source to be used is a dataset from Jaya Jaya Maju, which can be downloaded from [this link](https://github.com/dicodingacademy/a590-Belajar-Penerapan-Data-Science/tree/7cb1fd79a2914f6990d47f1dfc6e60c588c1a6ae/a590_proyek_pertama).

**Download scripts:**
Download necessary files. All files are available at https://github.com/Ngno/jayajaya/tree/main:
   a. **Download Requirement Script:**
       `requirements.txt`

   b. **Download the Dashboard Python Script:**
      `dashboard.py`

   c. **Download the Trained Model:**
      `trained_rf_model.joblib`.

# Setup Environment

To establish a suitable development environment for this project, ensure you have Python installed. You can set up a virtual environment using Anaconda or pipenv. Here's how you can proceed. You can change the environment name (`env-name`) to your preferred name:

## Setup Environment - Anaconda
```
conda create --name env-name python=3.11.4
conda activate env-name
pip install -r requirements.txt
```

## Setup Environment - Shell/Terminal
```
mkdir env-name
cd env-name
pip install pipenv
pipenv shell
pip install -r requirements.txt
```

Ensure all project files are located in the same directory as your environment directory.
You can check the directory by:

## Find Directory - Anaconda
```
conda info --envs
```

## Find Directory - Shell/Terminal:
```
pipenv --venv
```


## Run steamlit app
Once the environment is set up and all project files are located in the same directory, you can proceed to run the Streamlit app using the command:

```
streamlit run dashboard.py
```

### Business Dashboard
The dashboard is published on Streamlit here: https://attrition-dashboard.streamlit.app/
The business dashboard app consists of four pages: JJM  Attrition Dashboard, About, Attrition Analysis, Attrition Predictor. 
- **JJM Attrition Dashboard:** provides visualizations of the factors affecting the attrition rate, such as demoographic factors, monthly income, distance, environment satisfaction, relationship satisfaction, and work-life balance. This dashboard will offer HR managers insights into employee group conditions, enabling them to make better decisions.
- **About:** provides information about the dashboard app in general, dataset, and data grouping.
- **Attrition Analysis:** provides numerical tables related to correlation between features and attrition. Conclusion and recommendation action are included here
- **Attrition Predictor:** provides a machine learning model deployment to help user predict the probability of employees' attrition based on inputed value.


### Conclusion
The analysis indicates that attrition is notably higher among male, single employees aged 28-32 years who live 20-30 km from the office, particularly in the roles of Laboratory Technician, Sales Executive, Research Scientist, and Sales Representative at Job Level 1. These positions are also characterized by lower median monthly incomes, with significant income discrepancies in HR Level 1, Laboratory Technicians Level 3, Research Scientist Level 1, and Sales Representative Level 2. Additionally, departing employees report lower work environment satisfaction, with 54.75% experiencing overtime. Interestingly, those not working overtime, especially aged 43-47, report even lower satisfaction, suggesting possible disengagement. Attrition is highest among employees with job involvement level 3. Lower relationship satisfaction is also seen in attrition 'yes', mostly in female aged 18-37, while male tends to have more stabil relationship satisfaction except for age group 33-37 within Sales and Research & Development (R&D) department. Work-life balance is another critical issue, with women having better balance with shorter commutes (0-9 km) and men generally struggling unless they who work overtime.

### Action Items to be Recommended
To address the identified issues and reduce the attrition rate, the following actions are recommended:

**Action Item 1: Enhance Compensation and Benefits**
   - Increase Compensation for High-Attrition Roles: Address the lower median monthly incomes in high-attrition roles such as Laboratory Technician, Research Scientist, and Sales Representative. Conduct a market salary benchmarking exercise and adjust compensation packages to be more competitive, especially for Job Level 1 positions. Implement performance-based incentives and bonuses to enhance job satisfaction and retention.
   - Offer additional benefits like transportation allowances for employees living far from the office and performance-based bonuses to incentivize and retain talent.

**Action Item 2: Enhance Work Environment and Engagement Programs:**
   - Improve work environment satisfaction by providing regular feedback mechanisms, such as suggestion boxes or anonymous surveys, and act on the feedback. 
   - Introduce mental health support services and organize annual company retreats or team-building events to foster a more engaging and supportive work environment.
   - Encourage flexible work arrangements, particularly for employees aged 43-47, to address dissatisfaction among those not working overtime.
   - Improve environment satisfaction by improving comfortability and convenience for indoor workers with low work-life balance such as Laboratory Technicians and Research Scientists. As instance, providing ergonomist chairs and a fresh mini garden next to the Research and Development department. 

**Action Item 3: Promote Work-Life Balance Initiatives**
   - Implement policies to support work-life balance, such as flexible working hours, remote work options, and reduced commute times. 
   - For roles with high overtime, ensure that employees are compensated fairly and recognize their efforts through additional time off or rewards. 
   - Encourage a culture of taking regular breaks and vacations to prevent burnout.
