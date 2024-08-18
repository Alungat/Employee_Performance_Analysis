import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import io
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import warnings

# Ignore Warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Employee Performance Evaluation App",
    page_icon="🏢",
)

st.write("# Welcome to the Employee Evaluation App")
st.sidebar.success("Select a demo above.")

st.markdown(
    """Employee Performance evaluation is one of the crucial processes in most organizations. These evaluations provide a structured framework for assessing and 
    managing employee contributions, development, and overall effectiveness within the company. This in turn ultimately contributes to organizational effectiveness 
    and growth. Employee performance and effectiveness towards organizational goals are vital elements that impact an organization's objectives, 
    to which low employee performance and effectiveness negatively affect service delivery and client satisfaction.

The significant drop in employee effectiveness and morale at INX Future Inc (INX) - one of the leading data analytics and automation solutions providers, 
made Mr. Brain, the CEO at INX, initiate a data science project to analyze employee performance data, identify the underlying causes, and make informed decisions 
to address the issues without harming overall employee morale.

The CEO notes the need to analyze the current employee data and find the core underlying causes of these performance issues. Additionally, he expects clear indicators 
of non-performing employees so that any penalization of non-performing employees, if required, may not significantly affect other employee morals.

This project will employ predictive analytics and a machine learning model to forecast outcomes. By utilizing historical data, the model will be trained to generate 
precise or estimated values for new datasets. The implementation of a Machine Learning model within this learning framework aims to enhance prediction accuracy. 
Ultimately, these insights will empower Mr. Brain to make informed decisions."""
)

# Hiding the Streamlit rerun menu from the user
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Configure the page
st.write("# Our Data")
st.sidebar.success("Data Facts")
st.write('## Employee Performance Data Description')

st.markdown("""Definition of Attributes:

EmpNumber: Employee number is an identification code assigned by employers to individual employees

Age: Employee's age

Gender: Employee's gender

EducationBackground:is the educational accomplishments acquired by the employee

MaritalStatus:defines whether the employee is married single or divorced .

EmpDepartment: Department the employee works in.

EmpJobRole: Job role of the employee.

BusinessTravelFrequency: Frequency of business travel

DistanceFromHome: Distance from home to workplace

EmpEducationLevel: Education level on a scale of 1-5 1 being Below College and 5 being Doctor.

EmpEnvironmentSatisfaction: Satisfaction with the work environment

EmpHourlyRate: Hourly rate of the employee

EmpJobInvolvement: Level of job involvement on a scale of 1 to 4 with 1 being Low and 4 being Very high

EmpJobLevel: Job level within the company

EmpJobSatisfaction: Job satisfaction level on a scale of 1 to 4 with 1 being Low and 4 being Very high.

NumCompaniesWorked: Number of companies the employee has worked at furing his carrer.

OverTime: Whether the employee works overtime

EmpLastSalaryHikePercent: Last salary hike percentage

EmpRelationshipSatisfaction: Relationship satisfaction level on a scale of 1 to 4 with 1 being Low and 4 being Very high.

TotalWorkExperienceInYears: Total years of work experience during their carrer journey

TrainingTimesLastYear: Number of training sessions attended last year.

EmpWorkLifeBalance: Work-life balance satisfaction on a scale of 1 to 4 with 1 being Bad and 4 being Best.

ExperienceYearsAtThisCompany: Years of experience at the current company.

ExperienceYearsInCurrentRole: Years of experience in the current role.

YearsSinceLastPromotion: Years since the last promotion.

YearsWithCurrManager: Years with the current manager.

Attrition: Whether the employee has left the company.

PerformanceRating: Employee performance rating on a scale of 1 to 4 with 1 being Low and 4 Outstanding
""")

try:
    # Loading data
    data = pd.read_excel('INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8 (1).xlsx')

    # Sidebar for user options
    st.sidebar.header("Options")
    columns_to_show = st.sidebar.multiselect("Select columns to display", options=data.columns, default=list(data.columns))

    # Ensure there is at least one column selected
    if not columns_to_show:
        st.error("Please select at least one column to display.")
    else:
        # Getting user input on how many rows to display
        rows = st.slider('Slide along to view more parts of the data', 0, 15, 5)

        # Display the dataframe
        st.write("### Data Preview")
        st.dataframe(data[columns_to_show].head(rows))

        # Display basic information
        st.write("### Data Information")
        buffer = io.StringIO()
        data.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        # Display descriptive statistics
        st.write("### Descriptive Statistics")
        st.write(data.describe(include='all'))

        # Missing values
        st.write("### Missing Values")
        st.write(data[columns_to_show].isnull().sum())

except Exception as e:
    st.error(f"Error loading the file: {e}")
    print("An error occurred:", e)

st.markdown("""
* **Columns:** The data has 28 columns
* **Rows:** There are a total of 1200 rows
* **DataTypes:** There are 2 data types: int64(19) and object(9)
* **Duplicates:** There are no duplicates
* **Missing Values:** There are no missing values
""")

# Hiding the Streamlit rerun menu from the user
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Sidebar for visualization options
st.sidebar.header("Visualization Options")
visualization_type = st.sidebar.selectbox(
    "Select the type of visualization",
    ["Correlation Heatmap", "Distribution Plot", "Box Plot", "Pair Plot"]
)

# Loading data
data = pd.read_excel('INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8 (1).xlsx')

# Drop unnecessary columns
data.drop('EmpNumber', axis=1, inplace=True)

# Encode categorical variables using LabelEncoder
label_encoders = {}
for col in data.select_dtypes(include='object').columns:
    LE = LabelEncoder()
    data[col] = LE.fit_transform(data[col])
    label_encoders[col] = LE

# Correlation Heatmap
if visualization_type == "Correlation Heatmap":
    st.write("### Correlation Heatmap")
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt='.2f')
    st.pyplot(plt)

# Distribution Plot
elif visualization_type == "Distribution Plot":
    st.write("### Distribution Plot")
    column = st.selectbox("Select a column for distribution", data.columns)
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], kde=True)
    st.pyplot(plt)

# Box Plot
elif visualization_type == "Box Plot":
    st.write("### Box Plot")
    column = st.selectbox("Select a column for box plot", data.columns)
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=data[column])
    st.pyplot(plt)

# Pair Plot
elif visualization_type == "Pair Plot":
    st.write("### Pair Plot")
    selected_columns = st.multiselect("Select columns for pair plot", data.columns, default=list(data.columns)[:4])
    if len(selected_columns) < 2:
        st.error("Please select at least two columns for the pair plot.")
    else:
        sns.pairplot(data[selected_columns])
        st.pyplot(plt)

# Create features and target variables
X = data.drop(['PerformanceRating'], axis=1)
y = data['PerformanceRating']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Get user input
EmpNumber = st.text_input('What is your EmpNumber?').capitalize()

if EmpNumber:
    st.write(f"Hello {EmpNumber}, please complete the form below.")
else:
    st.write("Please enter your EmpNumber")

# Instantiate and fit the model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Get feature input from user
st.subheader('User Inputs')
col1, col2 = st.columns(2)
with col1:
    Age = st.number_input('Employee age', min_value=0)
    EmpDepartment = st.selectbox('Employee Department', label_encoders['EmpDepartment'].classes_)
    EmpJobRole = st.selectbox('Employee Role', label_encoders['EmpJobRole'].classes_)
    DistanceFromHome = st.number_input('Distance from home in miles', min_value=0)
    EmpEnvironmentSatisfaction = st.selectbox('Environment Satisfaction', [1, 2, 3, 4])

with col2:
    EmpHourlyRate = st.number_input('Hourly Rate in Dollars', min_value=0)
    EmpLastSalaryHikePercent = st.number_input('Salary Hike Percentage', min_value=0)
    TotalWorkExperienceInYears = st.number_input('Work Experience', min_value=0)
    EmpWorkLifeBalance = st.selectbox('Work-Life Balance', [1, 2, 3, 4])
    ExperienceYearsAtThisCompany = st.number_input('Experience in this Company: Years', min_value=0)
    ExperienceYearsInCurrentRole = st.number_input('Experience in this Role: Years', min_value=0)
    YearsSinceLastPromotion = st.number_input('Time since last promotion: Years', min_value=0)
    YearsWithCurrManager = st.number_input('Years with current manager', min_value=0)

# Transform user input to match the training data encoding
user_input = pd.DataFrame({
    'Age': [Age],
    'EmpDepartment': [label_encoders['EmpDepartment'].transform([EmpDepartment])[0]],
    'EmpJobRole': [label_encoders['EmpJobRole'].transform([EmpJobRole])[0]],
    'DistanceFromHome': [DistanceFromHome],
    'EmpEnvironmentSatisfaction': [EmpEnvironmentSatisfaction],
    'EmpHourlyRate': [EmpHourlyRate],
    'EmpLastSalaryHikePercent': [EmpLastSalaryHikePercent],
    'TotalWorkExperienceInYears': [TotalWorkExperienceInYears],
    'EmpWorkLifeBalance': [EmpWorkLifeBalance],
    'ExperienceYearsAtThisCompany': [ExperienceYearsAtThisCompany],
    'ExperienceYearsInCurrentRole': [ExperienceYearsInCurrentRole],
    'YearsSinceLastPromotion': [YearsSinceLastPromotion],
    'YearsWithCurrManager': [YearsWithCurrManager]
})

# Ensure all features match the model's training features
missing_features = set(X_train.columns) - set(user_input.columns)
for feature in missing_features:
    user_input[feature] = 0  # or a default value appropriate for your use case

user_input = user_input[X_train.columns]

# Display user input
st.subheader('User input values')
st.dataframe(user_input)

# Making predictions
prediction = rf.predict(user_input)

# Display prediction
performance_labels = ['LOW performer', 'GOOD performer', 'EXCELLENT performer', 'OUTSTANDING performer']
st.text(f'The Employee is a {performance_labels[prediction[0] - 1]}')

#Display the model accuracy
st.write("Model accuracy: ", round(metrics.accuracy_score(y_test, rf.predict(X_test)), 2) * 100)

# Hiding the Streamlit rerun menu from the user
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
