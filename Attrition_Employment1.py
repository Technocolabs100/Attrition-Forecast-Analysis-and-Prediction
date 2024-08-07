#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
file_path = 'Dataset_HR_Employee_Attrition.csv'
data = pd.read_csv(file_path)

print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())

display(data)


# In[3]:


data = data.drop_duplicates()
data = data.dropna()
print(data.isnull().sum())


# In[4]:



data_encoded = pd.get_dummies(data, columns=['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus'], drop_first=True)

print(data_encoded.head())


# In[5]:


data_encoded.rename(columns={
    'Attrition_Yes': 'Attrition',
    'BusinessTravel_Travel_Frequently': 'Travel_Frequently',
    'BusinessTravel_Travel_Rarely': 'Travel_Rarely',
    'Department_Research & Development': 'Department_RD',
    'Department_Sales': 'Department_Sales',
    'Department_Human Resources': 'Department_HR',
    'Gender_Male': 'Gender_Male',
    'Gender_Female': 'Gender_Female',
    'JobRole_Sales Executive': 'JobRole_SalesExec',
    'JobRole_Research Scientist': 'JobRole_ResearchSci',
    'JobRole_Laboratory Technician': 'JobRole_LabTech',
    'JobRole_Manager': 'JobRole_Manager',
    'JobRole_Other': 'JobRole_Other',
    'MaritalStatus_Married': 'MaritalStatus_Married',
    'MaritalStatus_Single': 'MaritalStatus_Single',
    'MaritalStatus_Divorced': 'MaritalStatus_Divorced'
}, inplace=True)


print(data_encoded.head())


# In[1]:


conda install mkl-service


# In[3]:


import numpy
import mkl


# In[4]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:




