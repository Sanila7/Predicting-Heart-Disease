# Heart Disease Prediction Using Machine Learning

## Overview
This project explores the use of machine learning models to predict whether a patient has heart disease based on clinical parameters. The goal is to build a predictive model capable of achieving an accuracy rate of at least 85%. Various machine learning techniques and Python libraries are utilized to achieve this, including data preprocessing, feature engineering, model training, and evaluation.

## Problem Definition
Given clinical parameters about a patient, can we predict whether or not the patient has heart disease? This task is based on the dataset from the Cleveland Heart Disease Database, which contains several features related to a patient's health.

## Dataset
The dataset comes from the UCI Machine Learning Repository, specifically the Cleveland dataset, and can be accessed [here](https://archive.ics.uci.edu/ml/datasets/heart+Disease). A version of the dataset is also available on Kaggle [here](https://www.kaggle.com/ronitf/heart-disease-uci).

### Features
The dataset consists of the following features:
1. **age** - Age in years
2. **sex** - Sex of the patient (1 = male; 0 = female)
3. **cp** - Chest pain type
   - 0: Typical angina
   - 1: Atypical angina
   - 2: Non-anginal pain
   - 3: Asymptomatic
4. **trestbps** - Resting blood pressure (in mm Hg)
5. **chol** - Serum cholesterol in mg/dl
6. **fbs** - Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
7. **restecg** - Resting electrocardiographic results
8. **thalach** - Maximum heart rate achieved
9. **exang** - Exercise induced angina (1 = yes; 0 = no)
10. **oldpeak** - ST depression induced by exercise relative to rest
11. **slope** - Slope of the peak exercise ST segment
12. **ca** - Number of major vessels (0-3) colored by fluoroscopy
13. **thal** - Thalium stress result (1,3 = normal; 6 = fixed defect; 7 = reversible defect)
14. **target** - Target variable indicating the presence of heart disease (1 = yes; 0 = no)

## Evaluation Criteria
To evaluate the model, an accuracy of at least 85% will be considered satisfactory for predicting heart disease. Precision, recall, F1-score, and the confusion matrix will also be used to assess model performance.

## Tools and Libraries
The following Python libraries are used for data analysis, visualization, and machine learning:
- **pandas** - For data manipulation and analysis
- **NumPy** - For numerical operations
- **Matplotlib** and **Seaborn** - For data visualization
- **Scikit-learn** - For machine learning models, model evaluation, and hyperparameter tuning

## Steps Involved
1. **Data Loading**: The dataset is loaded using pandas to inspect its shape and structure.
2. **Exploratory Data Analysis (EDA)**: Descriptive statistics and visualizations are used to understand the relationships between features.
3. **Model Selection and Training**: Various models like Logistic Regression, K-Nearest Neighbors, and Random Forest are tested for classification.
4. **Model Evaluation**: The models are evaluated using cross-validation and performance metrics like accuracy, precision, recall, F1-score, and the confusion matrix.
5. **Hyperparameter Tuning**: RandomizedSearchCV and GridSearchCV are used to find the best hyperparameters for the models.
6. **Final Model**: The best performing model is selected for final predictions.

## Example Code
```python
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv('heart-disease.csv')

# Preprocess data, train models, and evaluate performance
X = df.drop('target', axis=1)
y = df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model (Logistic Regression example)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
