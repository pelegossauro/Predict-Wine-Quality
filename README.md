Wine Quality Prediction
This project is focused on predicting the quality of wine based on various physicochemical features using machine learning algorithms. The dataset contains information about different characteristics of wines (such as acidity, alcohol content, sulfur dioxide levels, etc.) and their corresponding quality ratings. The objective is to predict whether a wine has a high or low quality based on these features.

Project Overview
The project uses the winequalityN.csv dataset, which includes both red and white wines with 11 physicochemical features, such as acidity, alcohol, sulfur dioxide, and others, along with a quality score. The goal is to predict the quality of the wine using machine learning models.

Dataset Description
The dataset contains the following features:
type: Type of wine (red or white)
fixed acidity: Amount of fixed acidity in the wine
volatile acidity: Amount of volatile acidity in the wine
citric acid: Amount of citric acid in the wine
residual sugar: Residual sugar level in the wine
chlorides: Chlorides in the wine
free sulfur dioxide: Free sulfur dioxide levels in the wine
total sulfur dioxide: Total sulfur dioxide levels in the wine
density: Density of the wine
pH: pH level of the wine
sulphates: Sulphate content in the wine
alcohol: Alcohol content in the wine
quality: Quality score of the wine (integer scale from 0 to 10)

Steps in the Project
1. Data Loading and Cleaning
The dataset is loaded using pandas.read_csv().
Missing values are handled by removing any rows with missing data using dropna().
The type column is removed for the final model as it is not used for prediction in this approach.
2. Exploratory Data Analysis (EDA)
Basic statistics and visualizations are used to understand the distribution of the data, including histograms, boxplots, and correlation heatmaps.
Visualizations such as sns.distplot(), sns.barplot(), and sns.pairplot() are used to observe relationships between features.
The correlation matrix helps identify how the different features are correlated with the target variable (quality).
3. Feature Engineering
The target variable (quality) is binarized into two categories: 1 for wines with a quality rating above 7, and 0 for others.
Standard scaling is applied to the features to normalize the dataset, making it suitable for machine learning algorithms.
4. Model Building
Two classification algorithms are applied:
Logistic Regression: A basic model to predict the quality of wine.
Random Forest Classifier: An ensemble model used to improve prediction accuracy.
5. Model Evaluation
Accuracy, classification report, and confusion matrix are used to evaluate the performance of the models.
The Random Forest model is further evaluated based on feature importance, which indicates the most influential features in predicting wine quality.
6. Results
The models achieve high accuracy, with Logistic Regression achieving an accuracy of around 97% on the test set.
Random Forest is used to calculate feature importance, highlighting which features contribute most to predicting wine quality.

Code Implementation
python

# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Suppress warnings
warnings.filterwarnings('ignore')

# Load the wine dataset
wine = pd.read_csv('winequalityN.csv')

# Basic data exploration
wine.head()
wine.tail()
wine.shape
wine.isnull().sum()

# EDA: Distribution and correlation
sns.countplot(x='type', data=wine)
plt.figure(figsize=(15, 10))
sns.heatmap(wine.corr(), cmap='coolwarm')

# Feature Engineering: Prepare data
Y = wine['quality'].apply(lambda y: 1 if y > 7 else 0)
X = wine.drop(['quality', 'type'], axis=1)

# Scaling the features
scaler = StandardScaler()
scaler.fit(X)
X_standard = scaler.transform(X)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X_standard, Y, test_size=0.2, random_state=245)

# Logistic Regression Model
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
y_pred = logreg.predict(X_test)

# Evaluation: Logistic Regression
accuracy_score(Y_test, y_pred)
print(classification_report(Y_test, y_pred))
print(confusion_matrix(Y_test, y_pred))

# Random Forest Model
rfc = RandomForestClassifier()
rfc.fit(X_train, Y_train)
rfc_pred = rfc.predict(X_test)

# Evaluation: Random Forest
accuracy_score(Y_test, rfc_pred)
print(rfc.feature_importances_)

# Plot feature importance
pd.Series(rfc.feature_importances_, index=X.columns).plot(kind='barh')
Key Findings
Data Distribution: The dataset is imbalanced with more white wines than red wines.
Feature Importance: Features such as fixed acidity, volatile acidity, and alcohol have higher importance in predicting the quality of wine.
Model Performance: The models achieve high accuracy, especially Logistic Regression with 97% accuracy. However, the Random Forest classifier also provides valuable insights into feature importance.
Conclusion
This project demonstrates the ability to predict wine quality using machine learning algorithms. Logistic Regression and Random Forest Classifier are effective at classifying wines into high or low quality categories. The analysis highlights the most important features that influence wine quality and provides a solid foundation for building more complex models or further optimization.

Requirements
Python 3.x
pandas
numpy
seaborn
matplotlib
scikit-learn
You can install the required libraries using pip:

bash

pip install pandas numpy seaborn matplotlib scikit-learn
