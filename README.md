# Loan-Prediction-Using-Machine-Learning

## Project Overview
This project aims to predict the likelihood of loan approval for applicants using machine learning techniques. Two supervised learning algorithms, Decision Tree Classification and Naive Bayes Classification, were implemented to build predictive models.

## Table of Contents
- [Introduction](#introduction)
- [Libraries Used](#libraries-used)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Building](#model-building)
  - [Decision Tree Classification](#decision-tree-classification)
  - [Naive Bayes Classification](#naive-bayes-classification)
- [Model Evaluation](#model-evaluation)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction
The project focuses on predicting loan approvals based on applicant data. The goal is to assist financial institutions in making data-driven decisions.

## Libraries Used
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical operations
- **Matplotlib**: Data visualization
- **Scikit-learn**: Machine learning algorithms

## Dataset
The dataset contains information on loan applicants, including:
- ApplicantIncome
- CoapplicantIncome
- LoanAmount
- Loan_Amount_Term
- Credit_History
- Gender, Married, Education, Self_Employed, Property_Area
- Loan_Status (Target Variable)

## Data Preprocessing
- Handling missing values
- Encoding categorical variables
- Feature scaling

## Exploratory Data Analysis (EDA)
Conducted to understand data distribution and relationships between features. Key insights include:
- Income levels among applicants
- Credit history impact on loan approval
- Education level's influence

## Model Building

### Decision Tree Classification
- Splitting data into training and testing sets
- Training the model on the training data
- Evaluating model performance

### Naive Bayes Classification
- Similar steps as Decision Tree, with a focus on Naive Bayes methodology

## Model Evaluation
Models were evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

## Conclusion
The project demonstrated the potential of machine learning for predicting loan approvals, with both Decision Tree and Naive Bayes classifiers showing promising results.

## Future Work
- Implementing advanced algorithms like Random Forest or Gradient Boosting
- Adding more features to improve model accuracy
- Hyperparameter tuning for optimization

## Installation
To run this project, clone the repository and install the required libraries:
```bash
git clone https://github.com/samjoeld/Loan-Prediction-Using-Machine-Learning.git
cd Loan-Prediction-Using-Machine-Learning
pip install -r requirements.txt
