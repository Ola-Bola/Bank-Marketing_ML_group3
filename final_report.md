Final Project Report

Predicting Term Deposit Subscriptions Using Machine Learning

1. Project Overview

This project focuses on predicting whether a client will subscribe to a bank term deposit based on data from direct marketing campaigns. The objective is to develop a machine learning model that can identify potential customers who are more likely to subscribe to the product.

Direct marketing campaigns in the banking sector often rely on phone calls to customers. However, contacting every client is inefficient and costly. By using predictive modelling, financial institutions can target the most promising customers, improving marketing efficiency and reducing operational costs.

The project implements an end-to-end machine learning workflow including data preprocessing, exploratory data analysis, model training, and evaluation.

2. Business Problem

Banks frequently conduct marketing campaigns to promote financial products such as term deposits. These campaigns often involve contacting large numbers of customers, many of whom may not be interested in the product.

The key business challenge is:
How can banks identify customers who are most likely to subscribe to a term deposit before contacting them?

Solving this problem can help banks:
- reduce marketing costs;
- improve conversion rates;
- prioritize high-value customers;
- optimize marketing strategies.

Machine learning provides a data-driven approach to predicting customer behavior based on historical campaign data.

3. Dataset

This project uses the UCI Bank Marketing Dataset, which contains information from marketing campaigns conducted by a Portuguese banking institution.

Dataset source:
https://archive.ics.uci.edu/ml/datasets/bank+marketing

Key characteristics of the dataset:
Number of records: 41,188 client interactions.
Number of features: 20 input variables.

Target variable:
y — whether the client subscribed to a term deposit (yes/no)

The dataset includes several categories of features:
- Demographic features
- Age
- Job
- Marital status
- Education
- Financial indicators
- Housing loan
- Personal loan
- Credit default
- Campaign information
- Number of contacts performed
- Outcome of previous marketing campaign
- Contact details
- Month
- Day of week
- Communication type

The dataset is slightly imbalanced, as only a small percentage of customers subscribed to the product.

4. Methodology

The project follows a structured machine learning pipeline.

4.1 Data Preprocessing

The preprocessing stage includes:
- cleaning the dataset;
- handling missing values;
- encoding categorical variables;
- preparing features for modelling;
- splitting the dataset into training and testing sets.

These steps ensure the dataset is suitable for machine learning algorithms.

4.2 Exploratory Data Analysis (EDA)

Exploratory data analysis was conducted to understand the structure of the dataset and identify important patterns.

Key tasks included:
- examining feature distributions;
- analyzing correlations between variables;
- identifying potential outliers;
- evaluating class imbalance.

EDA helped guide feature selection and modelling decisions.

4.3 Machine Learning Models

Several classification models were considered to predict the likelihood of a customer subscribing to a term deposit.

The modelling process included:
- implementing baseline models;
- training machine learning algorithms;
- comparing model performance;
- selecting the best performing model.

Possible models include:
- Logistic Regression
- Decision Trees
- Random Forest
- Gradient Boosting methods

4.4 Model Evaluation

Model performance was evaluated using several metrics appropriate for classification tasks.

These metrics include:
Accuracy – overall correctness of predictions
Precision – proportion of positive predictions that were correct
Recall – proportion of actual positive cases identified
F1 Score – balance between precision and recall
ROC-AUC – ability of the model to distinguish between classes

Cross-validation techniques were used to ensure the model generalizes well to unseen data.

5. Key Findings

The machine learning models were able to identify patterns associated with higher subscription likelihood.

Key insights include:
- certain demographic factors influence subscription probability;
- previous marketing campaign outcomes are strong predictors;
- the number of contacts during campaigns impacts conversion rates.

These insights can help financial institutions design more effective marketing strategies.

6. Technology Stack

The project uses the following technologies:
Programming language: Python
Data analysis: Pandas, NumPy
Machine learning: Scikit-learn
Experimentation and reproducibility: ML pipeline scripts, Docker containers
Version control: Git and GitHub

These tools ensure the project is reproducible, scalable, and maintainable.

7. Team Collaboration

The project was completed by a team of four members, with responsibilities distributed across different stages of the machine learning workflow.

Roles included:
- data preparation and preprocessing;
- exploratory data analysis and feature engineering;
- machine learning modelling and evaluation;
- pipeline development and documentation.

This collaborative structure ensured efficient project development and accountability.

8. Limitations

Despite promising results, the project has several limitations:
- class imbalance in the dataset;
- limited contextual information about customers;
- model performance may vary across different banking markets;

Future improvements may include more advanced modelling techniques and additional data sources.

9. Future Work

Future improvements to the project could include:
- hyperparameter tuning for improved model performance;
- handling class imbalance using advanced techniques such as SMOTE;
- deploying the model as a real-time prediction service;
- integrating the model into banking CRM systems.

These enhancements would allow the solution to be used in real-world financial applications.

10. Conclusion

This project demonstrates how machine learning can be applied to improve marketing efficiency in the banking sector. By predicting which customers are most likely to subscribe to a term deposit, financial institutions can optimize their marketing campaigns and make more data-driven decisions.

The project highlights the value of combining data analysis, machine learning, and reproducible engineering practices to solve real-world business problems.