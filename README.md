# credit_risk_classification


# Purpose of this Report

For this project I took the historical data for the lending activity from a lending company that operates as peer-to-peer. This dataset includes seven columns that describe the loan, the borrowing company, and the lending company. Then the final column indicates the loan status as 0 (healthy loan) or 1 (high-risk loan). All of this information was used in order to create two supervised machine learning model with the goal of predicting future high-risk loans. The report below includes the results of the machine learning model and an analysis of the results. 

# Overview of the Analysis

The purpose of this analysis is to help the loan company determine whether or not the machine learning models created in this project will help them predict high-risk loans with enough success that the models should be implemented. 

The eight columns included in this dataset are 'loan_size', 'interest_rate', 'borrower_income', 'debt_to_income', 'num_of_accounts', 'derogatory_marks', 'total_debt', 'loan_status'. In order to make a predictions the columns 'loan_size', 'interest_rate', 'borrower_income', 'debt_to_income', 'num_of_accounts', 'derogatory_marks', and  'total_debt' are used as inputs, where ‘loan_status’ is what the models are trying to predict based on the inputs. 

Some general information about the dataset the models are based on is as follows:
•	There are 77535 total records 
•	75036 records with a loan status of 0 and 2500 with a loan status of 1. 

To create these machine learning models, first I separated the data into x and y values. The x values are columns 'loan_size', 'interest_rate', 'borrower_income', 'debt_to_income', 'num_of_accounts', 'derogatory_marks', and  'total_debt'. The y values for the dataset are from the ‘loan_status’ column. Then the data is separated into training and testing values using the code train_test_split. Next the type of model is chosen, in this case the Logistic Regression model is used for both models. The x_train and y_train values are then fit to the model. Predictions based on the model are then made using the x_test data and then compared to the expected value of y_test using the classification report. The main difference between the two models is that the second model uses the RandomOverSampler module, which resamples the data so that y values are equally balanced between 0 and 1. 

# Results

Machine Learning Model 1:
Overall accuracy: 0.99
For the target of 0 (healthy loan)
•	F1-score: 1.00
•	Precision: 1.00
•	Recall: 0.99
For the target of 1 (high-risk loan)
•	F1-score:  0.88
•	Precision: 0.85
•	Recall: 0.91


Machine Learning Model 2 (with `RandomOverSampler` module):
Overall accuracy: 0.99
 For the target of 0 (healthy loan)
•	F1-score: 1.00
•	Precision: 1.00
•	Recall: 0.99
For the target of 1 (high-risk loan)
•	F1-score:  0.91
•	Precision: 0.84
•	Recall: 0.99



# Summary

The overall accuracy for Model 1 is 0.99. Then the model predicted a healthy loan with an f1-score of 1.00. This would indicate that the accuracy of a healthy loan is 1.0, but since the recall is .99 it is likely that the f1-score is closer to .999999999, that got rounded to 1.00, so an accuracy of .999999 is most likely a better representation of a healthy loan. Then the accuracy of a high-risk loan is .88, which is determined by looking at the f1-score. Additionally, since the recall is .91 and the precision is 0.85, the model has fewer instances where it predicts false negatives compared to false positives for this category. Overall, this model is excellent at predicting healthy loans and pretty good overall, but less accurate, when specifically trying to predict high-risk loans.

When considering how well Model 2 performs it is very important to look at the classification matrix's results for both the healthy loan and high-risk loans and not just the accuracy score. This model has an overall accuracy score of 0.99, however the f1-score is 1.00 (most likely .999999999 which got rounded to 1.00) for predicting a healthy loan and a f1-score of 0.91 for predicting a high-risk loan. Additionally, since the recall is 0.99 and the precision is 0.84, the model has fewer instances where it predicts false negatives compared to false positives for this category. Even though the precision decreased by .01, since the recall went up by 0.08 I consider this model an improvement for predicting high-risk loans and of course still an excellent model for predicting healthy loans. For this reason, I recommend using Model 2 for future predictions since it seems to be the better model of the two for predicting high-risk loans and does a fair job predicting that loan status. 

#Installation

For this project the following imports are required:
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler

#Support

If additional help is needed, I recommend searching Stack Overflow for assistance. 

#Authors and acknowledgment

This project was completed by Kelsey Brantner. I reference the following website to learn more about RandomOverSampler https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html

#License

According to the file’s source site, “Data for this dataset was generated by edX Boot Camps LLC, and is intended for educational purposes only.

#Project status

At this time, I consider the code to be complete. 



