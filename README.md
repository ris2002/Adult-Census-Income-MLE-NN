# Adult-Census-Income-MLE-NN
This is my 9th project. This project will be done in a similar style to my 7th project, except it will not be end-to-end like it. I am only going to involve NN and ensemble techniques, as I am learning those techniques to apply them to regression tasks.
## Problem Analysis
* Type-Classification
* Business Goal - The business goal is to identify individuals who are likely to earn more than $50,000 annually using demographic and employment-related information.
* ML/NN Objective - The machine learning (or neural network) objective is to train a model that accurately predicts the probability that an individual’s income exceeds $50K, while handling class imbalance and mixed data types.
## Dataset Overview
The dataset was exported from Kaggle (https://www.kaggle.com/datasets/uciml/adult-census-income). It consists of 32000 rows and 14 columns. In that 14 col,6 columns are numerical columns and the rest are categorical cols, including the output feature.
## Pre-Processing Strategy and EDA
* Found around 2000 rows with missing data. I decided to delete all those rows as I thought 32k is not a small number and losing 2k rows would not make a difference, but increase the prediction than  replacing the NaN values with the mode categories.
* In the categorical data columns, I have found out that in each column, one label repeats 75% times than the others. To solve this, I have decided to implement Smote Analysis to balence the labels for correct prediction.
* I have decided to remove 'capital.gain' and 'capital.loss' columns. This is because at least 27-28k columns are 0, and this problem wouldn't be solved using SMOTE analysis. This was my feeling
*          age	fnlwgt	education.num	hours.per.week
count	30162.000000	3.016200e+04	30162.000000	30162.000000
mean	38.437902	1.897938e+05	10.121312	40.931238
std	13.134665	1.056530e+05	2.549995	11.979984
min	17.000000	1.376900e+04	1.000000	1.000000
25%	28.000000	1.176272e+05	9.000000	40.000000
50%	37.000000	1.784250e+05	10.000000	40.000000
75%	47.000000	2.376285e+05	13.000000	45.000000
max	90.000000	1.484705e+06	16.000000	99.000000

## Selected Model
* Custom NN
* Random Forest 
* ADABOOST
* Graadient Boosting
* XG Boost
## Model Reasoning and hypothesis

## Model Hyperparameters tuning
