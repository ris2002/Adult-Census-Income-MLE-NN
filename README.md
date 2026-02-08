# Adult-Census-Income-MLE-NN
This is my 9th project. This project will be done in a similar style to my 7th project, except it will not be end-to-end like it. I am only going to involve NN and ensemble techniques, as I am learning those techniques to apply them to regression tasks.
## Problem Analysis
* Type-Classification
* Business Goal - The business goal is to identify individuals who are likely to earn more than $50,000 annually using demographic and employment-related information.
* ML/NN Objective - The machine learning (or neural network) objective is to train a model that accurately predicts the probability that an individual’s income exceeds $50K, while handling class imbalance and mixed data types.
## Dataset Overview
The dataset was exported from Kaggle (https://www.kaggle.com/datasets/uciml/adult-census-income). It consists of 32000 rows and 14 columns. In that 14 col,6 columns are numerical columns and the rest are categorical cols, including the output feature.
## Pre-Processing Strategy and EDA
* Approximately 2,000 rows contained missing values, accounting for roughly 6% of the dataset. Since the missingness was primarily in categorical variables and not systematic, I chose to remove these rows rather than perform mode imputation, which could further amplify majority category bias. Given the remaining dataset size (~30k rows), this trade-off was acceptable.
* The target variable (income >50K) is significantly imbalanced. To address this during model training, I applied SMOTENC after the train-test split to synthetically oversample the minority class while preserving categorical feature integrity. this doesnt apply to target var it applies to all categorical data
* The capital.gain and capital.loss features were removed due to extreme zero-inflation (over 90% zeros). Since this project focuses on experimenting with neural networks—which are sensitive to sparsity and outliers—I opted to exclude these features to reduce noise and stabilize training. Tree-based models were still evaluated for comparison.
* Basic EDA of  numerical data.
| Statistic | age        | fnlwgt        | education.num | hours.per.week |
|-----------|------------|---------------|---------------|----------------|
| count     | 30162.0000 | 30162.0000    | 30162.0000    | 30162.0000     |
| mean      | 38.4379    | 189793.8000   | 10.1213       | 40.9312        |
| std       | 13.1347    | 105653.0000   | 2.5500        | 11.9800        |
| min       | 17.0000    | 13769.0000    | 1.0000        | 1.0000         |
| 25%       | 28.0000    | 117627.2000   | 9.0000        | 40.0000        |
| 50%       | 37.0000    | 178425.0000   | 10.0000       | 40.0000        |
| 75%       | 47.0000    | 237628.5000   | 13.0000       | 45.0000        |
| max       | 90.0000    | 1484705.0000  | 16.0000       | 99.0000        |


## Selected Model
* Custom NN
* Random Forest 
* ADABOOST
* Graadient Boosting
* XG Boost
## Model Reasoning and hypothesis

## Model Hyperparameters tuning
