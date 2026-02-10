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
* The capital.gain and capital.loss features were identified as zero-inflated, with a large majority of observations containing zero values. While initially considered for removal due to sparsity concerns, these features were retained after NN-oriented analysis. Rather than dropping them, the following transformations are planned: Binary indicator features capturing the presence of non-zero values, Log-transformation (log1p) to compress extreme magnitudes, Standard scaling to ensure stable gradient behaviour. This approach allows the neural network to learn rare but highly informative signals without introducing bias.
* column fnlwgt is analysed as a useless feature, as this is for a group of individuals, not for a single individual, so it was removed.
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

### Feature Analysis 
* Age Observation: The box plot indicates high variability (large IQR) and is slightly right-skewed, with a longer upper whisker and several outliers representing senior citizens.Decision: No non-linear transformations (like Log) will be applied.Reasoning: Age is a critical feature with a complex relationship to income. Maintaining its original distribution prevents disrupting the predictive signals needed by the Neural Network. Standard Scaling will be used later to handle magnitude.
*  Capital Gain & Capital Loss Observation: These features are extremely right-skewed and heavily "zero-inflated," with the vast majority of data points concentrated at 0 and a long tail of extreme outliers.Decision: Apply Log Transformation (\(\log (x+1)\)) to compress the range and reduce skewness. Additionally, create two new binary features representing the presence of a gain or loss (1 if \(>0\), else 0).Reasoning: This "dual-feature" approach allows the NN to distinguish between the existence of investment income and its magnitude, while protecting the model from unstable gradients caused by extreme values. 
*  Hours per Week Observation: The distribution shows low skewness, with the median centered near 40 hours. However, there are numerous outliers on both the high and low ends of the spectrum.Decision: No Log Transformation is required due to low skewness. However, outlier handling is mandatory.Reasoning: The IQR method will be used to identify and manage these outliers. This ensures that extreme work schedules (e.g., 99 hours/week) do not disproportionately bias the model's weight updates.
*   Education Num (Referencing the Image) Observation: This feature presents a near-ideal box plot. While there is a slight left skew and a few low-end outliers, the distribution is relatively stable.Decision: No changes or transformations will be applied to this data.Reasoning: Since education.num is an ordinal representation of education levels with a narrow range (1–16), it is already well-suited for the model. The few low-education outliers are valid and provide important predictive value.
*   While feature correlation analysis was explored, it was not used for feature elimination. Both neural networks and ensemble-based classifiers are capable of handling correlated inputs and learning nonlinear relationships. Therefore, emphasis was placed on distribution analysis, encoding strategy, and handling skewed and zero-inflated features.

### Exploring Categorical Data
* In the given categorical dataset,['workclass', 'education',' marital.status', 'occupation',' relationship', 'race', 'sex','native.country'], only 'education' can be considered as ordinal data, the rest all can be considered as nominal data. Although occupation does not have a universally agreed-upon ordinal structure, it was intentionally treated as an ordinal feature for experimental purposes. 
#### Dataset Distribution Summary
##### Employment & Workclass
The vast majority of the workforce resides in the private sector.
Private: 22,286
Self-employed (Not Inc): 2,499
Local Government: 2,067
State Government: 1,279
Self-employed (Inc): 1,074
Federal Government: 943
Without Pay: 14
##### Education Levels
Education is highly concentrated at the high school and undergraduate levels.
HS-grad: 9,840
Some-college: 6,678
Bachelors: 5,044
Masters: 1,627
Assoc-voc: 1,307
Doctorate: 375
Preschool: 45
##### Marital Status
Married (Civilian Spouse): 14,065
Never-married: 9,726
Divorced: 4,214
Separated/Widowed: 1,766
Other: 391
##### Occupation
The top occupational categories are nearly tied in frequency:
Prof-specialty: 4,038
Craft-repair: 4,030
Exec-managerial: 3,992
Adm-clerical: 3,721
Sales: 3,584
##### Demographics
Race: Predominantly White (25,933), followed by Black (2,817) and Asian-Pac-Islander (895).
Sex: Male (20,380) and Female (9,782).
We can see in each feature , one categorical label dominates about more than 50%. This might be a problem when used in a basic models like Linear or Logistic regressions, but will not be a problem in ensemble classification techniques and NN only iif encoded properly 
### Scaling and Encoding
* For scaling numerical features, I will use Robust Scalar to scale all numerical features except the 'Education Num' feature. Usage of Robust Scalr is used because most of the numerical features except 'Education Num' feature, have outliers.
*  For 'Education Num', I will use Standard Scalar as the feature is near perfect.
*  For Categorical data, I will use OHE to encode nominal features and use OrdinalEncoders to encode ordinal feartures
*  For the target, I will use Label encoding



## Selected Model
* Custom NN
* Random Forest 
* ADABOOST
* Gradient Boosting
* XG Boost
## Model Reasoning and hypothesis

## Model Hyperparameters tuning
