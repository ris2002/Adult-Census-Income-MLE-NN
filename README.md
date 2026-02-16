# Adult-Census-Income-MLE-NN
This is my 9th project. This project will be done in a similar style to my 7th project (https://github.com/ris2002/Diabetes_Prediction-MLE), except it will not be end-to-end like it. I am only going to involve NN and ensemble techniques, as I am learning those techniques to apply them to regression tasks.
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
* PCA is not required as the dataset is low diamention and mixed data.
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
### Train-Test Split and Oversampling Strategy

* The dataset was first cleaned (missing values removed, irrelevant features dropped) and categorical features were integer-encoded to prepare for SMOTENC.
* The data was split into training and test sets before oversampling.
* Reason: The test set must remain untouched to provide a realistic, unbiased evaluation of model performance. Splitting first prevents leakage of synthetic samples into the test set.
* SMOTENC was applied only to the training set to synthetically oversample the minority class while correctly handling categorical features.
* Reason: Oversampling before splitting would allow synthetic samples to “contaminate” the test set, inflating accuracy and giving a misleading evaluation.
* After oversampling, numeric features were scaled (RobustScaler for skewed features, StandardScaler for near-ideal features), and categorical features were one-hot encoded or prepared for embeddings as appropriate.
* Reason: Scaling and encoding are applied after SMOTENC so that the model sees properly formatted data without breaking the synthetic oversampled structure.
* This pipeline ensures that the model sees a balanced and properly formatted training set, while the test set remains a realistic, unseen representation of the original data.
## Post EDA, Pre-Training Analysis
* Size of x_train is (40816, 75)
* Size of y_train is (40816,)



## Selected Model
* Custom NN
* Random Forest 
* ADABOOST
* Gradient Boosting
* XG Boost
## Model Reasoning and hypothesis
### Neural Network Algorithm
* Neural Networks are nothing but mathematics to predict outputs on a dataset. My intuition is that using forward and backward propagations, it tries to train its weights and biases in such a way that it maps similar to the  data-set in a n-diamentional graph.
* The working of this technique is as follows-
  * Forward Pass (only 1 neuron,1 epoch)
     * Firstly, it assigns each neuron random weights and biases, here weights represent how important a particular input to a neuron is, and bias is like a personality trait.
     * It computes equation z=sum(w*x)+b and passes this function through an activation function to introduce some non-linearity to it , then it passes it through an output activation function to compute and then through a loss function to compute its losss
  * Backward pass
     * After it computes the loss, it finds the gradients wrt to the weights and biases and passes them back through  the layer. Using the optimiser, it computes the weights and biases from the final gradients and again repeats the  forward pass
* Disadvantages- Requires a lot of experimentation and research to land the correct model
* Reason- I am currently learning NN, and I want to compare how it compares with other ML models wrt to this classification dataset. 
##### Null Hypothesis
* Null-Hypothesis- NN  performs similarly or better than the other ML models and gives equal F1 scores and an overall minimum of 70% F1 score on 2 classes on the Adult Census dataset.
------------------------------------------------------------------------------------------------------------------------
### Boosting Algorithms
* Boosting algorithms are one kind of ensemble technique wherein a strong model is made up of several weak models instead of building a strong model from the get-go. This results in the model learning non-linear data and can be applied to both Classification and Regression Problems. Here, the weak model refers to a weak Decision Tree or  Logistic Regression model.
* All boosting algorithms are sequential in nature.
#### ADABoost
Model Overview-
* It uses only  DT, specifically shallow DT known as stump, as its weak modelor Base Learner.
* The working of this model is as follows-
     * In a dataset, it first assigns equal but uniform weights to each record.
     * If there are n features, it creates n stumps and selects the stump with less entropy (checks if the split is pure or near pure) and trains the data on it.
     * After training data on that stump, it finds the total errors and performance of that stump.
     * Why find the total errors and performance of that stump? That is because, in all boosting techniques, we need to pass incorrect records or the margin of error to the next base learner so that the next model puts more emphasis on resolving that.
     * In Ada-Boost, we increase the weights of those incorrectly classified records so that the next base learner emphasises  learning from the mistakes of the previous model to predict correctly.
* One disadvantage is that Adaboost is slower to train and sensitive to noise.
* Reason- This is a simple mechanism of a boosting algorithm, and I want to take it as a baseline reference in terms of accuracy and speed of prediction. 
##### Null Hypothesis
* Null-Hypothesis- ADABoost gives an F1 score minimum of 70% and performs equally on both classes on the Adult Census dataset.

#### Gradient Boost
Model Overview-
* It uses both shallow DT(stumps) or Logistic Regression, as its weak modelor Base Learner.
* The working of this model is as follows-
     * The working is similar to NN where each base learner passes gradients to the next one to learn from the error of the previous one's mistakes and predict properly or more closly when it comes to the last base learner. Here, there is no updation of weights like in NN
     * Firstly, the 1st base learner predicts the output y^, it calculates the residuals using the formula r1=y-y^ and it passes the the features,r1 and y^ as inputs.
     * In the next Base Learner, it predicts a new set of predictions, y_pred,1 with residuals r2. Now it calculates the new inputs to be given to the next Base Learner using the formula y1^=y^+(Learning_Rate)*r2 and it repeats further. 
* Some disadvantages is that Gradient Boost is highly sensitive to overfitting, class imbalance, outliers and complex hyperparameter tuning.
* Reason- This is a much more complex mechanism compared to ADABoost and is little similar to the passing of gradients in NN during backpropagation, for this data, I have done EDA to remove the known outliers and handle class imbalence and I wnt check what happens when this kind of ML algorithm with processes similar to NN iss applied to my dataset    .
##### Null Hypothesis
* Null-Hypothesis- Gradient Boosting gives an F1 score minimum of 70% and performs equally on both classes on the Adult Census dataset.

#### XGBoost
*XGBoost (eXtreme Gradient Boosting) is an optimised and regularised version of the standard Gradient Boosting Machine (GBM).
* Reason- This is an advanced version of Gradient Boosting, and I want to chk how it compares to the Gradient Boosting Mechanism. 
##### Null Hypothesis
* Null-Hypothesis- XGBoosting gives an F1 score minimum of 70% and performs equally on both classes on the Adult Census dataset.

------------------------------------------------------------------------------------------------------------------------
### Bagging Algorithms
* This technique is also known as bootstrap aggregation. What happens here is we have multiple  base learners(high-variance models) which are given to train on the randomly subsetted data first, and they together come to a decision to predict results. This is done to reduce overfitting to improve the stability and accuracy of the models.
#### Random Forest

Model Overview-
* It uses only  DT, as its  Base Learner.
* DT is one of the highly variant ML models. The main goal of the Random Forest/Bagging Technique is to convert this high-variance DT into low variance
* The working of this model is as follows-
     * Firstly bootstrapping process takes place, basically row sampling, feature sampling with replacement is given to each DT to train on.
     * By doing this, each DT will become an expert wrt to each subset of the dataset.
     * Internally, it produces an output, and the majority prediction is given as the final output (aggregation). By the voting process, we convert the highly variant DT into low variance.
* Some disadvantages: High Resource Consumption (Memory & Storage), Slow Prediction Speed (Inference).
* Reason- This is an even simpler mechanism compared to the boosting algorithms mentioned above. My intuition is that sometimes simplicity triumphs complexity.
##### Null Hypothesis
* Null-Hypothesis- Random Forest gives an F1 score minimum of 70% and performs equally on both classes on the Adult Census dataset.
## Evaluation Metrics Strategy 
* In a confusion matrix, we have TP,FN,FP,TN based on 2 classes, True and False.
* True Positive means in the true class, the labels were correctly detected as True.
* False Negative means in the true class, the labels were wrongly detected as False.
* True Negatives means in the false class, the labels were correctly detected as False.
* False Positives means in the false class, the labels were wrongly detected as True.
* Recall=TP/TP+FN, recall answers the question - "Out of all real positives, how many were detected correctly".
* Precision=TP/TP+FP means that out of the values detected as true, how many were correctly flagged.
* Higher Precision means the model can distinguish the true values correctly.
* Higher Recall
* F1 score is nothing but the harmonic mean of precision and recall.
* The harmonic mean: Punishes imbalance, Becomes low if either precision or recall is low ,Forces both to be reasonably high
* Hence, I will be evaluating the F1 scores only

## Model Hyperparameters tuning
### Neural Networks
* Test-1
   * epoch-1000, loss_func-CrossEntropyLoss, Optimizerr-SGD,LR-0.01,Acivation Func-Relu,Neurrons-128+2,2 layers
   * Reason for selecting these values and a shallow NN and no dropout or Batch Normalisation is just to get a baseline to improve accuracy further.
   * I have taken large neurons at first to serve as a neutral starting point and I felt 32, or 64 would train much as I have around 40816 records after SMOTE.
   * Class 0
        * Precision: 0.92 → When the model predicts 0, it’s correct 92% of the time (very good).
        * Recall: 0.78 → It correctly finds 78% of all actual 0 cases (decent).
        * F1-score: 0.85 → Strong overall performance for this class.
   * Class 1
        * Precision: 0.56 → When the model predicts 1, it’s correct only 56% of the time (moderate/low).
        * Recall: 0.81 → It catches 81% of actual 1 cases (good).
        * F1-score: 0.66 → Moderate performance.
   *  I have run the same model, but with a threshold of 0.7 to increase the precision of class 1, but I have ended up with similar results.
   *  With a total f1 score of 75%, it is a good benchmark but if it has come to individual classes it iis not good.
   *  The reason for the good total F1 score might be attributed to the usage of high neons in the hidden layer.
   *  The observation from the loss curve and the accuracy curve is it is converging at around 200 epochs, which implies that model has stopped learning the patterns so fastlty, maybe with the same epoch little slower LR,optimer ad batch size, we can make the model learn patterns a little more slowly so that it can give a good F1 score for class 1.
   *  
* Test-2
   * epoch-1000, loss_func-CrossEntropyLoss, Optimizerr-SGD,LR-0.01,Acivation Func-Relu,Neurrons-128+2,2 layers (same as 1st test) included batch size=200, threshold=0.7
   * Reason for not changing the hyperparameters is that I wanted to test with a batch size.
  
   * Class 0
        * Precision: 0.80
        * Recall: 0.98
        * F1-score: 0.88
   * Class 1
        * Precision: 0.83
        * Recall: 0.31
        * F1-score: 0.45
   *  I have run the same model, but with a threshold of 0.7 to increase the precision of class 1, but I have ended up with similar results.
   *  With a total f1 score of 65.2%, we can see it has decreased, but if you notice class wise there there has been some improvement in the department of precision in class 1 annd recall in class 0.
   *   There has been little decrease in the value of precision in class 0 and a significant decrease in  recall in class 1.  
   *  The reason for the good precision in class 1 and recall in class 0 may be attributed to the batch size  and threshold. Threshold also played a role in crashing the recall value in class 1.
   *  In Test-2, mini-batching (batch size = 200) caused the weights to be updated more frequently with slightly noisier gradients. This shifted the model’s output probabilities, making it more conservative predicting class 1. As a result, class 1 precision increased, but recall dropped significantly.
   *  Due to the  threshold at 0.7, what is happening is that the model is selecting probabilities greater than 0.7 this reduces the recall for class     1. The recall for class 0 is 0.98 that implies that the model is 
   * <img width="1366" height="1144" alt="image" src="https://github.com/user-attachments/assets/8c86a81b-94df-4130-9f3e-44012d2bf22e" />

   *  <img width="1342" height="1018" alt="image" src="https://github.com/user-attachments/assets/e295f5cd-51da-4893-9601-23ccf400e87f" />
   * From the above pictures, we can see that the accuracy and the loss curves are convering quicklin in the epoch range of 0-100. My intuition is model has somewhat stopped leaning after the curve has stabilised and if  the leranig rate is less and with the adding og dropout and batch normalisation dropping the batch size  there is a posibility of accepting null hypothesis.
* Test-3
   * epoch-1000, loss_func-CrossEntropyLoss, Optimizerr-SGD,Acivation Func-Relu,Neurrons-128+2,2 layers,batch size=200, threshold=0.7 (same as 1st and 2nd test) included dropout layer(0.25%)and changed LR to LR-0.001
   * Reason for not changing the hyperparameters is that I wanted to test with a dropout layer and reduced LR.
  
   * Class 0
        * Precision: 0.93
        * Recall: 0.79
        * F1-score: 0.85  
   * Class 1
        * Precision: 0.57
        * Recall: 0.83
        * F1-score: 0.68
   *  I have run the same model, but with a dropout layer and reduced LR.
   *  With a total f1 score of 76.5%, we can see it has increased overall, the class wise metrics are similar to the first test.
   *   We can see in individual class results, it matches the the test-1 This might be because of the usage of dropout layer which drops the neurons randomly duringg each epoch of traininng so that each neuron would update itss weights indepently and no neiron woruld be dependednt on the other.
   * <img width="1316" height="1068" alt="image" src="https://github.com/user-attachments/assets/27512b13-79b8-4e75-910d-5f713d9ca1bd" />
   *  <img width="1326" height="1062" alt="image" src="https://github.com/user-attachments/assets/ac35568a-4253-4947-a04f-d36db40f819a" />
   * The accuraacy and loss graphs are relativly the same as the tests before. And the main improvement is to test with Batch-Normalisation Layer and leaving the rest of params the same.
   * This test is more efficent than the test-1 because in test-1 thhreshold-0.5, no mini batching and no dropuot layer
