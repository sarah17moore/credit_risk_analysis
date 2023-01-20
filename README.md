# Credit Risk: Supervised Machine Learning
A classwork example in which models to predict credit risk are assessed using machine learning techniques like imbalanced-learn, SMOTE, and SMOTEEN. 

---
# Overview
"LendingClub: a peer-to-peer lending services company" is requesting that we analyze their credit-risk dataset using 4 different machine learning models. These models will be used to make predictions about credit-risk based on patterns present within the dataset. Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. This means we would need to apply more weight towards good loans, predicting that good loans should always outnumber the amount of risky loans in a model. In statistics and data analysis, this is otherwise known as class imbalance - a situation where the existing classes in a dataset aren't equally represented. 

## Purpose
Machine learning models with unbalanced classes will be evaluated to determine which model works best for this given data set. We will test:
* Oversampling using RandomOverSampler and SMOTE (Synthetic Minority Oversampling Technique)
* Undersampling with ClusterCentroids
* Combination Sampling with SMOTEENN (Synthetic Minority Oversampling Technique + Edited NearestNeighbors)
* Ensemble classifiers to reduce bias with BalancedRandomForestClassifier and EasyEnsembleClassifier

<sub>*These models are from Python libraries scikit-learn(sklearn) and imbalanced-learn*</sub>

---

# Results
## Oversampling
If one class has too few instances in a training set, we can choose more observations from that class for training purposes. 

### RandomOverSampler
In random oversampling, instances of the minority class are randomly selected and added to the training set until the minority and majority classes are balanced. 

#### Balanced Accuracy Score
![balanced accuracy score for ROS](Resources/balanced_accuracy_ROS.png)

* The balanced accuracy score is the percentage of predictions that are correct. This is a comparison of the original dataset that was set aside for testing and the models predictions that were correct. RandomOverSampler was only accurate 65% of the time. 

#### Confusion Matrix
![confusion matrix for ROS](Resources/cm_ROS.png)
<sub>The confusion matrix shows the amount of "True Positives" (top left cell), "False Positives" (bottom left cell), "False Negatives" (top right cell), and "True Negatives" (bottom right cell). </sub>

#### Imbalanced Classification Report
![imbalanced report for ROS](Resources/report_ROS.png)

* Precision (pre) is the measure of how reliable a positive classification is. Since precision is so high for low_risk and so low for high_risk, RandomOverSampler is not as precise as can be even though the overall score is quite high. Precision comes to be 0.99 or 99%.
* Recall (rec) is the ability of the classifier to find all the positive samples. Recall (also known as sensitivity) is quite low for high_risk and low_risk, which indicated that there is a considerate amount of false negative values meaning the model was often incorrect. Recall comes to be 0.59 or 59%.
* F1 Score (f1) is a weighted average of the recall and precision values. Since recall and precision were both less than ideal, F1 comes to be 0.73 or 73%.
* Geometric mean and index balanced accuracy will not be evaluated in this project.

### SMOTE
This is an alternate oversampling approach to increase the size of the minority class for training purposes. New (synthetic) observation instances are generated to add values to the minority class. This is also known as interpolation.

#### Balanced Accuracy Score
![balanced accuracy score for SMOTE](Resources/balanced_accuracy_SMOTE.png)

* The SMOTE model was only correct 66% of the time.

#### Confusion Matrix
![confusion matrix for SMOTE](Resources/cm_SMOTE.png)
<sub>The confusion matrix shows the amount of "True Positives" (top left cell), "False Positives" (bottom left cell), "False Negatives" (top right cell), and "True Negatives" (bottom right cell). </sub>

#### Imbalanced Classification Report
![imbalanced report for SMOTE](Resources/report_SMOTE.png)

* Precision is quite high at 99%, although the same high_risk and low_risk inconsistency is present.
* Recall is moderate at 69% with both high and low_risk scoring quite poor.
* F1 is improved compared to ROS at 81%. Since precision is quite high, this is to be expected.
* Geometric mean and index balanced accuracy will not be evaluated in this project.

## Undersampling
If one class has too few instances in a training set, we can randomly select observations from that class until the majority and minority classes are balanced for training purposes. 

### ClusterCentroids
Similar to SMOTE, new (synthetic) observation instances called centroids are generated to add values to the minority class. The majority class is then undersampled down to the size of the minority class. 

#### Balanced Accuracy Score
![balanced accuracy score for CC](Resources/balanced_accuracy_CC.png)

* The ClusterCentroid model was only correct 54% of the time. This is the least accurate we have seen so far.

#### Confusion Matrix
![confusion matrix for CC](Resources/cm_CC.png)
<sub>The confusion matrix shows the amount of "True Positives" (top left cell), "False Positives" (bottom left cell), "False Negatives" (top right cell), and "True Negatives" (bottom right cell). </sub>

#### Imbalanced Classification Report
![imbalanced report for CC](Resources/report_CC.png)

* Precision is quite high at 99%, although the same high_risk and low_risk inconsistency is present.
* Recall is low at 40% with both high_risk performing moderately and low_risk scoring quite poor.
* F1 has decreased compared to ROS and SMOTE at 56%.
* Geometric mean and index balanced accuracy will not be evaluated in this project.

## Combination of Over and Undersampling
Using a combination of over and undersampling is a strategy that can help avoid over-reliance on immediate "neighbors" of a cluster's points as well as assist with providing a clearer picture of the distribution of the data. It takes elements from over and undersampling models and combines them to create a more robust model.

### SMOTEENN
SMOTEENN first oversamples the minority class with SMOTE. Next it cleans the resulting data with an undersampling strategy. If the two nearest neighbors of a datapoint belong to two different classes, the datapoint is dropped. 

#### Balanced Accuracy Score
![balanced accuracy score for SMOTEENN](Resources/balanced_accuracy_SMOTEENN.png)

* The SMOTEENN model was only correct 66% of the time. 

#### Confusion Matrix
![confusion matrix for SMOTEENN](Resources/cm_SMOTEENN.png)
<sub>The confusion matrix shows the amount of "True Positives" (top left cell), "False Positives" (bottom left cell), "False Negatives" (top right cell), and "True Negatives" (bottom right cell). </sub>

#### Imbalanced Classification Report
![imbalanced report for SMOTEENN](Resources/report_SMOTEENN.png)

* Precision is quite high at 99%, although the same high_risk and low_risk inconsistency is present.
* Recall is moderately low at 61% with both high_risk and low_risk performing moderately.
* F1 has decreased compared to ROS and SMOTE at 75%. This is slightly better than the ROS model but is less than the SMOTE model.
* Geometric mean and index balanced accuracy will not be evaluated in this project.

## Ensemble classifiers
Ensemble learning combines multiple models to help improve the accuracy and robustness of models. It also helps to decrease the variance of the model. Therefore, overall performance of the model is increased. 

### BalancedRandomForestClassifier
This model randomly undersamples each bootstrap sample to balance them. 

#### Balanced Accuracy Score
![balanced accuracy score for brfc](Resources/balanced_accuracy_brfc.png)

* The BalancedRandomForestClassifier model was correct 78% of the time. 

#### Confusion Matrix
![confusion matrix for brfc](Resources/cm_brfc.png)
<sub>The confusion matrix shows the amount of "True Positives" (top left cell), "False Positives" (bottom left cell), "False Negatives" (top right cell), and "True Negatives" (bottom right cell). </sub>

#### Imbalanced Classification Report
![imbalanced report for brfc](Resources/report_brfc.png)

* Precision is quite high at 99%, although the same high_risk and low_risk inconsistency is present.
* Recall is moderately high at 87% with both high_risk and low_risk performing moderately well. This is the best recall we have seen this far. 
* F1 has increased compared to all of the other models at 93%. Since the precision and recall scores are moderately high and high, this is to be expected. 
* Geometric mean and index balanced accuracy will not be evaluated in this project.

### EasyEnsembleClassifier
This model combines AdaBoost learners that learn from mistakes of weak classifiers to output stronger classifier predictions. The learners are trained on difference balanced bootstrap samples where balance is acheived by random undersampling. 

#### Balanced Accuracy Score
![balanced accuracy score for eec](Resources/balanced_accuracy_eec.png)

* The EasyEnsembleClassifier model was correct 93% of the time. This is the highest balanced accuracy score out of all of the models.

#### Confusion Matrix
![confusion matrix for eec](Resources/cm_eec.png)
<sub>The confusion matrix shows the amount of "True Positives" (top left cell), "False Positives" (bottom left cell), "False Negatives" (top right cell), and "True Negatives" (bottom right cell). </sub>

#### Imbalanced Classification Report
![imbalanced report for eec](Resources/report_eec.png)

* Precision is quite high at 99%, although the same high_risk and low_risk inconsistency is present. However, high_risk has the highest precision score of all of the models tested. 
* Recall is high at 94% with both high_risk and low_risk performing similarly and very well. This is the the highest recall score of all of the models tested. 
* F1 has increased compared to all of the other models at 97%. This is the the highest F1 score of all of the models tested. 
* Geometric mean and index balanced accuracy will not be evaluated in this project.

# Summary 
To summarize, The EasyEnsembleClassifier model offered the highest scores for the balanced accuracy score, recall, and F1. In addition, precision is very high at 99% and the high_risk classifier has the highest precision score of all of the models tested. The high scores for this model is most likely due to the combination of boosting and random undersampling to maximize learning efficiency. 

I would highly recommend using EasyEnsembleClassifier model for continued tests. It appears the efficiency of the model is unbeatable, and clearly producers more accurate, more precise, and more sensitive results. 

---

It is to be noted that machine learning models may not be as robust as real-life instances due to only taking a small set of data compared to the large size of the original dataset. Close to 99% of the applications in the original dataset (before the sample or training was taken) were classified as "low risk". This disparity between the actual dataset and real-life instances should require further analysis.
This is the total of each classification before the data was sampled or trained:
![classification totals before training](Resources/original_data.png)
