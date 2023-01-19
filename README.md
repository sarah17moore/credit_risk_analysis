# Credit Risk: Supervised Machine Learning
A classwork example in which models to predict credit risk are assessed using machine learning techniques like imbalanced-learn, SMOTE, and SMOTEEN. 

---
# Overview
"LendingClub: a peer-to-peer lending services company" is requesting that we analyze their credit-risk dataset using 4 different machine learning models. These models will be used to make predictions about credit-risk based on patterns present within the dataset. Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. This means we would need to apply more weight towards good loans, predicting that good loans should always outnumber the amount of risky loans in a model. 

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
### RandomOverSampler
#### Balanced Accuracy Score
![balanced accuracy score for ROS](Resources/balanced_accuracy_ROS.png)

#### Confusion Matrix
![confusion matrix for ROS](Resources/cm_ROS.png)

#### Imbalanced Classification Report
![imbalanced report for ROS](Resources/report_ROS.png)

### SMOTE
#### Balanced Accuracy Score
![balanced accuracy score for SMOTE](Resources/balanced_accuracy_SMOTE.png)

#### Confusion Matrix
![confusion matrix for SMOTE](Resources/cm_SMOTE.png)

#### Imbalanced Classification Report
![imbalanced report for SMOTE](Resources/report_SMOTE.png)

## Undersampling
### ClusterCentroids
#### Balanced Accuracy Score
![balanced accuracy score for CC](Resources/balanced_accuracy_CC.png)

#### Confusion Matrix
![confusion matrix for CC](Resources/cm_CC.png)

#### Imbalanced Classification Report
![imbalanced report for CC](Resources/report_CC.png)

## Combination of Over and Undersampling
### SMOTEENN
#### Balanced Accuracy Score
![balanced accuracy score for SMOTEENN](Resources/balanced_accuracy_SMOTEENN.png)

#### Confusion Matrix
![confusion matrix for SMOTEENN](Resources/cm_SMOTEENN.png)

#### Imbalanced Classification Report
![imbalanced report for SMOTEENN](Resources/report_SMOTEEN.png)

## Ensemble classifiers
### BalancedRandomForestClassifier
#### Balanced Accuracy Score
![balanced accuracy score for brfc](Resources/balanced_accuracy_brfc.png)

#### Confusion Matrix
![confusion matrix for brfc](Resources/cm_brfc.png)

#### Imbalanced Classification Report
![imbalanced report for brfc](Resources/report_brfc.png)

### EasyEnsembleClassifier
#### Balanced Accuracy Score
![balanced accuracy score for eec](Resources/balanced_accuracy_eec.png)

#### Confusion Matrix
![confusion matrix for eec](Resources/cm_eec.png)

#### Imbalanced Classification Report
![imbalanced report for eec](Resources/report_eec.png)

# Summary 
This is where the summary will go

---

It is to be noted that machine learning models may not be as robust as real-life instances due to only taking a small set of data compared to the large size of the original dataset. Close to 99% of the applications in the original dataset (before the sample or training was taken) were classified as "low risk". This disparity between the actual dataset and real-life instances should require further analysis.
