# Car Insurance Claims Prediction

## Overview

This machine learning project focuses on predicting car insurance claims using decision trees and various ensemble methods. The goal is to determine whether a policyholder will file a claim in the next 6 months based on a range of features, including policy tenure, driver details, car specifics, and more. The project emphasizes the interpretability of decision trees in the insurance domain.

## Dataset

The dataset used in this project is sourced from Kaggle and is available in the "archive" folder within this Git repository. It contains valuable information about policyholders and their historical insurance claims, making it a crucial component of the predictive modeling process.

## Decision Trees and Interpretability

Decision trees were chosen as the primary modeling technique due to their high level of interpretability. The project explores the use of decision trees with different depths to balance accuracy and interpretability, ensuring that the models can be easily understood and explained.

## Ensemble Methods

In addition to decision trees, this project also explores ensemble methods for improving predictive performance. The following ensemble methods have been applied:

- Random Forest: An ensemble of decision trees to enhance predictive accuracy and reduce overfitting.

- AdaBoost: An adaptive boosting algorithm that combines multiple weak learners to create a strong predictive model.

- Bagging Classifiers: Bagging, or Bootstrap Aggregating, is used to create an ensemble of decision trees to improve model stability.

## Code and Notebooks

- `notebook.ipynb`: This Jupyter notebook contains the entire workflow of the project, from data preprocessing and exploratory data analysis to model training, evaluation, and interpretation.

- `source.gv`: This file contains the decision tree graph generated during the modeling process, showcasing the interpretability of the model. It provides a visual representation of how the decision tree makes predictions.

