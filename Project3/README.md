# Repository for delivery of project 3 FYS-STK4155
### Author: Jose Luis Barreiro Tome


The repository is composed of:

Folders:

## Code:

01_hyperparameter_tuning.py: source code containing the test for tuning hyperparameters in all classifiers

02_UrbanTrees_clf.py: classification for obtaining overall accuracy, 10 fold cross-validation accuracy, several figures (confusion matrix, ROC curve, cumulative gain curve, plot of a decision tree, feature importance in RF), F1/precision/recall scores and a fast test with many classifiers (LazyClassifier).

Bias_Var_Lasso.py	Bias-Variance tradeoff for Lasso
Bias_Var_MLP.py	Bias-Variance tradeoff for MLP
Bias_Var_OLS.py	Bias-Variance tradeoff for OLS
Bias_Var_RF.py		Bias-Variance tradeoff for RF
Bias_Var_Ridge.py	Bias-Variance tradeoff for Ridge

## References: 
Some pdf with references used in the report

## input_data: 

Urban_Forestry_Street_Trees.csv:	Original tree dataset
input_trees.csv: 				Final dataset used for classification, with LiDAR predictors
Urban Forestry Street Trees.html: 	Urban Forestry Street Trees metadata

## LiDAR

batchfiles: 		lastools batch scripts for LiDAR pre-processing
predictors: 		LiDAR-based features
QC: 				info about the LiDAR dataset
Lidar_metadata.txt	ALS metadata

## Results

All plots used in the report + more, screenshots & bias-variance plots

## Report

Report for Project 3 and for the exercise Bias-Variance tradeoff
 




