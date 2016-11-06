# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Import modules
import seaborn 
import pandas as pd
import numpy as np
import sklearn.linear_model as ln
from sklearn.model_selection import cross_val_score


#Import train and test datasets
house_train = pd.read_csv('train.csv')
house_test = pd.read_csv('test.csv')


#Data exploration
#Dimension of dataset
house_train.shape
#Column datatypes
house_train.dtypes
#Number of NaN values in each column
house_train.notnull().sum()

#Summary statistics
#Numeric variables
house_train.describe()
#Categorical variables
categorical = house_train.dtypes[house_train.dtypes == "object"].index
house_train[categorical].describe()

#Data preprocessing
#Drop ID variable
house_new = house_train.drop('Id', 1)
#Data imputation
#Impute numerical missing values with average
house_na = house_new.fillna(house_new.mean())
#Impute categorical data
house = house_na.apply(lambda x:x.fillna(x.value_counts().index[0]))



#check for NaN values
house.notnull().sum()
#Convert categorical datatype to dummy variables in dataframe
house_final = pd.get_dummies(house)


#Visualizations
#Diagonal correlation matrix
seaborn.corrplot(house, annot = False, diag_names = False)

#Histogram of dependent variable
house['SalePrice'].hist(bins = 50)



#Model building
#Separate and x and y variables
#training data
X_train = house_final.drop('SalePrice', axis = 1)
Y_train = house_final['SalePrice']
#test data - WRONG THERES NO SALES PRICE IN TEST SET 
#X_test = house_test.drop('SalePrice', axis = 1)
#Y_test = house_test['SalePrice']

#Model 1
#All variables 

#Other OLS method with scikitlearn
#Create linear regression object
regr = ln.LinearRegression()
#Train model with training sets
regr.fit(X_train, Y_train)

#Evaluation
#Cross validation score for MSE 10 folds
MSE_Scores = cross_val_score(regr, X_train, Y_train, scoring = 'neg_mean_squared_error', cv = 10)
#Take average of all cross validation folds
np.mean(MSE_Scores)
#Cross validation score for MAE 10 fold
MAE_Scores = cross_val_score(regr, X_train, Y_train, scoring = 'mean_absolute_error', cv = 10)
#Take average of all cross validation folds
np.mean(MAE_Scores)
#Cross validation score for R2 10 folds
R2_Scores = cross_val_score(regr, X_train, Y_train, scoring = 'r2', cv = 10)
#Take average of all cross validation folds
np.mean(R2_Scores)


#Model scores low - requires feature engineering

