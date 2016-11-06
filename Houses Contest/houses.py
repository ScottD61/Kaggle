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
from pandas.tools.plotting import scatter_matrix


#Import train and test datasets
house_train = pd.read_csv('train.csv')
house_test = pd.read_csv('test.csv')


#Data exploration
#Name of variables
house_train.columns
#Dimension of dataset
house_train.shape
#Column datatypes
house_train.dtypes
#Number of numeric variables
house_train.select_dtypes(include = ['int']).columns.values
#Number of categorical variables
house_train.select_dtypes(include = ['object']).columns.values
#Number of NaN values in each column
house_train.isnull().sum()

#Summary statistics
#Numeric variables
house_train.describe()
#Categorical variables
categorical = house_train.dtypes[house_train.dtypes == "object"].index
house_train[categorical].describe()

#Drop ID variable
house_new = house_train.drop('Id', 1)




#Visualizations

#Step 2
#Univariate data exploration
#Histogram of dependent variable
house_new['SalePrice'].hist(bins = 50)

#Scatterplot matrix of numeric variables
scatter_matrix(house_new, alpha=0.2, figsize=(6, 6))

#Barcharts of all categorical variables
#Types of dwellings
a = house_new['MSSubClass'].value_counts()
a.plot(kind = 'bar')

#Zoning classification
b = house_new['MSZoning'].value_counts()
b.plot(kind = 'bar')







#Step 3
#Bi-variate data exploration - there needs to be a reason why
#Correlations
#Diagonal correlation matrix
seaborn.corrplot(house_new, annot = False, diag_names = False)

#Get top 10 correlations with the price variable

#Get top correlations with variables to each other 

#Do saleprice with OverallQual
#Saleprice with YearBuilt
#Saleprprice with GrLivArea
#

#Step 4
#Data imputation
#Impute numerical missing values with average
house_na = house_new.fillna(house_new.mean())
#Impute categorical data
house = house_na.apply(lambda x:x.fillna(x.value_counts().index[0]))


#Step 5 outlier detection

#Step 6 variable transformation
#Convert categorical datatype to dummy variables in dataframe
house_final = pd.get_dummies(house)


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

