# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Import modules
import seaborn 
import pandas as pd
import sklearn.linear_model as ln

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
house = house_train.drop('Id', 1)

#Impute missing values with average
house.fillna(house.mean())
#check for NaN values
house.notnull().sum()

#Visualizations
#Diagonal correlation matrix
ax = seaborn.corrplot(house, annot = False, diag_names = False)

#List of highest correlation pairs
#c = house.corr().abs()
#s = c.unstack()
#so = s.order(kind = "quicksort", ascending = False)

#print(so[-1470:-81])


#Histogram of dependent variable
house['SalePrice'].hist(bins = 50)






#Model building
#Separate and x and y variables
#training data
X_train = house.drop('SalePrice', axis = 1)
Y_train = house['SalePrice']
#test data
X_test = house.drop('SalePrice', axis = 1)
Y_test = house['SalePrice']


#Other OLS method with scikitlearn
#Create linear regression object
#regr = ln.LinearRegression()
#Train model with training sets
#regr.fit(X, Y)

#Evaluation

#Cross validation score for MSE 10 folds
#MSE_Scores = cross_val_score(regr, X_train, y_train, scoring = 'mean_squared_error', cv = 10)


#Data cleaning
#Fill in missing data


#Drop ID column


