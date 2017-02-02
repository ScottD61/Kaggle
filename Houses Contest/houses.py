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
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


#Import train and test datasets
house_train = pd.read_csv('train.csv')
house_test = pd.read_csv('test.csv')

#Data cleaning
#Replace NA factor name in categorical variables with another name
house_train['Alley'].replace('NA', 'NoAlley', inplace = True)
house_test['Alley'].replace('NA', 'NoAlley', inplace = True)
house_train['BsmtQual'].replace('NA', 'NB', inplace = True)
house_test['BsmtQual'].replace('NA', 'NB', inplace = True)
house_train['BsmtCond'].replace('NA', 'NB', inplace = True)
house_test['BsmtCond'].replace('NA', 'NB', inplace = True)
house_train['BsmtExposure'].replace('NA', 'NB', inplace = True)
house_test['BsmtExposure'].replace('NA', 'NB', inplace = True)
house_train['BsmtFinType1'].replace('NA', 'NB', inplace = True)
house_test['BsmtFinType1'].replace('NA', 'NB', inplace = True)
house_train['BsmtFinType2'].replace('NA', 'NB', inplace = True)
house_test['BsmtFinType2'].replace('NA', 'NB', inplace = True)
house_train['FireplaceQu'].replace('NA', 'NF', inplace = True)
house_test['FireplaceQu'].replace('NA', 'NF', inplace = True)
house_train['GarageType'].replace('NA', 'NG', inplace = True)
house_test['GarageType'].replace('NA', 'NG', inplace = True)
house_train['GarageFinish'].replace('NA', 'NG', inplace = True)
house_test['GarageFinish'].replace('NA', 'NG', inplace = True)
house_train['GarageQual'].replace('NA', 'NG', inplace = True)
house_test['GarageQual'].replace('NA', 'NG', inplace = True)
house_train['GarageCond'].replace('NA', 'NG', inplace = True)
house_test['GarageCond'].replace('NA', 'NG', inplace = True)
house_train['PoolQC'].replace('NA', 'NP', inplace = True)
house_test['PoolQC'].replace('NA', 'NP', inplace = True)
house_train['Fence'].replace('NA', 'NF', inplace = True)
house_test['Fence'].replace('NA', 'NF', inplace = True)
house_train['MiscFeature'].replace('NA', 'NoF', inplace = True)
house_test['MiscFeature'].replace('NA', 'NoF', inplace = True)


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


#SKIP univariate for categorical
#Barcharts of all categorical variables
#Types of dwellings
#a = house_new['MSSubClass'].value_counts()
#a.plot(kind = 'bar')

#Zoning classification
#b = house_new['MSZoning'].value_counts()
#b.plot(kind = 'bar')



#Step 3
#Bi-variate data exploration - there needs to be a reason why
#Correlations
#Diagonal correlation matrix
seaborn.corrplot(house_new, annot = False, diag_names = False)

#Get top 10 correlations with the price variable


#Get top correlations with variables to each other 

#Regression lines on sales prices and several others
#Do saleprice with OverallQual
#Saleprice with YearBuilt
#Saleprprice with GrLivArea


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

#export to .csv to check for nan
house_final.to_csv('house_final.csv')


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

#Model 2 - OLS with decreased features

#Model 3 - OLS after PCA
#list of principal components
n_components = np.array([3, 5, 10, 15, 20, 25, 30, 35, 40, 45])

#Search for number of principal components using gridsearch
#PCA object
pca = PCA(whiten = True)
#Convert dataframes to matrix 
#Independent variables
X_mat = X_train.as_matrix()
#Dependent variable
Y_mat = Y_train.as_matrix()
#Gridsearch
clf = GridSearchCV(estimator = pca, param_grid = dict(n_components = n_components)) 
#Fit linear model 
res = clf.fit(X_mat, Y_mat)  #problem, there is an inf value, DELETE IT 
#Get number of principal components
pca.get_params(res)

#Other way - fixed the problem, it was gridsearch
#To do - get gridsearch to work

pca = PCA(n_components = 10, whiten = True)
#Fit model
results = pca.fit(X_mat)
#Get number of principal components
pca.get_params(results)
#Transform data
x_pca = pca.transform(X_mat)

#Fit model with PCA
#Train model with training sets
regr.fit(x_pca, Y_train)

#Evaluation
#Cross validation score for MSE 10 folds
MSE_Scores_pca = cross_val_score(regr, x_pca, Y_train, scoring = 'neg_mean_squared_error', cv = 10)
#Take average of all cross validation folds
np.mean(MSE_Scores_pca)
#Cross validation score for MAE 10 fold
MAE_Scores_pca = cross_val_score(regr, x_pca, Y_train, scoring = 'mean_absolute_error', cv = 10)
#Take average of all cross validation folds
np.mean(MAE_Scores_pca)
#Cross validation score for R2 10 folds
R2_Scores_pca = cross_val_score(regr, x_pca, Y_train, scoring = 'r2', cv = 10)
#Take average of all cross validation folds
np.mean(R2_Scores_pca)


#Model 4 - Lasso w/ all features
#Model 5 - Ridge w/ all features

#Submit results in test set
#Create join of the id in test set and answers in test set
#solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})
