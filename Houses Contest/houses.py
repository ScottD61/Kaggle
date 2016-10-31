# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Import modules
import pandas as pd

#Import dataset
house_train = pd.read_csv('train.csv')

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

#Model building

