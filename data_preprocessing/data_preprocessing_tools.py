#### Data Preprocessing Tools

##### Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

##### Importing the dataset
###
# we create the data frame
dataset = pd.read_csv('Data.csv')

# features - The Columns with which we are going to predict the Dependent Variable
# For this we create the Matrix of Features
# How To:
# [:] range without upper/lower bounds means to take all the rows; [:,] which columns we want to select with the indexes; [:, :-1] means all the columns except the last column who has index is -1
# In short, these are the Independent Variables
x = dataset.iloc[:, :-1].values

# Dependent Variable Vector - The last column (purchased yes/no)
# We only want the last column, so we remove the range
y = dataset.iloc[:, -1].values

print(x)
print(y)

##### Taking care of missing data

##### Encoding categorical data

###### Encoding the Independent Variable

###### Encoding the Dependent Variable

##### Splitting the dataset into the Training set and Test set

##### Feature Scaling
