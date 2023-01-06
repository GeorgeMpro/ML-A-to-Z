#### Data Preprocessing Tools

##### Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Need scikit-learn instead of sklearn in the project > python interpreter
from sklearn.impute import SimpleImputer

# We will use these tools to transform the values of the columns into numbers
# in addition, we will try to remove connection or order between them - no order between the countries
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

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
# we will use the library to deal with missing data

# We will replace the missing salary by the average of salaries
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# now we apply the imputer on the matrix of features. We want to access columns with numbers only - age, salary.
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

print(x)

##### Encoding categorical data

###### Encoding the Independent Variable

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])],
                       remainder='passthrough')  # we want o keep the rest of the columns
# Will return the first column with One Hot Encoding
# We need to turn the ct object to a numpy array which is expected by the machine learning models
x = np.array(ct.fit_transform(x))
# Notice: each country is encoded as a vector of size 3; with no numerical order between them
print('\nEncode countries into a vectors of size 3, made out of 0,1 and without order between them')
print(x)

# label encode yes/no to 1/0
le = LabelEncoder()
y = le.fit_transform(y)
print('\nEncode Yes/No to 1/0')
print(y)


# We apply the Feature scaling AFTER splitting the dataset into the Training and Test sets
##### Splitting the dataset into the Training set and Test set



##### Feature Scaling
