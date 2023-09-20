# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Gtagorical data
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[ : , 3] = labelencoder_X.fit_transform(X[ : , 3])
ct = ColumnTransformer([('one-hot-encoder', OneHotEncoder(categories='auto'),[3])], remainder='passthrough')
X = ct.fit_transform(X)

#Avoid Dummy vars Trap
X = X[ : , 1: ]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#Fittind Multible Reg
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)


#Building model using Backward Elimination
import statsmodels.api as sm
X = np.append(arr = np.ones((50,1)).astype(int) , values = X , axis = 1 )
X_opt = np.array(X[:, [0, 1, 2, 3, 4, 5]], dtype=float)
ols = sm.OLS(endog = y , exog= X_opt).fit()
ols.summary()

X_opt = np.array(X[:, [0, 1, 3, 4, 5]], dtype=float)
ols = sm.OLS(endog = y , exog= X_opt).fit()
ols.summary()

X_opt = np.array(X[:, [0, 3, 4, 5]], dtype=float)
ols = sm.OLS(endog = y , exog= X_opt).fit()
ols.summary()

X_opt = np.array(X[:, [0, 3, 5]], dtype=float)
ols = sm.OLS(endog = y , exog= X_opt).fit()
ols.summary()


X_opt = np.array(X[:, [0, 3]], dtype=float)
ols = sm.OLS(endog = y , exog= X_opt).fit()
ols.summary()
