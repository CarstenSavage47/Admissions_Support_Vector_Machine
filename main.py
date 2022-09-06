# Thank you to Scikit-Learn project https://scikit-learn.org/stable/modules/svm.html

import torch  # torch provides basic functions, from setting a random seed (for reproducability) to creating tensors.
import torch.nn as nn  # torch.nn allows us to create a neural network.
import torch.nn.functional as F  # nn.functional give us access to the activation and loss functions.
from sklearn.utils import column_or_1d
from torch.optim import SGD  # optim contains many optimizers. Here, we're using SGD, stochastic gradient descent.
import matplotlib.pyplot as plt  ## matplotlib allows us to draw graphs.
import seaborn as sns  ## seaborn makes it easier to draw nice-looking graphs.
import os
from tqdm import tqdm
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing
from statsmodels.formula.api import ols
from sklearn import svm

import pandas
import numpy as np

Admissions = pandas.read_excel('/Users/carstenjuliansavage/Desktop/IPEDS_data.xlsx')
pandas.set_option('display.max_columns', None)

# Filtering dataset for input and output variables only

AdmissionsSlim = (Admissions
    .filter(['Percent admitted - total',
             'ACT Composite 75th percentile score',
             'Historically Black College or University',
             'Total  enrollment',
             'Total price for out-of-state students living on campus 2013-14',
             'Percent of total enrollment that are White',
             'Percent of total enrollment that are women'])
    .dropna()
)

AdmissionsSlim.columns

AdmissionsSlim.columns = ['Per_Admit','ACT_75TH','Hist_Black','Total_ENROLL','Total_Price','Per_White','Per_Women']

# Defining 'Selective' as an Admittance Rate Under 50%
AdmissionsSlim['Per_Admit_Dum'] = np.where(AdmissionsSlim['Per_Admit'] < 50,1,0)
#AdmissionsSlim['Per_Admit'] = np.where(AdmissionsSlim['Per_Admit'] < 50,1,0)
AdmissionsSlim['Hist_Black'] = np.where(AdmissionsSlim['Hist_Black'] == 'Yes',1,0)

# Create a new variable, which is the percentage of total enrollment that are non-white.
AdmissionsSlim = (AdmissionsSlim
    .assign(Per_Non_White=lambda a: 100-a.Per_White)
)

X = AdmissionsSlim[['ACT_75TH',
                    'Hist_Black',
                    'Total_ENROLL',
                    'Total_Price',
                    'Per_Non_White',
                    'Per_Women']]
y = AdmissionsSlim[['Per_Admit_Dum']]
y_Regression = AdmissionsSlim[['Per_Admit']]

# Split dataframe into training and testing data. Remember to set a seed.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47, stratify=y)

X_train_R, X_test_R, y_train_R, y_test_R = train_test_split(X, y_Regression, test_size=0.2, random_state=47, stratify=y)

# Scaling the data to be between 0 and 1
min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.fit_transform(X_test)
X_train_R = min_max_scaler.fit_transform(X_train_R)
X_test_R = min_max_scaler.fit_transform(X_test_R)
y_train_R = min_max_scaler.fit_transform(y_train_R)

# Let's confirm that the scaling worked as intended.
# All values should be between 0 and 1 for all variables.
X_Stats = pandas.DataFrame(X_train)
pandas.set_option('display.max_columns', None)
X_Stats.describe()

y_train_Stats = pandas.DataFrame(y_train)
y_test_Stats = pandas.DataFrame(y_test)
y_train_Stats.describe()
y_test_Stats.describe()

y_train = column_or_1d(y_train, warn=True)
y_train_R = column_or_1d(y_train_R, warn=True)

# Using support vector machine model to fit X the training data.

# For the classification, we are using the modified dummy variable version of Per_Admit.
clf = svm.SVC()
clf.fit(X_train, y_train)

# For the regression, we're using the unmodified Per_Admit (not the dummy).
# This is because we want to predict continuous outputs.
regr = svm.SVR()
regr.fit(X_train_R, y_train_R)


# This is the prediction function for the classification support vector machine model.
def SVM_CLF_Prediction(ACT_75TH,Hist_Black,Total_ENROLL,Total_Price,Per_Non_White,Per_Women):
    Prediction = clf.predict([[ACT_75TH,Hist_Black,Total_ENROLL,Total_Price,Per_Non_White,Per_Women]])
    return Prediction

# The classifications are working as expected.
SVM_CLF_Prediction(ACT_75TH=1,Hist_Black=0,Total_ENROLL=1,Total_Price=1,Per_Non_White=0,Per_Women=1)

SVM_CLF_Prediction(ACT_75TH=0.2,Hist_Black=0.2,Total_ENROLL=0.2,Total_Price=0.2,Per_Non_White=0.5,Per_Women=0.5)



# We're predicting the Selectivity rate based on these inputs using the regression version.
# This is the prediction function for the regression support vector machine model.
def SVM_REG_Prediction(ACT_75TH,Hist_Black,Total_ENROLL,Total_Price,Per_Non_White,Per_Women):
    Prediction = regr.predict([[ACT_75TH,Hist_Black,Total_ENROLL,Total_Price,Per_Non_White,Per_Women]])
    return Prediction

# We can try to input some values in the SVM and evaluate the output.
# In the regression, it is predicting continuous values, which are the percentage rates for admittance.

SVM_REG_Prediction(ACT_75TH=1.0,
                   Hist_Black=0.5,
                   Total_ENROLL=0.5,
                   Total_Price=1.0,
                   Per_Non_White=0.5,
                   Per_Women=0.5
                   )