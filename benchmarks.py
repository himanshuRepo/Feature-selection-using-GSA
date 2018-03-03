# -*- coding: utf-8 -*-
"""
Python code of Feature Selection using Gravitational Search Algorithm (GSA) and Support Vector Machine (SVM)


Coded by: Mukesh Saraswat (saraswatmukesh@gmail.com), Himanshu Mittal (emailid: himanshu.mittal224@gmail.com) and Raju Pal (emailid: raju3131.pal@gmail.com)
The code template used is similar given at link: https://github.com/himanshuRepo/GSA_PythonCode and https://github.com/7ossam81/EvoloPy.

 -- Purpose: Define the objective function for feature selection as minimization of accuracy_score (cost function)
              and parameters: function Name, lowerbound, upperbound, dimensions

Code compatible:
 -- Python: 2.* or 3.*
"""

import numpy
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def F1(indi,df,x=None):
  """ Fitness function for selected features """
  # a user-defined threshold value
  thr=0.5
  if x is None:
    x = []
  x = list(indi)
  x=numpy.asarray(x)
  x=x.reshape(1, -1)
  for i in range(x.shape[1]):
    if x[0,i]> thr:
      x[0,i]=1.
    else:
      x[0,i]=0.

  [_,oneInd]=numpy.where(x == 1.)


  # Check number of features selected, if no feature is selected then select the first feature by default
  if oneInd.size:
  	pass
  else:
  	oneInd=numpy.zeros((1, 1),dtype=numpy.int)

  oneInd=oneInd.reshape(1, -1)
  # df1=pd.read_csv("Iris.csv")
  df1=pd.DataFrame(df, copy=True)
  df2=df1.iloc[:,1:]
  X=df2.iloc[:, oneInd[0,:]]
  X=X.values
  y=df2.iloc[:, -1]
  y_act=y.astype('category').cat.codes
  y_act=y_act.values
  X_train, X_test, y_train, y_test = train_test_split(X, y_act,test_size=0.30)
  clf = SVC()
  clf.fit(X_train, y_train)
  y_pred=clf.predict(X_test)
  s=accuracy_score(y_test, y_pred)
  return s




