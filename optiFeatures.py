# -*- coding: utf-8 -*-
"""
Python code of Feature Selection using Gravitational Search Algorithm (GSA) and Support Vector Machine (SVM)


Coded by: Mukesh Saraswat (saraswatmukesh@gmail.com), Himanshu Mittal (emailid: himanshu.mittal224@gmail.com) and Raju Pal (emailid: raju3131.pal@gmail.com)
The code template used is similar given at link: https://github.com/himanshuRepo/GSA_PythonCode and https://github.com/7ossam81/EvoloPy.

 -- Purpose: Defining the optimal features after all the iterations of GSA with different accuracy measures (TP, FP, TN, FN)
              and parameters: function Name, lowerbound, upperbound, dimensions

Code compatible:
 -- Python: 2.* or 3.*
"""

import numpy
import math
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def optiFeats(indi,df,x=None):
  """ Optimal set of features """
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

  print("The optimal features are:")
  print(x)
  print("\n")
  print("The index of optimal features are:")
  print(oneInd)
  print("\n")
  # Save the index of selected features
  numpy.savetxt("FeatureIndex.csv", oneInd, delimiter=",")

  oneInd=oneInd.reshape(1, -1)
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
  ConfMatrix=confusion_matrix(y_test, y_pred)
  time.sleep(5)
  ConfMatrix1D=ConfMatrix.flatten()
  #print(ConfMatrix1D)
  printAcc=[]
  printAcc.append(accuracy_score(y_test, y_pred,normalize=True)) 

  classification_results= numpy.concatenate((printAcc,ConfMatrix1D))
  # print(classification_results)
  return classification_results




