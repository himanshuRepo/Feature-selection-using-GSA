# -*- coding: utf-8 -*-
"""
Python code of Feature Selection using Gravitational Search Algorithm (GSA) and Support Vector Machine (SVM)


Coded by: Mukesh Saraswat (saraswatmukesh@gmail.com), Himanshu Mittal (emailid: himanshu.mittal224@gmail.com) and Raju Pal (emailid: raju3131.pal@gmail.com)
The code template used is similar given at link: https://github.com/himanshuRepo/GSA_PythonCode and https://github.com/7ossam81/EvoloPy.

 -- Purpose: Main File::
                Calling the Gravitational Search Algorithm(GSA) Algorithm 
                for selecting the optimal feature set

Code compatible:
 -- Python: 2.* or 3.*
"""
import GSA as gsa
import benchmarks
import csv
import numpy
import time
import pandas as pd
import optiFeatures


def selector(algo,func_details,popSize,Iter,df):
    function_name=func_details[0]
    lb=func_details[1]
    ub=func_details[2]
    dim=func_details[3]
    

    if(algo==0):
        x=gsa.GSA(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter,df)  
        testClassification_results=optiFeatures.optiFeats(x.gBest,df)  
        x.testAcc=testClassification_results[0]
        x.testTP=testClassification_results[1]
        x.testFN=testClassification_results[2]
        x.testFP=testClassification_results[3]
        x.testTN=testClassification_results[4] 
    return x
    
    
# Select optimizers
GSA= True # Code by Himanshu Mittal




Algorithm=[GSA]
datasets=["Iris"]
        
# Select number of repetitions for each experiment. 
# To obtain meaningful statistical results, usually 30 independent runs 
# are executed for each algorithm.
Runs=1

# Select general parameters for all optimizers (population size, number of iterations)
PopSize = 5
iterations= 10

#Export results ?
Export=True


#ExportToFile="YourResultsAreHere.csv"
#Automaticly generated name by date and time
ExportToFile="experiment"+time.strftime("%Y-%m-%d-%H-%M-%S")+".csv" 

# Check if it works at least once
atLeastOneIteration=False


# CSV Header for for the cinvergence 
CnvgHeader=[]

for l in range(0,iterations):
	CnvgHeader.append("Iter"+str(l+1))


for j in range (0, len(datasets)):        # specfiy the number of the datasets
    # df=pd.read_csv("Iris.csv")
    dataset=datasets[j]+".csv"
    df=pd.read_csv(dataset)
    for i in range (0, len(Algorithm)):
        if(Algorithm[i]==True): # start experiment if an Algorithm and an objective function is selected
            for k in range (0,Runs):
                
                func_details=["F1",0,1,(len(df.columns)-2)] # 2 is subracted as in IRIS.csv, only 4 columns define the features
                x=selector(i,func_details,PopSize,iterations,df)
                if(Export==True):
                    with open(ExportToFile, 'a') as out:
                        writer = csv.writer(out,delimiter=',')
                        if (atLeastOneIteration==False): # just one time to write the header of the CSV file
                            header= numpy.concatenate([["Optimizer","Dataset","objfname","Experiment","startTime","EndTime","ExecutionTime", "testAcc", "testTP","testFN","testFP","testTN"],CnvgHeader])
                            writer.writerow(header)
                        a=numpy.concatenate([[x.Algorithm,datasets[j],x.objectivefunc,k+1,x.startTime,x.endTime,x.executionTime, x.testAcc, x.testTP,x.testFN,x.testFP,x.testTN],x.convergence])
                        writer.writerow(a)
                    out.close()
                atLeastOneIteration=True # at least one experiment
                
if (atLeastOneIteration==False): # Faild to run at least one experiment
    print("No Optomizer or Cost function is selected. Check lists of available optimizers and cost functions") 
        
        
