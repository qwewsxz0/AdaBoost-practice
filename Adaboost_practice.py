# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:52:06 2019

@author: IRIS168
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_blobs




datMat,classLabels=make_gaussian_quantiles(n_samples=13000,n_features=2,n_classes=2,mean=[1,2],cov=2)
plt.scatter(datMat[:,0],datMat[:,1],marker='o',c=classLabels,s=3)
plt.show()
print(classLabels)
datMat,classLabels=make_blobs(n_samples=13000,n_features=2,centers=[[1,1],[4,4]],cluster_std=[0.4,0.2])
plt.scatter(datMat[:,0],datMat[:,1],marker='o',c=classLabels,s=3)
plt.show()
#print(classLabels)


def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen]<=threshVal] =-1
    else:
        retArray[dataMatrix[:,dimen]>threshVal] =-1
    return retArray

def buildStump(dataArr, classLabels,D):
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m,n = np.shape(dataMatrix)
    numSteps = 100.0
    bestStump,bestClasEst = {},np.mat(np.zeros((m,1)))
    minError = float("inf")
    for i in range(n):
        rangMin = dataMatrix[:,i].min()
        rangMax = dataMatrix[:,i].max()
#        print('dataMatrix[:,i]',i,dataMatrix[:,i])
        stepSize = (rangMax - rangMin)/numSteps
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:
                threshVal = (rangMin + float(j)*stepSize)
                predictedVals = \
                            stumpClassify(dataMatrix,int(i),threshVal,inequal)
#                print('predictedVals',inequal,predictedVals)
                errArr = np.mat(np.ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr
                print(weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
#                    print('bestStump',bestStump)
    return bestStump,minError,bestClasEst

def adaBoostTrainDS(dataArr,classLabels,numIt = 40):
    weakClassArr = []
    errorRateRecord = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m,1))/m)
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr, classLabels,D)
#        print("D:",D.T)
        alpha = float(0.5*np.log((1-error)/max(error,1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
#        print("classEst:",classEst.T)
        expon = np.multiply(-1*alpha*np.mat(classLabels).T,classEst)
        D = np.multiply(D,np.exp(expon))
        D = D/D.sum()
        aggClassEst +=alpha*classEst
#        print("aggClassEst:",aggClassEst.T)
        aggErrors = \
        np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T,
                    np.ones((m,1)))
        errorRate = aggErrors.sum()/m
        errorRateRecord.extend([errorRate])
        print("total error:",errorRate,"\n")
        if errorRate == 0.0:
            break
    return weakClassArr,errorRateRecord
#datMat,classLabels = loadimpData()
#print(datMat)
#print(np.shape(datMat))
#a,_ = np.shape(datMat)

#datMatlen = len(datMat)
size,_ = np.shape(datMat)
D = np.mat(np.ones((size,1))/size)
#print(buildStump(datMat,classLabels,D))
classLabels = (classLabels-0.5)*2

#numIt = 100
#weakClassArr , errorRateRecord = adaBoostTrainDS(datMat,classLabels,numIt)
#
#
#print(weakClassArr)
#
##print(np.dtype(weakClassArr))
##print(weakClassArr[1]['thresh'])
##thresh = weakClassArr[1]['thresh']
##if weakClassArr[1]['dim'] ==0:
##    plt.plot([thresh,thresh],[min(datMat[:,1]),max(datMat[:,1])])
##else:
##    plt.plot([min(datMat[:,0]),max(datMat[:,0])],[thresh,thresh])
##plt.scatter(datMat[:,0],datMat[:,1],marker='o',c=classLabels,s=3)
##plt.show()
#print(errorRateRecord)
#plt.scatter(datMat[:,0],datMat[:,1],marker='o',c=classLabels,s=3)

#plt.show()
#for i in range(numIt):
#    thresh = weakClassArr[i]['thresh']
#    if weakClassArr[i]['dim'] ==0:
#        plt.plot([thresh,thresh],[min(datMat[:,1]),max(datMat[:,1])])
#    else:
#        plt.plot([min(datMat[:,0]),max(datMat[:,0])],[thresh,thresh])
#plt.scatter(datMat[:,0],datMat[:,1],marker='o',c=classLabels,s=3)
#plt.show()
#
#plt.plot(errorRateRecord)
#plt.show()



for num_weak_classifier in range (1,10):
    print('num_weak_classifier',num_weak_classifier)
    weakClassArr , errorRateRecord = adaBoostTrainDS(datMat,classLabels,num_weak_classifier)
    print(weakClassArr)
    plt.show()
    for i in range(num_weak_classifier):
        thresh = weakClassArr[i]['thresh']
        if weakClassArr[i]['dim'] ==0:
            plt.plot([thresh,thresh],[min(datMat[:,1]),max(datMat[:,1])])
        else:
            plt.plot([min(datMat[:,0]),max(datMat[:,0])],[thresh,thresh])
    plt.scatter(datMat[:,0],datMat[:,1],marker='o',c=classLabels,s=3)
#    plt.savefig('C:\test.png')
    plt.show()
    plt.plot(errorRateRecord)
#    plt.savefig('C:\error_test.png')
    plt.show()