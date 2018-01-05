#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 21:41:13 2017

@author: annaguidi, samanthayip
"""

import numpy as np
from scipy import misc
from matplotlib import pylab as plt
import matplotlib.cm as cm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
%matplotlib inline

train_labels, train_data = [], []
for line in open('./faces/train.txt'):
    im = misc.imread(line.strip().split()[0])
    train_data.append(im.reshape(2500,))
    train_labels.append(line.strip().split()[1])
train_data, train_labels = np.array(train_data, dtype=float), np.array(train_labels, dtype=int)

print(train_data.shape, train_labels.shape)
plt.imshow(train_data[10, :].reshape(50,50), cmap = cm.Greys_r)
plt.show()

print(train_data[10, 10])



test_labels, test_data = [], []
for line in open('./faces/test.txt'):
    im = misc.imread(line.strip().split()[0])
    test_data.append(im.reshape(2500,))
    test_labels.append(line.strip().split()[1])
test_data, test_labels = np.array(test_data, dtype=float), np.array(test_labels, dtype=int)
"""
Average Face. Compute the average face µ from the whole training set by summing up every
column in X then dividing by the number of faces. Display the average face as a grayscale
image.
"""

 
average_face = train_data.mean(axis=0)

plt.imshow(average_face.reshape(50,50), cmap = cm.Greys_r)
plt.show()

'''
Mean Subtraction. Subtract average face µ from every column in X. That is, xi
:= xi − µ,
where xi is the i-th column of X. Pick a face image after mean subtraction from the new X
and display that image in grayscale. Do the same thing for the test set Xtest using the precomputed
average face µ in (c).
'''


new_matrix = []

for x in train_data:
    new_matrix.append((x - average_face))
    
new_matrix = np.array(new_matrix)

plt.imshow(new_matrix[10, :].reshape(50,50), cmap = cm.Greys_r)
plt.show()



new_test_matrix = []

for x in test_data:
    new_test_matrix.append((x - average_face))
    
new_test_matrix = np.array(new_test_matrix)

plt.imshow(new_test_matrix[10, :].reshape(50,50), cmap = cm.Greys_r)
plt.show()

'''
Perform Singular Value Decomposition (SVD) on training set X (X = UΣV^T) to get
matrix VT, where each row of VT has the same dimension as the face image.
'''

U, s, V = np.linalg.svd(new_matrix, full_matrices=False)

S = np.diag(s)

print(V.shape)

'''
We refer to vi, the i-th row of VT, as i-th eigenface.
Display the first 10 eigenfaces as 10 images in grayscale.
WHICH first 10 eigenfaces??
'''
for i in range(10):
    plt.imshow(V[i, :].reshape(50,50), cmap = cm.Greys_r)
    plt.show()

'''
for i in range(10):
    plt.imshow(new_test_matrix[i, :].reshape(50,50), cmap = cm.Greys_r)
    plt.show()
'''

'''
Low-rank Approximation. Since Σ is a diagonal matrix with non-negative real numbers on
the diagonal in non-ascending order, we can use the first r elements in Σ together with first
r columns in U and first r rows in VT to approximate X. That is, we can approximate X by
Xˆr = U[:,: r ] Σ[: r,: r ] VT[: r,:]. The matrix Xˆr is called rank-r approximation of X. Plot the
rank-r approximation error kX−XˆrkF 2 as a function of r when r = 1, 2,..., 200.
'''

plotted = []

for r in range(1, 201, 1):
    Xr = np.dot(U[:,:r],S[:r,:r])
    Xr = np.dot(Xr, V[:r,:])
    plotted.append(np.linalg.norm(new_matrix - Xr))
    

plt.scatter(range(0, 200, 1),plotted, marker=".")
plt.show()

'''
Eigenface Feature. The top r eigenfaces VT[: r,:] = {v1, v2,..., vr }
T span an r -dimensional linear subspace of the original image space called face space,
whose origin is the average face µ, and whose axes are the eigenfaces {v1, v2,..., vr }.
Therefore, using the top r eigenfaces {v1, v2,..., vr }, we can represent a 2500-dimensional face
image z as an r -dimensional feature vector f: f = VT[: r,:] z = [v1, v2,..., vr ]T z.
Write a function to generate r -dimensional feature matrix F and Ftest for training images X and
test images Xtest, respectively (to get F, multiply X to the transpose of first r rows of VT,
F should have same number of rows as X and r columns; similarly for Xtest).
'''

effs = []
def generateFeatureMatrix():
    for r in range(1, 201, 1):
        VT = V[:r,:].T
        F = np.dot(new_matrix,VT)
        effs.append(F)
    return effs
        
generateFeatureMatrix()


ftest = []
def generateFeatureTestMatrix():
    for r in range(1, 201, 1):
        VT = V[:r,:].T
        F_test = np.dot(new_test_matrix,VT)
        ftest.append(F_test)
    return ftest

generateFeatureTestMatrix()

'''
Face Recognition. Extract training and test features for r = 10. Train a Logistic Regression
model using F and test F on Ftest. Report the classification accuracy on the test set. Plot the
classification accuracy on the test set as a function of r when r = 1, 2,..., 200. Use “one-vsrest”
logistic regression, where a classifier is trained for each possible output label. Each
classifier is trained on faces with that label as positive data and all faces with other labels as
negative data. sklearn calls this “ovr” mode.
'''
f_r10 = effs[9]
test_feature_ten = ftest[9]

OVR = OneVsRestClassifier(LogisticRegression()).fit(f_r10,train_labels)

print(OVR.score(test_feature_ten,test_labels))

accuracy = []

def classifier(r):
    OVR = OneVsRestClassifier(LogisticRegression()).fit(effs[r - 1],train_labels)
    accuracy.append(OVR.score(ftest[r - 1],test_labels))
    
for r in range(1, 201, 1):
    classifier(r)
    
plt.plot(accuracy)
plt.show()