
# coding: utf-8

# In[546]:


"""
Created on Mon Sep 18 21:41:13 2017

@author: annaguidi, samanthayip
"""
import numpy as np
import pandas as pd
import sklearn
from scipy import misc
from matplotlib import pylab as plt
import matplotlib.cm as cm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import datasets, cross_validation, metrics, model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics 
from sklearn.naive_bayes import BernoulliNB
import csv


# In[394]:


get_ipython().magic('matplotlib inline')


# In[395]:


training = pd.read_json('train.json')


# In[396]:


training


# In[397]:


testing = pd.read_json('test.json')


# In[536]:


testing


# ## Tell us about the data. How many samples (dishes) are there in the training set?

# In[398]:


len(training)


# In[399]:


print("there are", len(training), "sample dishes in the training set")


# ## How many categories (types of cuisine)?

# In[400]:


cuisine_types = training.cuisine.unique()


# In[401]:


len(cuisine_types)


# In[402]:


print("there are", len(cuisine_types),"categories of cuisine")


# ## Use a list to keep all the unique ingredients appearing in the training set.
# ## How many unique ingredients are there?

# In[403]:


#this makes a series of ingredients
ingredients = training.ingredients


# In[404]:


ingredients = np.array(ingredients)


# In[405]:


ingredients_array = np.hstack(ingredients)


# In[406]:


len(set(ingredients_array))


# In[407]:


print("there are", len(set(ingredients_array)), "number of uinque ingredients")


# In[408]:


ingredients_list = list(set(ingredients_array))


# In[409]:


set(ingredients_array)


# In[410]:


ingredients_list


# In[411]:


# ing_list = training.ingredients.tolist()
# flat_list_ing = [item for sublist in ing_list for item in sublist]
# len(set(flat_list_ing))


# ### Represent each dish by a binary ingredient feature vector. Suppose there are d different ingredients in total from the training set, represent each dish by a 1×d binary ingredient vector x, where xi = 1 if the dish contains ingredient i and xi = 0 otherwise. For example, suppose all the ingredients we have in the training set are { beef, chicken, egg, lettuce, tomato, rice } and the dish is made by ingredients { chicken, lettuce, tomato, rice }, then the dish could be represented by a 6×1 binary vector [0, 1, 0, 1, 1, 1] as its feature or attribute. Use n ×d feature matrix to represent all the dishes in training set and test set, where n is the number of dishes.

# In[412]:


training_array = np.array(training)


# In[479]:


testing_array = np.array(testing)


# In[413]:


feature_matrix = np.zeros((len(training_array), len(set(ingredients_array))))


# In[481]:


feature_test_matrix = np.zeros((len(testing_array), len(set(ingredients_array))))


# In[414]:


#lets see how necessary this is
ingredient_dictionary = dict(zip(ingredients_list, range(len(ingredients_list))))


# In[489]:


for q, dish in enumerate(training_array):
    for i in dish[2]:
        feature_matrix[q][ingredient_dictionary.get(str(i))] = 1


# In[490]:


for q, dish in enumerate(testing_array):
    for i in dish[1]:
        feature_test_matrix[q][ingredient_dictionary.get(str(i))] = 1


# In[495]:


testing_array[1][1]


# In[491]:


sum(feature_test_matrix[1])


# In[416]:


sum(feature_matrix[0])


# In[417]:


len(training.ingredients[0])


# In[418]:


# vectors = []
# for item in ingredients_list:
#     for item2 in training_array[0][2]:
#         if item == item2:
#             vectors.append(1)
#         else:
#             vectors.append(0)


# # Using Naïve Bayes Classifier to perform 3 fold cross-validation on the training set and report your average classification accuracy. Try both Gaussian distribution prior assumption and Bernoulli distribution prior assumption.

# In[419]:


x_trainf, y_trainf, z_trainf = cross_validation.KFold(len(feature_matrix[:,:]), n_folds=3)


# In[420]:


x_trainf


# In[421]:


train_indices = x_trainf[0]
traindata = feature_matrix[train_indices]


# In[422]:


train_indices_y = y_trainf[0]
traindata_y = feature_matrix[train_indices_y]


# In[423]:


train_indices_z = z_trainf[0]
traindata_z = feature_matrix[train_indices_z]


# In[424]:


traindata.shape


# In[425]:


trainlabels = training['cuisine'][train_indices]


# In[426]:


trainlabels_y = training['cuisine'][train_indices_y]


# In[427]:


trainlabels_z = training['cuisine'][train_indices_z]


# In[428]:


trainlabels.shape


# In[429]:


trainlabels


# In[430]:


model = GaussianNB()


# In[431]:


model.fit(traindata,trainlabels)


# In[432]:


k1 = model.predict(traindata[0:6000])


# In[433]:


sklearn.metrics.accuracy_score(trainlabels[0:6000], k1, normalize=True, sample_weight=None)


# In[434]:


model.fit(traindata_y,trainlabels_y)


# In[435]:


k2 = model.predict(traindata_y[0:6000])


# In[436]:


sklearn.metrics.accuracy_score(trainlabels_y[0:6000], k2, normalize=True, sample_weight=None)


# In[437]:


model.fit(traindata_z,trainlabels_z)


# In[438]:


k3 = model.predict(traindata_z[0:6000])


# In[439]:


sklearn.metrics.accuracy_score(trainlabels_y[0:6000], k3, normalize=True, sample_weight=None)


# In[440]:


clf = BernoulliNB()


# In[441]:


clf.fit(traindata, trainlabels)


# In[442]:


k4 = clf.predict(traindata[0:6000])


# In[443]:


sklearn.metrics.accuracy_score(trainlabels[0:6000], k4, normalize=True, sample_weight=None)


# In[444]:


clf.fit(traindata_y, trainlabels_y)


# In[445]:


k5 = clf.predict(traindata_y[0:6000])


# In[446]:


sklearn.metrics.accuracy_score(trainlabels_y[0:6000], k5, normalize=True, sample_weight=None)


# In[447]:


clf.fit(traindata_z, trainlabels_z)


# In[448]:


k6 = clf.predict(traindata_y[0:6000])


# In[449]:


sklearn.metrics.accuracy_score(trainlabels_y[0:6000], k6, normalize=True, sample_weight=None)


# ## For Gaussian prior and Bernoulli prior, which performs better in terms of cross-validation accuracy? Why? Please give specific arguments

# Bernoulli perfoms better
# Gaussian assumes that features follow a normal distribution, which is not necessarily the case with the ingredients in our samples
# Bernoulli is useful if your feature vectors are binary (i.e. zeros and ones), which is the case in this exercise: either the ingredient is present, or it is not. Therefore the distribution is much more accurate.

# # Using Logistic Regression Model to perform 3 fold cross-validation on the training set and report your average classification accuracy

# In[450]:


LogReg = LogisticRegression()


# In[451]:


logreg_fit = LogReg.fit(traindata, trainlabels)


# In[452]:


y_pred1 = LogReg.predict(traindata)


# In[453]:


sklearn.metrics.accuracy_score(trainlabels, y_pred1, normalize=True, sample_weight=None)


# In[454]:


logreg_fit = LogReg.fit(traindata_y, trainlabels_y)


# In[455]:


y_pred2 = LogReg.predict(traindata_y)


# In[456]:


sklearn.metrics.accuracy_score(trainlabels_y, y_pred2, normalize=True, sample_weight=None)


# In[457]:


logreg_fit = LogReg.fit(traindata_z, trainlabels_z)


# In[458]:


y_pred3 = LogReg.predict(traindata_z)


# In[459]:


sklearn.metrics.accuracy_score(trainlabels_z, y_pred3, normalize=True, sample_weight=None)


# ## Train your best-performed classifier with all of the training data, and generate test labels on test set. Submit your results to Kaggle and report the accuracy.

# In[477]:


#train your best-performed classifier with all of the training data
logreg_fit = LogReg.fit(traindata, trainlabels)


# In[478]:


#now it's time to predict based on testing data, which needs to be in 1s and 0s


# In[498]:


final_predictions = LogReg.predict(feature_test_matrix)


# In[499]:


final_predictions


# In[517]:


len(testing)


# In[520]:


len(final_predictions)


# In[539]:


pred_id = list(zip(final_predictions,testing['id']))


# In[540]:


pred_id


# In[542]:


#write to csv file


# In[580]:


import csv
import sys

f = open(sys.argv[1], 'wt')
try:
    writer = csv.writer(f)
    writer.writerow( ('id', 'cuisine') )
    for line in pred_id:
        writer.writerow( (line[1], line[0]) )
finally:
    f.close()

