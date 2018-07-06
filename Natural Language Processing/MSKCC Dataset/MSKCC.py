# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 11:15:00 2017

@author: LS
"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import keras as ks
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,explained_variance_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
#from sklearn.cross_validation import cross_val_score

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU
from sklearn.model_selection import StratifiedKFold
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import LabelEncoder


################ Processing training set#################

training_text=pd.read_csv("C:/Users/LS/Downloads/Kaggle/MSKCC data/training_text.csv")
training_text.shape
list(training_text)


training_variants=pd.read_csv("C:/Users/LS/Downloads/Kaggle/MSKCC data/training_variants.csv")
training_variants.shape
list(training_variants)



training = pd.concat([training_text,training_variants],axis=1)
training.shape
#training = pd.concat([training['ID'],training[' Text'],training['Gene'],training['Variation'],training['Class']],axis=1)
training = pd.concat([training[' Text'],training['Gene'],training['Variation'],training['Class']],axis=1)
list(training)    

training.to_csv('C:/Users/LS/Downloads/Kaggle/MSKCC data/training.csv')


#################### Processing test set ####################

test_text=pd.read_csv("C:/Users/LS/Downloads/Kaggle/MSKCC data/test_text.csv")
test_text.shape
list(test_text)
len(test_text)

test_variants=pd.read_csv("C:/Users/LS/Downloads/Kaggle/MSKCC data/test_variants.csv")
test_variants.shape
list(test_variants)
testlength = len(test_variants)


test = pd.concat([test_text,test_variants],axis=1)
test.shape
#test = pd.concat([test['ID'],test[' Text'],test['Gene'],test['Variation']],axis=1)
test = pd.concat([test[' Text'],test['Gene'],test['Variation']],axis=1)
list(test)    
test.to_csv('C:/Users/LS/Downloads/Kaggle/MSKCC data/test.csv')












####################################################################################################################
####################################################################################################################
####################################################################################################################
#################################Submission 1#######################################################################
import random
training_text=pd.read_csv("C:/Users/LS/Downloads/Kaggle/MSKCC data/training_text.csv")
training_text.shape
list(training_text)


training_variants=pd.read_csv("C:/Users/LS/Downloads/Kaggle/MSKCC data/training_variants.csv")
training_variants.shape
list(training_variants)


traininglength = len(training_variants)
testlength = len(test_variants)

Classlabel_1 = int(568*testlength/traininglength) 
Classlabel_2 = int(452*testlength/traininglength)
Classlabel_3 = int(89*testlength/traininglength)+1
Classlabel_4 = int(686*testlength/traininglength)
Classlabel_5 = int(242*testlength/traininglength)+1
Classlabel_6 = int(275*testlength/traininglength)+1
Classlabel_7 = int(953*testlength/traininglength)+1
Classlabel_8 = int(19*testlength/traininglength)
Classlabel_9 = int(37*testlength/traininglength)



TrainPredicted_Class = np.array([1]*568 + [2]*452 + [3]*89 + [4]*686 + [5]*242 + [6]*275 + [7]*953 + [8]*19 + [9]*37) 
TestPredicted_Class = np.array([1]*Classlabel_1 + [2]*Classlabel_2 + [3]*Classlabel_3 + [4]*Classlabel_4 + [5]*Classlabel_5 + [6]*Classlabel_6 + [7]*Classlabel_7 + [8]*Classlabel_8 + [9]*Classlabel_9)

random.shuffle(TrainPredicted_Class)
training_variants["TrainPredicted_Class"] = TrainPredicted_Class 

Accuracy = pd.DataFrame(np.array([0]*len(training_variants)))
training_variants["Accuracy"] = Accuracy[0]

count = 0
for i in range(0,len(training_variants)-1):
    if(training_variants["TrainPredicted_Class"][i] == training_variants["Class"][i]):
        training_variants["Accuracy"][i] == 1
        count = count+1

TrainingAccuracy= 100*count/len(training_variants)



random.shuffle(TestPredicted_Class)

N = len(TestPredicted_Class)
K = np.unique(TestPredicted_Class)[-1]

TestClass = np.zeros((N, K))
TestClass.shape

for i in range(N):
    TestClass[i, TestPredicted_Class[i]-1] = 1


TestClass = pd.DataFrame(TestClass)
TestClass.to_csv("C:/Users/LS/Downloads/Kaggle/MSKCC data/Submission Folder/Submission_1.csv")
####################################################################################################################
####################################################################################################################
####################################################################################################################
########################Unsupervised learning on Genes and Variations###############################################

#################Did not complete - not worth the effort

#import random
#training_variants=pd.read_csv("C:/Users/LS/Downloads/Kaggle/MSKCC data/training_variants.csv")
#test_variants=pd.read_csv("C:/Users/LS/Downloads/Kaggle/MSKCC data/test_variants.csv")
#
#training_variants = pd.concat([training_variants['Gene'],training_variants['Variation']],axis=1)
#test_variants = pd.concat([test_variants['Gene'],test_variants['Variation']],axis=1)
#training_variants.shape
#test_variants.shape
#list(training_variants)
#
#test_variants.shape
#list(test_variants)
#
#test_variants["Gene"][0]
#training_variants["Gene"][0]
#
#
#frames = [training_variants,test_variants]
#variant_unsupervised = pd.concat(frames,axis = 0,ignore_index = True)
#variant_unsupervised.shape
#variant_unsupervised["Gene"][0]
#
#
#list(variant_unsupervised)
#
#variant_unsupervised_gene = pd.DataFrame(variant_unsupervised["Gene"])
#list(variant_unsupervised_gene)
#variant_unsupervised_gene.shape
#
#encoder = LabelEncoder()
#encoder.fit(variant_unsupervised_gene["Gene"])
#variant_unsupervised_gene["Gene"] = encoder.transform(variant_unsupervised_gene["Gene"])
#list(variant_unsupervised_gene)
#variant_unsupervised_gene.shape
#
#N = len(variant_unsupervised_gene)
#K = len(np.unique(variant_unsupervised_gene["Gene"]))
#T = np.zeros((N, K))
#T.shape
#
#for i in range(0,N):    
#    T[i, variant_unsupervised_gene["Gene"][i]] = 1
#
#
#variant_unsupervised_variation = pd.DataFrame(variant_unsupervised["Variation"])
#list(variant_unsupervised_variation)
#variant_unsupervised_variation.shape
#
#encoder = LabelEncoder()
#encoder.fit(variant_unsupervised_variation["Variation"])
#variant_unsupervised_variation["Variation"] = encoder.transform(variant_unsupervised_variation["Variation"])
#list(variant_unsupervised_variation)
#variant_unsupervised_variation.shape
#variant_unsupervised_variation["Variation"][0]
#N = len(variant_unsupervised_variation)
#K = len(np.unique(variant_unsupervised_variation["Variation"]))
#T = np.zeros((N, K))
#T.shape
#
#for i in range(0,N):    
#    T[i, variant_unsupervised_gene["Gene"][i]] = 1
#
#
#
#
##8988/2 = 4494
##variant_unsupervised['Gene'][3320]
##variant_unsupervised['Gene'][3321]
#variant_unsupervised.to_csv("C:/Users/LS/Downloads/Kaggle/MSKCC data/Submission Folder/variant_unsupervised.csv")
#
#
#
#list(variant_unsupervised)
#variant_unsupervised.shape
#T.shape
#variant_unsupervised['Gene'].size








####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
training_variants=pd.read_csv("C:/Users/LS/Downloads/Kaggle/MSKCC data/training_variants.csv")
training_variants.shape
list(training_variants)



encoder = LabelEncoder()
encoder.fit(training_variants["Gene"])
training_variants["Gene"] = encoder.transform(training_variants["Gene"])
training_variants["Gene"].shape


N = len(training_variants)
K = len(pd.Series.unique(training_variants["Gene"]))

# turn Y into an indicator matrix for training
T = np.zeros((N, K))
for i in range(N):
    T[i, training_variants["Gene"][i]] = 1

T.shape
T = pd.DataFrame(T)
training_variants = pd.concat([training_variants,T],axis = 1)
list(training_variants)







test_variants=pd.read_csv("C:/Users/LS/Downloads/Kaggle/MSKCC data/test_variants.csv")
test_variants.shape
list(test_variants)

encoder = LabelEncoder()
encoder.fit(test_variants["Gene"])
test_variants["Gene"] = encoder.transform(test_variants["Gene"])
test_variants["Gene"].shape


N = len(test_variants)
K = len(pd.Series.unique(test_variants["Gene"]))

# turn Y into an indicator matrix for training
T = np.zeros((N, K))
for i in range(N):
    T[i, test_variants["Gene"][i]] = 1

T.shape
T = pd.DataFrame(T)
test_variants = pd.concat([test_variants,T],axis = 1)
list(test_variants)



####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
######################################Processing the text field ####################################################


test_text=pd.read_csv("C:/Users/LS/Downloads/Kaggle/MSKCC data/test.csv",encoding='cp1252')
training_text=pd.read_csv("C:/Users/LS/Downloads/Kaggle/MSKCC data/training.csv",encoding='cp1252')

list(test_text)
list(training_text)

test_text = pd.DataFrame(test_text[" Text"])
training_text = pd.concat([training_text[" Text"], training_text["Class"]],axis = 1)

training_text.to_csv("C:/Users/LS/Downloads/Kaggle/MSKCC data/training.csv")
test_text.to_csv("C:/Users/LS/Downloads/Kaggle/MSKCC data/test.csv")

test_text[" Text"][0]
test_text.shape
training_text[" Text"][0]
training_text.shape



consolidated = pd.concat([training_text,test_text],axis = 0,ignore_index = True)
list(consolidated)
consolidated.shape
consolidated[" Text"][3321]
consolidated["Class"][8988]
consolidated.to_csv("C:/Users/LS/Downloads/Kaggle/MSKCC data/consolidated.csv")



####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
######################################Machine learning on Text to Term Document#####################################

split_Train=pd.read_csv("C:/Users/LS/Downloads/Kaggle/MSKCC data/split_Train.csv",encoding='cp1252')
split_Test=pd.read_csv("C:/Users/LS/Downloads/Kaggle/MSKCC data/split_Test.csv",encoding='cp1252')
split_Train.shape
split_Test.shape
#split_Train_X = split_Train.ix[:,1:3425]
split_Train_X = split_Train.ix[:,1:3425]
split_Test_X = split_Test.ix[:,1:3426]
split_Test_X.shape
#split_Train_X.to_csv("C:/Users/LS/Downloads/Kaggle/MSKCC data/split_Train_X.csv")
split_Test_X.to_csv("C:/Users/LS/Downloads/Kaggle/MSKCC data/split_Test_X.csv")



split_Train_Y = split_Train["Class"]
split_Train_X.shape
split_Train_X = split_Train_X.as_matrix()
split_Train_Y = split_Train_Y.as_matrix()
split_Test_X = split_Test_X.as_matrix()
#split_Train_X = split_Train_X.T

N = len(split_Train_X)
K = len(np.unique(split_Train_Y))
T = np.zeros((N, K))
for i in range(N):
    T[i, split_Train_Y[i]-1] = 1

T.shape

def Model():
    model = Sequential()
    model.add(Dense(9, input_dim=3424, activation='sigmoid'))
    model.add(Dense(9, activation='sigmoid'))
#    model.add(Dense(9, activation='sigmoid'))
    model.add(Dense(9, activation='softmax'))
    model.compile(optimizer='adam', metrics = ['accuracy'], loss='categorical_crossentropy')
    return model






model = Model()
#model.fit(split_Train_X,T, epochs = 10000,batch_size = 10)
model.fit(split_Train_X,T, epochs = 5000,batch_size = 10)


score = model.evaluate(split_Train_X,T)
Ypredicted = model.predict(split_Test_X) 
#Ypredicted = pd.DataFrame(Ypredicted)
#Ypredicted.to_csv("C:/Users/LS/Downloads/Kaggle/MSKCC data/Ypredicted.csv")

Ypredicted = np.squeeze(np.asarray(Ypredicted))
Class = np.argmax(Ypredicted,axis = 1)+1
#np.unique(Class)
Class = pd.DataFrame(Class)
Class = Class.as_matrix()

Class.shape
N = len(Class)
K = len(np.unique(Class))
PredictedClass = np.zeros((N, K))

for i in range(len(Class)):
    PredictedClass[i, Class[i]-1] = 1

PredictedClass = pd.DataFrame(PredictedClass) 

PredictedClass.to_csv("C:/Users/LS/Downloads/Kaggle/MSKCC data/PredictedClass.csv")



#model.metrics_names[0]
#score[0]
#estimators = []
#estimators.append(('standardize', StandardScaler()))
#estimators.append(('mlp', KerasRegressor(build_fn=Model, epochs=100, batch_size = 10, verbose=0)))
#
#pipeline = Pipeline(estimators)
#kfold = KFold(n_splits=10)
##results = cross_val_score(pipeline, Xtrain, Ytrain, cv=kfold)
#
#results1 = cross_val_score(pipeline, Xtrain, Ytrain, cv=kfold)
#results1.mean()
#
#Model().evaluate(Xtest,Ytest)



















################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################
###########################################Splitting the train as well as text data for training and texting####################################################



################################################################################################################################################################
################################################################################################################################################################
################################################Segregating the training data set###############################################################################


training_variants = pd.read_excel("C:/Users/LS/Downloads/Kaggle/MSKCC data/Method 2/Data/Data.xlsx", sheetname = "training_variants", index = False)
#training_variants.shape
#list(training_variants)
split_Train=pd.read_csv("C:/Users/LS/Downloads/Kaggle/MSKCC data/Method 2/Data/split_Train.csv",encoding='cp1252')
split_Train.shape
split_Train["ID"]


#################################   Binary encoding - one hot encoding ('Gene' and 'Variation')#############################################################

Gene = np.unique(training_variants["Gene"])
Index = ["ID"]
Gene = np.concatenate((Index,Gene),axis = 0)
len(Gene)

Variation = np.unique(training_variants["Variation"])
Index = ["ID"]
Variation = np.concatenate((Index,Variation),axis = 0)
len(Variation)

#################################   Binary encoding - one hot encoding ('Gene' and 'Variation')#############################################################
encoder = LabelEncoder()
encoder.fit(training_variants["Gene"])
training_variants["Gene"] = encoder.transform(training_variants["Gene"])
training_variants["Gene"].shape


encoder = LabelEncoder()
encoder.fit(training_variants["Variation"])
training_variants["Variation"] = encoder.transform(training_variants["Variation"])
training_variants["Variation"].shape



#################################   Creating Dataframe for the dummy variables('Gene' and 'Variation')######################################################
N = len(training_variants)
Kg = len(np.unique(training_variants["Gene"]))
#len(Gene)
Tg = np.zeros((N, Kg))
for i in range(N):
    Tg[i, training_variants["Gene"][i]-1] = 1
Tg = pd.DataFrame(Tg)
Tg = pd.DataFrame(Tg).reset_index()
Tg.columns = [Gene]
list(Tg)
#Tg.drop(["ID"])
#training_variants["Gene"][0]

N = len(training_variants)
Kv = len(np.unique(training_variants["Variation"]))
Tv = np.zeros((N, Kv))
for i in range(N):
    Tv[i, training_variants["Variation"][i]-1] = 1
Tv = pd.DataFrame(Tv)
Tv.shape
Tv["ID"]
Tv = pd.DataFrame(Tv).reset_index()
Tv.columns = [Variation]
list(Tv)

##############unique_unique_train#######################
split_Train_Y = pd.DataFrame(split_Train["Class"])
list(split_Train_Y)
split_Train_Y.shape
split_Train_X = split_Train.ix[:,0:3425]
split_Train["ID"]
split_Train["Class"]

split_Train_X.shape
#split_Train_X = split_Train_X.as_matrix()

unique_unique_train = pd.concat([Tg,Tv],axis = 1)
#unique_unique_train.ix[:,3:]
unique_unique_train.shape
list(unique_unique_train)[263]
unique_unique_train[Index]
unique_unique_train = pd.concat([unique_unique_train, split_Train_X], axis =1)
unique_unique_train = unique_unique_train.drop("ID", axis = 1)
unique_unique_train.shape

##############NA_unique_test#######################
NA_unique_train = pd.DataFrame(Tv)
NA_unique_train = pd.concat([NA_unique_train, split_Train_X], axis = 1)
NA_unique_train["ID"]
NA_unique_train = NA_unique_train.drop("ID", axis = 1)
NA_unique_train.shape

##############unique_NA_test#######################
unique_NA_train = pd.DataFrame(Tg)
unique_NA_train = pd.concat([unique_NA_train, split_Train_X], axis = 1) 
unique_NA_train = unique_NA_train.drop("ID", axis = 1)
unique_NA_train.shape

##############NA_NA_test###########################
NA_NA_train = pd.DataFrame(split_Train_X)
NA_NA_train = NA_NA_train.drop("ID", axis = 1)
NA_NA_train.shape



################################################################################################################################################################
################################################################################################################################################################
################################################Segregating the testing data set################################################################################

split_Test=pd.read_csv("C:/Users/LS/Downloads/Kaggle/MSKCC data/Method 2/Data/split_Test.csv",encoding='cp1252')
split_Test['ID']

##############unique_unique_test#######################
unique_unique = pd.read_excel("C:/Users/LS/Downloads/Kaggle/MSKCC data/Method 2/Data/Data.xlsx", sheetname = "unique_unique")
list(unique_unique)


encoder = LabelEncoder()
encoder.fit(unique_unique["Gene"])
unique_unique["Gene"] = encoder.transform(unique_unique["Gene"])
unique_unique["Gene"].shape


encoder = LabelEncoder()
encoder.fit(unique_unique["Variation"])
unique_unique["Variation"] = encoder.transform(unique_unique["Variation"])
unique_unique["Variation"].shape




N = len(unique_unique)
Kg = len(np.unique(training_variants["Gene"]))
#len(Gene)
Tg = np.zeros((N, Kg))
for i in range(N):
    Tg[i, unique_unique["Gene"][i]-1] = 1
Tg = pd.DataFrame(Tg)
Tg = pd.DataFrame(Tg).reset_index()
Tg.columns = [Gene]
len(list(Tg))
Tg["ID"] = unique_unique["ID"]

N = len(unique_unique)
Kv = len(np.unique(training_variants["Variation"]))
Tv = np.zeros((N, Kv))
for i in range(N):
    Tv[i, unique_unique["Variation"][i]-1] = 1
Tv = pd.DataFrame(Tv)
Tv.shape
#Tv["ID"]
Tv = pd.DataFrame(Tv).reset_index()
Tv.columns = [Variation]
Tv["ID"] = unique_unique["ID"]

list(Tv)



unique_unique['ID']
unique_unique_test = pd.merge(Tv,Tg, on = 'ID')
unique_unique_test = pd.merge(unique_unique_test,split_Test, on = 'ID')
unique_unique_ID = pd.DataFrame(unique_unique_test['ID'])
unique_unique_test = unique_unique_test.drop("ID", axis = 1)
unique_unique_test.shape
#unique_unique_test['ID']

##############NA_unique_test#######################
NA_unique = pd.read_excel("C:/Users/LS/Downloads/Kaggle/MSKCC data/Method 2/Data/Data.xlsx", sheetname = "NA_unique")
list(NA_unique)
#encoder = LabelEncoder()
#encoder.fit(unique_unique["Gene"])
#unique_unique["Gene"] = encoder.transform(unique_unique["Gene"])
#unique_unique["Gene"].shape

encoder = LabelEncoder()
encoder.fit(NA_unique["Variation"])
NA_unique["Variation"] = encoder.transform(NA_unique["Variation"])
NA_unique["Variation"].shape




#N = len(NA_unique)
#Kg = len(np.unique(training_variants["Gene"]))
##len(Gene)
#Tg = np.zeros((N, Kg))
#for i in range(N):
#    Tg[i, NA_unique["Gene"][i]-1] = 1
#Tg = pd.DataFrame(Tg)
#Tg = pd.DataFrame(Tg).reset_index()
#Tg.columns = [Gene]
#list(Tg)


N = len(NA_unique)
Kv = len(np.unique(training_variants["Variation"]))
Tv = np.zeros((N, Kv))
for i in range(N):
    Tv[i, NA_unique["Variation"][i]-1] = 1
Tv = pd.DataFrame(Tv)
Tv.shape
Tv = pd.DataFrame(Tv).reset_index()
Tv.columns = [Variation]
Tv["ID"] = NA_unique["ID"]

list(Tv)



#NA_unique['ID']
NA_unique_test = pd.DataFrame(Tv)
NA_unique_test = pd.merge(NA_unique_test,split_Test, on = 'ID')
NA_unique_test['ID']
NA_unique_ID = pd.DataFrame(NA_unique_test["ID"])
NA_unique_test = NA_unique_test.drop("ID", axis = 1)
NA_unique_test.shape
#NA_unique_test['ID']

##############unique_NA_test#######################
unique_NA = pd.read_excel("C:/Users/LS/Downloads/Kaggle/MSKCC data/Method 2/Data/Data.xlsx", sheetname = "unique_NA")

encoder = LabelEncoder()
encoder.fit(unique_NA["Gene"])
unique_NA["Gene"] = encoder.transform(unique_NA["Gene"])
unique_NA["Gene"].shape

#encoder = LabelEncoder()
#encoder.fit(unique_NA["Variation"])
#unique_NA["Variation"] = encoder.transform(unique_NA["Variation"])
#unique_NA["Variation"].shape




N = len(unique_NA)
Kg = len(np.unique(training_variants["Gene"]))
#len(Gene)
Tg = np.zeros((N, Kg))
for i in range(N):
    Tg[i, unique_NA["Gene"][i]-1] = 1
Tg = pd.DataFrame(Tg)
Tg = pd.DataFrame(Tg).reset_index()
Tg.columns = [Gene]
Tg["ID"] = unique_NA["ID"]
list(Tg)


#N = len(unique_NA)
#Kv = len(np.unique(training_variants["Variation"]))
#Tv = np.zeros((N, Kv))
#for i in range(N):
#    Tv[i, unique_NA["Variation"][i]-1] = 1
#Tv = pd.DataFrame(Tv)
#Tv.shape
##Tv["ID"]
#Tv = pd.DataFrame(Tv).reset_index()
#Tv.columns = [Variation]
#list(Tv)


unique_NA_test = pd.DataFrame(Tg)
unique_NA_test = pd.merge(unique_NA_test,split_Test, on = 'ID')
unique_NA_test["ID"]
unique_NA_ID = pd.DataFrame(unique_NA_test['ID'])
unique_NA_test = unique_NA_test.drop("ID", axis = 1)

unique_NA_test.shape



##############NA_NA_test###########################
NA_NA = pd.read_excel("C:/Users/LS/Downloads/Kaggle/MSKCC data/Method 2/Data/Data.xlsx", sheetname = "NA_NA")
NA_NA = pd.DataFrame(NA_NA['ID'])
NA_NA['ID']
list(NA_NA)
NA_NA_test = pd.merge(NA_NA, split_Test, on='ID')
NA_NA_ID = pd.DataFrame(NA_NA_test['ID'])
NA_NA_test = NA_NA_test.drop("ID", axis = 1)

NA_NA_test.shape
NA_NA_test['ID']




################################################################################################################################################################
################################################################################################################################################################
############################################################## Model creation ##################################################################################


############################# unique_unique #############################
#NA_unique_train["ID"]
 split_Train_Y.shape

def Model():
    model = Sequential()
    model.add(Dense(9, input_dim=6684, activation='sigmoid'))
    model.add(Dense(9, activation='sigmoid'))
#    model.add(Dense(9, activation='sigmoid'))
    model.add(Dense(9, activation='softmax'))
    model.compile(optimizer='adam', metrics = ['accuracy'], loss='categorical_crossentropy')
    return model

split_Train_Y = split_Train_Y.as_matrix() 
N = len(split_Train_Y)
K = len(np.unique(split_Train_Y))
Ty = np.zeros((N, K))
for i in range(N):
    Ty[i, split_Train_Y[i]-1] = 1

Ty.shape

#unique_unique_train = unique_unique_train.as_matrix() 
#unique_unique_test = unique_unique_test.as_matrix() 


model = Model()
#model.fit(split_Train_X,T, epochs = 10000,batch_size = 10)
model.fit(unique_unique_train,Ty, epochs = 5000,batch_size = 10)
score = model.evaluate(unique_unique_train,split_train_Y)
Y_unique_unique = model.predict(unique_unique_test) 





############################# NA_unique #############################
#NA_unique_train["ID"]
NA_unique_train.shape
split_Train_Y.shape

def Model():
    model = Sequential()
    model.add(Dense(9, input_dim=6420, activation='sigmoid'))
    model.add(Dense(9, activation='sigmoid'))
#    model.add(Dense(9, activation='sigmoid'))
    model.add(Dense(9, activation='softmax'))
    model.compile(optimizer='adam', metrics = ['accuracy'], loss='categorical_crossentropy')
    return model

#split_Train_Y = split_Train_Y.as_matrix() 
N = len(split_Train_Y)
K = len(np.unique(split_Train_Y))
Ty = np.zeros((N, K))
for i in range(N):
    Ty[i, split_Train_Y[i]-1] = 1

Ty.shape

#NA_unique_train = NA_unique_train.as_matrix() 


model = Model()
#model.fit(split_Train_X,T, epochs = 10000,batch_size = 10)
model.fit(NA_unique_train,Ty, epochs = 5000,batch_size = 10)
score = model.evaluate(NA_unique_train,Ty)
#NA_unique_test = NA_unique_test.as_matrix() 
Y_NA_unique = model.predict(NA_unique_test) 





############################# unique_NA #############################
unique_NA_train.shape
split_Train_Y.shape

def Model():
    model = Sequential()
    model.add(Dense(9, input_dim=3688, activation='sigmoid'))
    model.add(Dense(9, activation='sigmoid'))
#    model.add(Dense(9, activation='sigmoid'))
    model.add(Dense(9, activation='softmax'))
    model.compile(optimizer='adam', metrics = ['accuracy'], loss='categorical_crossentropy')
    return model

#split_Train_Y = split_Train_Y.as_matrix() 
N = len(split_Train_Y)
K = len(np.unique(split_Train_Y))
Ty = np.zeros((N, K))
for i in range(N):
    Ty[i, split_Train_Y[i]-1] = 1

Ty.shape

#unique_NA_train = unique_NA_train.as_matrix() 


model = Model()
model.fit(unique_NA_train,Ty, epochs = 5000,batch_size = 10)

score = model.evaluate(unique_NA_train,Ty)
#score = [0.44324002951356667, 0.85215296577667843]
#unique_NA_test = unique_NA_test.as_matrix()
Y_unique_NA = model.predict(unique_NA_test) 




############################# NA_NA #############################
NA_NA_train.shape
#NA_NA_ID
split_Train_Y.shape

def Model():
    model = Sequential()
    model.add(Dense(9, input_dim=3424, activation='sigmoid'))
    model.add(Dense(9, activation='sigmoid'))
#    model.add(Dense(9, activation='sigmoid'))
    model.add(Dense(9, activation='softmax'))
    model.compile(optimizer='adam', metrics = ['accuracy'], loss='categorical_crossentropy')
    return model

#split_Train_Y = split_Train_Y.as_matrix() 
N = len(split_Train_Y)
K = len(np.unique(split_Train_Y))
Ty = np.zeros((N, K))
for i in range(N):
    Ty[i, split_Train_Y[i]-1] = 1

Ty.shape

#NA_NA_train = NA_NA_train.as_matrix() 
#NA_NA_test = NA_NA_test.as_matrix() 

model = Model()
#model.fit(split_Train_X,T, epochs = 10000,batch_size = 10)
model.fit(NA_NA_train,Ty, epochs = 5000,batch_size = 10)
score = model.evaluate(NA_NA_train,Ty)
Y_NA_NA = model.predict(NA_NA_test) 
NA_NA_test.shape
NA_NA_train.shape


Y_NA_NA.shape
Y_NA_NA = pd.DataFrame(Y_NA_NA)
NA_NA_ID.shape
NA_NA_ID = pd.DataFrame(NA_NA_ID)
Y_NA_NA = pd.concat([Y_NA_NA,NA_NA_ID],axis = 1)

Y_NA_unique.shape
NA_unique_ID.shape
Y_NA_unique = pd.DataFrame(Y_NA_unique)
NA_unique_ID = pd.DataFrame(NA_unique_ID)
Y_NA_unique = pd.concat([Y_NA_unique,NA_unique_ID],axis = 1)

Y_unique_NA.shape
unique_NA_ID.shape
Y_unique_NA = pd.DataFrame(Y_unique_NA)
unique_NA_ID = pd.DataFrame(unique_NA_ID)
Y_unique_NA = pd.concat([Y_unique_NA,unique_NA_ID],axis = 1)


Y_unique_unique.shape
unique_unique_ID.shape
Y_unique_unique = pd.DataFrame(Y_unique_unique)
unique_unique_ID = pd.DataFrame(unique_unique_ID)
Y_unique_unique = pd.concat([Y_unique_unique,unique_unique_ID],axis = 1)



submission = pd.concat([Y_unique_unique,Y_NA_unique,Y_unique_NA,Y_NA_NA],axis = 0)
submission.shape
list(submission)
submission["ID"][10]
submission = submission.sort_values("ID")
submission = pd.concat([submission["ID"],submission[0],submission[1],submission[2],submission[3],submission[4],submission[5],submission[6],submission[7],submission[8]],axis=1)

submission.to_csv("C:/Users/LS/Downloads/Kaggle/MSKCC data/Submission Folder/Submission_3.csv", index = False)





#Class = np.squeeze(np.asarray(submission))
Class = pd.concat([submission[0],submission[1],submission[2],submission[3],submission[4],submission[5],submission[6],submission[7],submission[8]],axis=1)
Class = Class.as_matrix() 
Class = (np.argmax(Class,axis = 1))+1
#np.unique(Class)
Class = pd.DataFrame(Class)
#Class = Class.as_matrix()
#Class.shape
#N = len(Class)
#K = 9
#Class.shape
#PredictedClass = np.zeros((N, K))
#
#for i in range(len(Class)):
#    PredictedClass[i, Class[i]-1] = 1

PredictedClass = pd.DataFrame(PredictedClass) 

Class.to_csv("C:/Users/LS/Downloads/Kaggle/MSKCC data/Submission Folder/PredictedClass_3.csv")
Phase_1 = pd.read_csv("C:/Users/LS/Downloads/Kaggle/MSKCC data/Submission Folder/stage1_solution_filtered.csv")
#Class = pd.concat([submission[0],submission[1],submission[2],submission[3],submission[4],submission[5],submission[6],submission[7],submission[8]],axis=1)
Phase_1.shape
list(Phase_1)
Phase_1_ID = pd.DataFrame(Phase_1["ID"])
Phase_1 = Phase_1.as_matrix() 
Phase_1 = (np.argmax(Phase_1[:,1:10],axis = 1))+1
Phase_1 = pd.DataFrame(Phase_1)
Phase_1 =pd.concat([Phase_1_ID,Phase_1],axis = 1)
Phase_1.to_csv("C:/Users/LS/Downloads/Kaggle/MSKCC data/Submission Folder/Phase_1.csv")




