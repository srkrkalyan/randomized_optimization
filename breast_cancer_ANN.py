#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 04 19:52:13 2019

@author: kalyantulabandu
"""

"""
Spyder Editor

This is a temporary script file.
"""
# Importing libraries
import pandas as pd
import numpy as np
import mlrose

# Importing the dataset
dataset = pd.read_csv('breast_cancer_data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Encoding categorical data
# No categorical data, so no encoding needed

# splitting X & y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Initialize neural network object and fit object

np.random.seed(3)

nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [50], activation = 'tanh', \
                                 algorithm = 'simulated_annealing', max_iters = 1500, \
                                 bias = True, is_classifier = True, learning_rate = 0.01, \
                                 early_stopping = True, clip_max = 5, max_attempts = 50,
                                 schedule = mlrose.GeomDecay(init_temp=2, decay=0.8, min_temp=0.001))

nn_model1.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

# Predict labels for train set and assess accuracy
y_train_pred = nn_model1.predict(X_train)

y_train_accuracy = accuracy_score(y_train, y_train_pred)

print(y_train_accuracy)


# Predict labels for test set and assess accuracy
y_test_pred = nn_model1.predict(X_test)

y_test_accuracy = accuracy_score(y_test, y_test_pred)

print(y_test_accuracy)

nn_model1.fitted_weights
nn_model1.loss
nn_model1.predicted_probs[0]



'''

#Importing keras libraries and packages
#import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential()

# Add input layer and first hidden layer
classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu', input_dim =  5))

# Adding the second hidden layer
classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to training set
history = classifier.fit(X_train, y_train, validation_split = 0.33, batch_size = 10, nb_epoch = 120)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# comparing predictions and targets to compute accuracy
from sklearn import metrics
print("Artificial Neural Network - accuracy - Breast Cancer Prediction: ", round(metrics.accuracy_score(y_test,y_pred),6))

# list all data in history
print(history.history.keys())


#plotting learning curves for neural network
import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('ANN Accuracy - Breast Cancer Data Set')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('ANN Loss - Breast Cancer Data Set')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''

