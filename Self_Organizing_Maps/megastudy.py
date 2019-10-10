# Mega Case Study

# Identify the frauds

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len=15)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o','s']
colors = ['r','g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[int(y[i])],
         markeredgecolor = colors[int(y[i])],
         markerfacecolor = 'None',
         markeredgewidth = 2)
show()

# Find the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(6,1)], mappings[(4,1)]), axis=0)
frauds = sc.inverse_transform(frauds)

customers = dataset.iloc[:, 1:].values

# Creating the dependent variable

fraudulent = np.array([int((a in list(frauds[:,0]))) for a in list(sc.inverse_transform(X)[:,0])])

# Creating the ANN
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

classifier = Sequential()
classifier.add(Dense(output_dim = 2, init = 'uniform', activation = 'relu', input_dim = 15))

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting classifier to the Training set

classifier.fit(customers, fraudulent, batch_size = 1, nb_epoch = 2)

# Predicting the probability of fraud
y_pred = classifier.predict(customers)
y_pred = y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis=1)
y_pred = y_pred[y_pred[:, 1].argsort()]


