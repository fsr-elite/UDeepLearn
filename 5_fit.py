5_fit.py#%% import
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

data_root = '.' # Change me to store data elsewhere

#%% Load the training datasets
fname = "notMNIST.pickle"
with open(fname, 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    pickle_data = pickle.load(f)

#%% Convert to individual variables to work with
# TODO: identify to do this through some "deal" type mechanism
train_dataset = pickle_data['train_dataset']
train_labels = pickle_data['train_labels']
valid_dataset = pickle_data['valid_dataset']
valid_labels = pickle_data['valid_labels']
test_dataset = pickle_data['test_dataset']
test_labels = pickle_data['test_labels']

#%% Put the training data into the correct format for sklearn
train_features = train_dataset.reshape((train_dataset.shape[0], -1))
valid_features = valid_dataset.reshape((valid_dataset.shape[0], -1))
test_features = test_dataset.reshape((test_dataset.shape[0], -1))

#%% Let's regress.
#print("Perform Logistic Regression")
#logistic = LogisticRegression(C=1e5)
#logistic.fit(train_features, train_labels)

pickle_file = os.path.join(data_root, 'notMNIST.pickle')

try:
  logistic_pickle = "logistic.pickle"
  f = open(pickle_file, 'wb')
  pickle.dump(logistic, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', logistic_pickle, ':', e)
  raise
