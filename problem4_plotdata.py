# Problem 4
# Convince yourself that the data is still good after shuffling!

#%% import
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

#%% Load the "letter", a NP array of [number-of-letters, 28, 28]
fname = "notMNIST.pickle"
with open(fname, 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    pickle_data = pickle.load(f)

train_dataset = pickle_data['train_dataset']
train_labels = pickle_data['train_labels']
valid_dataset = pickle_data['valid_dataset']
valid_labels = pickle_data['valid_labels']
test_dataset = pickle_data['test_dataset']
test_labels = pickle_data['test_labels']

#%%
# do a simple check. we'll only:
# 1) check that each class is in each dataset and that they are evenly distributed
# 2) check that each label has a value (first in dataset) that loks good

    #%%
# 1) check that each class is in each dataset and that they are evenly distributed
# Thus we're curious about how many for each and maybe some information about each.

unique_labels = np.unique(train_labels)

for i, unique_label in enumerate(unique_labels):
    # check the number in each class
    inds = np.nonzero(train_labels == i)
    print("Label", unique_label, "has", train_labels[inds].size, "elements.")