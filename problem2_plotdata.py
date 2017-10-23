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

# Load the "letter", a NP array of [number-of-letters, 28, 28]
fname = "./notMNIST_large/A.pickle"
fid = open(fname, mode = "rb")
letter = pickle.load(fid)
fid.close()

#%%
max_ind = letter.shape[0]
random_inds = np.random.randint(0, max_ind, 16)

#%%
plt.subplots_adjust(wspace = 1, hspace = 1)
plt.ioff()
plt.subplot(4, 4, 1)
for random_inds_index, letter_index in enumerate(random_inds):
    plt.subplot(4, 4, random_inds_index + 1)
    plt.imshow(letter[letter_index, :, :])
    plt.title(str(letter_index))

#%%
plt.show()