"""Check conditioning of each letter."""
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle

def calc_params(letter):
    """calculate the parameters for the letter argument."""
    mean_letter = letter.mean()
    std_letter = letter.std()
    return mean_letter, std_letter

folder = './notMNIST_large/'

# use list comprehension to create a list of the pickle files
filelist = os.listdir(folder)
pickle_filelist = [filelist for filelist in filelist if filelist.endswith('.pickle')]

num_letters = len(pickle_filelist)
all_means = np.zeros([num_letters,1])
all_stds = np.zeros_like(all_means)

for index, pickle_file in enumerate(pickle_filelist):
    fullfile = os.path.join(folder, pickle_file)
    print("Processing ", fullfile)
    letter = pickle.load(open(fullfile, mode = "rb"))
    mean_letter, std_letter = calc_params(letter)
    all_means[index] = mean_letter
    all_stds[index] = std_letter

    print("   mean and std of " + pickle_file + " = ", all_means[index], all_stds[index])

plt.plot(all_means, 'b')
plt.plot(all_stds, 'r')
plt.grid()
plt.show()