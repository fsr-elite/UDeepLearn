import numpy as np

test = np.array([4, 5, 5, 6, 7])
inds = np.nonzero(test > 5)
test_inds = test[inds]
print(test_inds)
print("test_inds.shape()", test_inds.size)