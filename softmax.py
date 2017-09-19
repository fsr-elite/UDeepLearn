"""Softmax."""

scores = [3.0, 1.0, 0.2]

import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    pass  # TODO: Compute and return softmax(x)

    exp_x = np.exp(x)
    sum_x = np.sum(exp_x, axis=0)
    softmax = exp_x/sum_x
 
    return softmax

print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])
softmax_scores = softmax(scores)

plt.plot(x, softmax_scores.T, linewidth=2)
plt.plot(x, np.sum(softmax_scores, axis=0), '--', linewidth=3)

plt.show()
print("done here")
