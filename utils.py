import numpy as np
from sklearn.metrics import fbeta_score

def optimise_f2_thresholds(y, p, num_classes=17, verbose=True, resolution=100):
  def mf(x):
    p2 = np.zeros_like(p)
    for i in range(num_classes):
      p2[:, i] = (p[:, i] > x[i]).astype(np.int)
    score = fbeta_score(y, p2, beta=2, average='samples')
    return score

  x = [0.2]*num_classes
  for i in range(num_classes):
    best_i2 = 0
    best_score = 0
    for i2 in range(resolution):
      i2 /= resolution
      x[i] = i2
      score = mf(x)
      if score > best_score:
        best_i2 = i2
        best_score = score
    x[i] = best_i2
    if verbose:
      print(i, best_i2, best_score)

  return x