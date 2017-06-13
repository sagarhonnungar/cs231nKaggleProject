def optimise_f2_thresholds(x_init, y, p, num_classes=17, verbose=True, resolution=100):
  def mf(x):
    p2 = np.zeros_like(p)
    for i in range(num_classes):
      p2[:, i] = (p[:, i] > x[i]).astype(np.int)
    score = fbeta_score(y, p2, beta=2, average='samples')
    return score

  x = list(x_init)
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

