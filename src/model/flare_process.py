import numpy as np

def sample_poisson_flares(rate, T, rng=None):
rng = np.random.default_rng() if rng is None else rng
t = 0.0
times = []
while t < T:
w = rng.exponential(1.0/rate)
t += w
if t < T:
times.append(t)
return np.array(times)
