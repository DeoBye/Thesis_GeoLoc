import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_distances, haversine_distances
import torch
import gstools as gs
import pickle
import sys

def fit_theoretical_semivariogram(bin_center, gamma, nugget=True):

    fit_model = gs.Stable(dim=2)
    fit_model.fit_variogram(bin_center, gamma, nugget=nugget)

    return fit_model

data = np.load('osv5mEmbtest.npz')


embeddings = data['embeddings']
locations = data['coordinates']

# rdm = np.random.RandomState(seed=42)
# indices = rdm.choice(embeddings.shape[0], size=100000, replace=False)

# embeddings = embeddings[indices]
# locations = locations[indices]


loc = locations * np.array([np.pi/2, np.pi]) / np.array([90, 180])

plt.figure()
plt.scatter(locations[:, 1], locations[:, 0], s=0.01)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Data distribution")
plt.savefig("osv5mtest_distribution.jpeg", dpi=300)

cosine_dist = cosine_distances(embeddings)
print(sys.getsizeof(cosine_dist))
gc_dist = haversine_distances(loc, loc)
print(sys.getsizeof(gc_dist))
# np.savez("/data/roof/osv5m/osv5m_cos_gc.npz", cosine_dist=cosine_dist, gc_dist=gc_dist)


means = []
variances = []
stds = []
step_size = 0.001


# compute semivariogram
for i in np.arange(0, np.pi, step_size):
    idx = (gc_dist > i) & (gc_dist <= i + step_size)
    print("Distance lag: {} to {}".format(i, i+step_size))
    mean, var, std = np.mean(cosine_dist[idx]), np.var(cosine_dist[idx]), np.std(cosine_dist[idx])
    means.append(mean)
    variances.append(var)
    stds.append(std)
    
xs = np.arange(0, np.pi, step_size) + step_size / 2
means = np.array(means)
variances = np.array(variances)
stds = np.array(stds)

np.savez("osv5mtest_stat.npz", xs=xs, means=means, variances=variances, stds=stds)

theoretical_semivariogram = fit_theoretical_semivariogram(xs, means)
with open('osv5m_fitted.pkl', 'wb') as file:
    pickle.dump(theoretical_semivariogram, file)

plt.figure(figsize=(15, 15))
plt.ylim([0, 0.4])
plt.scatter(gc_dist.flatten(), cosine_dist.flatten(), s=0.0001)
plt.plot(xs, means, 'r-', marker="x")
plt.fill_between(xs, means - np.sqrt(variances), means + np.sqrt(variances), color="orange", alpha=0.2)
plt.xlabel("Geographic Distance(radians)")
plt.ylabel("Cosine Distance")
plt.title("Semivariogram")
plt.savefig("osv5m_semivariogram.png", dpi=300)
