import matplotlib.pyplot as plt
import numpy as np
# from tslearn.clustering import TimeSeriesKMeans
# from tslearn.metrics import dtw
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import json
import random

data = {}
with open('track_history.json', 'r') as f:
    data = json.load(f)

representative_trajectories = []

for obj, traj_list in data.items():
    traj_array = np.array(traj_list)

    traj_array = traj_array[:, -4:]

    # Compute the mean trajectory
    rep_traj = np.mean(traj_array, axis=0)

    representative_trajectories.append(rep_traj)

scaler = StandardScaler()
normalized_trajectories = scaler.fit_transform(representative_trajectories)

print(representative_trajectories)
print(normalized_trajectories)

np.random.seed(42)
random.seed(42)

# DBA-KMeans clustering
n_clusters = 2
# model = TimeSeriesKMeans(n_clusters=n_clusters,
#                          metric="dtw", n_init=2, verbose=True)

# kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
gmm = GaussianMixture(n_components=n_clusters, random_state=42)

labels = gmm.fit_predict(normalized_trajectories)

for i, obj in enumerate(data.keys()):
    if labels[i] == 0:
        classification = 'rash'
    else:
        classification = 'non-rash'

    print(f"Object {obj} is classified as: {classification}")

# Visualizing results
pca = PCA(n_components=2)
pca_trajectories = pca.fit_transform(normalized_trajectories)

# Train the GMM model on the reduced data
gmm_reduced = GaussianMixture(n_components=n_clusters, random_state=42)
labels_reduced = gmm_reduced.fit_predict(pca_trajectories)

# Generate grid for contour plot
x_min, x_max = pca_trajectories[:, 0].min(
) - 1, pca_trajectories[:, 0].max() + 1
y_min, y_max = pca_trajectories[:, 1].min(
) - 1, pca_trajectories[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predict on the grid
Z = gmm_reduced.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the results
plt.figure()
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(pca_trajectories[:, 0], pca_trajectories[:,
            1], c=labels_reduced, s=40, cmap='viridis')
plt.title('Clusters and Decision Boundaries')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# for i in range(n_clusters):
#     plt.figure()
#     cluster_trajectories = normalized_trajectories[labels == i]
#     for traj in cluster_trajectories:
#         # Use scatter plot for better clarity
#         plt.plot(traj[0], traj[1], "o", alpha=0.3)
#     plt.xlabel('Feature 1 (Normalized)')
#     plt.ylabel('Feature 2 (Normalized)')
#     plt.title(f"Cluster {i} trajectories")
# plt.show()
