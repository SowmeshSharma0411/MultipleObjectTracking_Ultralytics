import matplotlib.pyplot as plt
import numpy as np
# from tslearn.clustering import TimeSeriesKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tslearn.metrics import dtw
import json
import random

data = {}
with open('track_history.json', 'r') as f:
    data = json.load(f)

representative_trajectories = []

for obj, traj_list in data.items():
    traj_array = np.array(traj_list)

    traj_array = traj_array[:, -4:-1]

    # Compute the mean trajectory
    rep_traj = np.mean(traj_array, axis=0)

    representative_trajectories.append(rep_traj)

scaler = StandardScaler()
normalized_trajectories = scaler.fit_transform(representative_trajectories)

print(normalized_trajectories)

np.random.seed(42)
random.seed(42)

# DBA-KMeans clustering
n_clusters = 2
# model = TimeSeriesKMeans(n_clusters=n_clusters,
#                          metric="dtw", n_init=2, verbose=True)

kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)

labels = kmeans.fit_predict(normalized_trajectories)

for i, obj in enumerate(data.keys()):
    print(
        f"Object {obj} is classified as: {'rash' if labels[i] == 0 else 'non-rash'}")

# Visualizing results
# for i in range(n_clusters):
#     plt.figure()
#     cluster_trajectories = representative_trajectories[labels == i]
#     print(cluster_trajectories)
#     for traj in cluster_trajectories:
#         #     # Assuming first 2 features for X, Y
#         plt.plot(traj[0], traj[1], "k-", alpha=0.3)
#     plt.xlabel('Change in X')
#     plt.ylabel('Change in Y')
#     plt.title(f"Cluster {i} trajectories")
# plt.show()
