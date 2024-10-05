import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import json
import random

data = {}
with open('track_history2.json', 'r') as f:
    data = json.load(f)

representative_trajectories = []

# Extract the last 4 features: deltaX, deltaY, InstVelocity, and NearbyObjects
for obj, traj_list in data.items():
    traj_array = np.array(traj_list)
    traj_array = traj_array[:, -4:]  # We are considering the last 4 features for each trajectory
    # Compute the mean trajectory
    rep_traj = np.mean(traj_array, axis=0)
    representative_trajectories.append(rep_traj)

scaler = StandardScaler()
normalized_trajectories = scaler.fit_transform(representative_trajectories)

np.random.seed(42)
random.seed(42)

# Gaussian Mixture Model clustering
n_clusters = 5
gmm = GaussianMixture(n_components=n_clusters, random_state=42)
labels = gmm.fit_predict(normalized_trajectories)

# Domain knowledge-based classification thresholds
inst_velocity_threshold = 0.1  # Adjust based on your domain knowledge
delta_threshold = 1.0          #    Adjust for deltaX, deltaY
nearby_objects_threshold = 4   # Adjust based on number of nearby objects

# Function to classify based on rules
def classify_cluster(centroid):
    deltaX, deltaY, inst_velocity, nearby_objects = centroid
    if (inst_velocity > inst_velocity_threshold and 
        (deltaX > delta_threshold or deltaY > delta_threshold)):
        return 'rash'
    elif (inst_velocity > inst_velocity_threshold and nearby_objects > nearby_objects_threshold):
        return 'rash'
    else:
        return 'non-rash'

# Apply the rule-based classification to each cluster's centroid
centroids = gmm.means_  # These are the centroids from GMM

for i, centroid in enumerate(centroids):
    classification = classify_cluster(centroid)
    print(f"Cluster {i} is classified as: {classification}")

# Classify each object based on its cluster label
for i, obj in enumerate(data.keys()):
    cluster_label = labels[i]
    centroid_classification = classify_cluster(centroids[cluster_label])
    print(f"Object {obj} is classified as: {centroid_classification}")

# Visualizing results
pca = PCA(n_components=2)
pca_trajectories = pca.fit_transform(normalized_trajectories)

# Train the GMM model on the reduced data for visualization
gmm_reduced = GaussianMixture(n_components=n_clusters, random_state=42)
labels_reduced = gmm_reduced.fit_predict(pca_trajectories)

# Generate grid for contour plot
x_min, x_max = pca_trajectories[:, 0].min() - 1, pca_trajectories[:, 0].max() + 1
y_min, y_max = pca_trajectories[:, 1].min() - 1, pca_trajectories[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# Predict on the grid
Z = gmm_reduced.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the results
plt.figure()
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(pca_trajectories[:, 0], pca_trajectories[:, 1], c=labels_reduced, s=40, cmap='viridis')
plt.title('Clusters and Decision Boundaries')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()