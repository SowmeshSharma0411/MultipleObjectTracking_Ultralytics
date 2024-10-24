import json
import pywt
import numpy as np
from sklearn.mixture import GaussianMixture
import pickle

data = {}
with open('rash_history.json', 'r') as f:
    data = json.load(f)

# Step 2: Function to extract the last 6 features
def extract_last_6_features(data):
    extracted_data = {}
    
    for key, time_series in data.items():
        filtered_series = []
        for timestep in time_series:
            filtered_timestep = timestep[2:]
            filtered_series.append(filtered_timestep)
        
        extracted_data[key] = filtered_series
        
    return extracted_data

def apply_dwt(rash_histiry_data, wavelet = 'db1'):
    dwt_features = {}
    
    for key, series in rash_histiry_data.items():
        series_dwt = []
        for timestep in series:
            # Apply DWT on each time step (6 features per step)
            coeffs = pywt.dwt(timestep, wavelet)  # Decompose using DWT
            print(coeffs)

            # Concatenate approximation and detail coefficients
            dwt_step = np.concatenate((coeffs[0], coeffs[1]))
            series_dwt.append(dwt_step)
        
        dwt_features[key] = series_dwt  # Store the transformed series
    
    return dwt_features

def cluster_dwt_features_gmm(dwt_data, max_clusters = 10):
    flattened_data = []
    keys = []

    # Flatten the DWT-transformed data for clustering
    for key, series in dwt_data.items():
        for timestep in series:
            flattened_data.append(timestep)
            keys.append(key)  # Track which time series each timestep belongs to

    flattened_data = np.array(flattened_data)
    
    # Apply Gaussian Mixture Model for clustering
    gmm = GaussianMixture(n_components=num_clusters, random_state=42)
    labels = gmm.fit_predict(flattened_data)
    
    # Map labels back to the original time series
    clustered_labels = {key: [] for key in dwt_data.keys()}
    for i, key in enumerate(keys):
        clustered_labels[key].append(labels[i])
    
    return clustered_labels

def label_time_series(clustered_labels, original_data):
    labeled_data = {}

    for key in original_data.keys():
        # Assign the cluster label to the entire time series: this method might have to change need a better repersentation
        majority_label = max(set(clustered_labels[key]), key=clustered_labels[key].count)
        labeled_data[key] = majority_label
    
    return labeled_data


filtered_data = extract_last_6_features(data)
dwt_transformed_data = apply_dwt(filtered_data)
gmm_clustered_labels = cluster_dwt_features_gmm(dwt_transformed_data)
gmm_labeled_time_series = label_time_series(gmm_clustered_labels, filtered_data)
file_save_path = 'track_history1_labelled.pkl'

print(gmm_clustered_labels)

with open(file_save_path, 'wb') as f:
    pickle.dump(gmm_labeled_time_series, f)
