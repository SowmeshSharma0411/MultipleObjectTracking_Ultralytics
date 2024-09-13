from ultralytics import YOLO
import cv2
import numpy as np
import time
import os
import json
from scipy.spatial import cKDTree
from filterpy.kalman import KalmanFilter
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import torch
import gc
import threading
from sklearn.neighbors import NearestNeighbors

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# ViewTransformer class from the first snippet
class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

def init_kalman_filter():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    kf.R *= 10
    kf.P *= 1000
    return kf

def update_kalman(kf, x, y):
    kf.predict()
    kf.update(np.array([x, y]))
    return kf.x

def calculate_velocity(track, frame_interval, fps, speed_up_factor):
    velocities = []
    time_interval = frame_interval / fps / speed_up_factor  # Adjust for sped-up video

    for i in range(1, len(track)):
        prev_x, prev_y = track[i-1][:2]
        curr_x, curr_y = track[i][:2]
        
        dx = curr_x - prev_x
        dy = curr_y - prev_y
        distance = np.sqrt(dx**2 + dy**2)
        
        # Convert distance from pixels to meters (assuming 1 pixel = 0.1 meters, adjust as needed)
        distance_meters = distance * 0.1
        
        velocity = distance_meters / time_interval
        speed = velocity * 3.6  # Convert to km/h
        
        velocities.append((dx, dy, speed))
    
    return velocities

def estimate_average_spacing(all_objects, k=5):
    if len(all_objects) < 2:
        return 150
    
    centroids = np.array([[obj[0], obj[1]] for obj in all_objects])
    sizes = np.array([max(obj[2], obj[3]) for obj in all_objects])
    
    k = min(k, len(all_objects) - 1)
    
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(centroids)
    distances, _ = nbrs.kneighbors(centroids)
    
    avg_distances = np.median(distances[:, 1:], axis=1)
    adjusted_spacing = avg_distances + (sizes / 4)
    
    return np.median(adjusted_spacing)

def count_nearby_objects(current_object, all_objects, frame_width, frame_height, distance_threshold):
    if len(all_objects) < 2:
        return 0
    
    # Extract centroids and sizes
    centroids = np.array([obj[:2] for obj in all_objects])
    sizes = np.array([max(obj[2], obj[3]) for obj in all_objects])
    
    # Create a KD-Tree for efficient nearest neighbor search
    tree = cKDTree(centroids)
    
    # Find all neighbors within the distance threshold
    current_centroid = np.array(current_object[:2])
    neighbors = tree.query_ball_point(current_centroid, distance_threshold)
    
    # Count neighbors, excluding self
    nearby_count = len(neighbors) - 1
    
    # Adjust count based on object sizes
    for i in neighbors:
        if i != all_objects.index(current_object):
            # Calculate the actual distance between object centroids
            distance = np.linalg.norm(centroids[i] - current_centroid)
            # Adjust the count based on object sizes
            if distance < (sizes[i] + sizes[all_objects.index(current_object)]) / 2:
                nearby_count += 1

    return nearby_count

start = time.time()

# Load the YOLOv8 model
model = YOLO("best.pt")

# Open the video file
# video_path = "SampleVids\\bombay_trafficShortened.mp4"
video_path = "SampleVids\\traffic_vid2Shortened.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = {}
kalman_filters = {}
tracker = "bytetrack.yaml"

# Create new directory for output
new_track_dir_no = max([int(d[5:]) for d in os.listdir("runs/detect") if d.startswith("track")] + [1]) + 1
new_track_dir = f"track{new_track_dir_no}"
new_track_path = os.path.join("runs/detect", new_track_dir)
os.makedirs(new_track_path)

# Set up video writer
output_video_path = os.path.join(new_track_path, "output.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print(fps, width, height)

frame_stride = max(1, int(fps / 1.5))  # Adaptive frame stride
frame_count = 0

# Vehicle classes in COCO dataset
vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# Set up view transformer
SOURCE = np.array([[0, 0], [width, 0], [width, height], [0, height]])
TARGET_WIDTH, TARGET_HEIGHT = 25, 250
TARGET = np.array([[0, 0], [TARGET_WIDTH - 1, 0], [TARGET_WIDTH - 1, TARGET_HEIGHT - 1], [0, TARGET_HEIGHT - 1]])
view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

frame_to_calculate_distance = 360  
dynamic_radius = 150
alpha = 0.2  # Smoothing factor for dynamic radius

speed_up_factor = 5
speed_window = 3
speed_threshold = 150

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model.track(frame, persist=True, tracker=tracker)
        
        if not (results and results[0] and results[0].boxes):
            continue

        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().tolist()
        else:
            track_ids = []
        
        boxes = results[0].boxes.xywh
        classes = results[0].boxes.cls.tolist()
        
        annotated_frame = results[0].plot()

        if frame_count == frame_to_calculate_distance:
            curr_frame_objects = [(box[0].item(), box[1].item(), box[2].item(), box[3].item()) for box, cls in zip(boxes, classes) if int(cls) in vehicle_classes]
            dynamic_radius = alpha * estimate_average_spacing(curr_frame_objects) + (1 - alpha) * dynamic_radius
        
        if frame_count % frame_stride == 0:
            for box, track_id, cls in zip(boxes, track_ids, classes):
                if int(cls) in vehicle_classes:
                    x, y, w, h = box
                    if track_id not in track_history:
                        track_history[track_id] = []
                        kalman_filters[track_id] = init_kalman_filter()
                    
                    kalman_state = update_kalman(kalman_filters[track_id], x, y)
                    smoothed_x, smoothed_y = kalman_state[0], kalman_state[1]
                    
                    # Transform the point using view transformer
                    transformed_point = view_transformer.transform_points(np.array([[smoothed_x, smoothed_y]]))
                    transformed_x, transformed_y = transformed_point[0]
                    
                    track_history[track_id].append((float(transformed_x), float(transformed_y), float(w), float(h), int(cls)))

                      # Calculate and display speed if we have enough history
                    if len(track_history[track_id]) >= speed_window:
                        recent_track = track_history[track_id][-speed_window:]
                        velocities = calculate_velocity(recent_track, frame_stride, fps, speed_up_factor)
                        avg_speed = np.mean([v[2] for v in velocities])
                        
                        # Apply speed threshold
                        avg_speed = min(avg_speed, speed_threshold)
                        
                        # Display speed on the annotated frame
                        cv2.putText(annotated_frame, f"{avg_speed:.1f} km/h", (int(x), int(y) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        out.write(annotated_frame)
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

torch.cuda.empty_cache()
torch.cuda.reset_max_memory_allocated()
torch.cuda.reset_peak_memory_stats()
gc.collect()

def process_trajectory(track_id, track, all_tracks, frame_width, frame_height, vehicle_classes, frame_stride, fps, speed_up_factor):
    min_track_length = 3
    if len(track) >= min_track_length:
        velocities = calculate_velocity(track, frame_stride, fps, speed_up_factor)
        
        nearby_objects = []
        distance_threshold = max(frame_width, frame_height) * 0.02  # Adjust this factor as needed
        
        for i, point in enumerate(track):
            current_frame_objects = [t[i] for t in all_tracks.values() if len(t) > i and int(t[i][4]) in vehicle_classes]
            nearby = count_nearby_objects(point, current_frame_objects, frame_width, frame_height, distance_threshold)
            nearby_objects.append(nearby)
        
        return str(track_id), [
            [
                x, y, w, h, 
                dx, dy, speed,
                nearby
            ]
            for (x, y, w, h, _), (dx, dy, speed), nearby in zip(track[1:], velocities, nearby_objects[1:])
        ]
    return None

# Calculate velocities and nearby objects
track_data = {}

max_threads = min(32, os.cpu_count() + 4)

with ThreadPoolExecutor(max_workers=max_threads) as executor:
    future_to_track = {
        executor.submit(
            process_trajectory, 
            track_id, 
            track, 
            track_history, 
            TARGET_WIDTH, 
            TARGET_HEIGHT, 
            vehicle_classes,
            frame_stride,
            fps,
            speed_up_factor
        ): track_id 
        for track_id, track in track_history.items()
    }
    for future in concurrent.futures.as_completed(future_to_track):
        result = future.result()
        if result:
            track_id, track = result
            track_data[track_id] = track

    executor.shutdown(wait=True)

threading.local().__dict__.clear()
gc.collect()

# Save track data
with open(os.path.join(new_track_path, "track_history.json"), "w") as json_f:
    json.dump(track_data, json_f, indent=4)

print("Saved to", new_track_path)
end = time.time()
print("It took", end - start, "seconds!")