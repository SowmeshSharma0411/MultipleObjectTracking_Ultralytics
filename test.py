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
from sklearn.neighbors import NearestNeighbors
from collections import deque
from ultralytics import YOLO

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)  # Disable gradient computation

# Constants
VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck
MIN_TRACK_LENGTH = 5
MAX_THREADS = min(32, os.cpu_count() + 4)
FRAME_CALCULATION_INTERVAL = 60  # Calculate dynamic radius every 60 frames

def init_kalman_filter():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    kf.R *= 10
    kf.P *= 1000
    return kf

@torch.jit.script
def calculate_velocity(prev: torch.Tensor, curr: torch.Tensor, time_interval: float, frame_diagonal: float) -> torch.Tensor:
    dx = curr[:, 0] - prev[:, 0]
    dy = curr[:, 1] - prev[:, 1]
    displacement = torch.sqrt(dx**2 + dy**2)
    velocity = displacement / time_interval
    avg_size = (prev[:, 2] + prev[:, 3] + curr[:, 2] + curr[:, 3]) / 4
    return torch.stack([dx, dy, velocity / avg_size / frame_diagonal, curr[:, 2], curr[:, 3]], dim=1)

def estimate_average_spacing(all_objects, k=5):
    if len(all_objects) < 2:
        return 150.0
    centroids = all_objects[:, :2]
    sizes = all_objects[:, 2:].max(axis=1)
    k = min(k, len(all_objects) - 1)
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(centroids)
    distances, _ = nbrs.kneighbors(centroids)
    avg_distances = np.median(distances[:, 1:], axis=1)
    adjusted_spacing = avg_distances + (sizes / 4)
    return float(np.median(adjusted_spacing))

def count_nearby_objects(current_object, all_objects, dynamic_radius):
    tree = cKDTree(all_objects)
    return len(tree.query_ball_point(current_object, dynamic_radius)) - 1

def process_trajectory(track_id, track, all_tracks, frame_width, frame_height, dynamic_radius, fps, frame_diagonal, frame_stride):
    if len(track) >= MIN_TRACK_LENGTH:
        track_tensor = torch.tensor(track)
        velocities = calculate_velocity(track_tensor[:-1], track_tensor[1:], frame_stride / fps, frame_diagonal)
        
        nearby_objects = []
        for i, point in enumerate(track[1:], 1):
            current_frame_objects = torch.stack([t[i][:2] for t in all_tracks.values() if len(t) > i and int(t[i][4]) in VEHICLE_CLASSES])
            nearby = count_nearby_objects(point[:2], current_frame_objects, dynamic_radius)
            nearby_objects.append(nearby)
        
        return str(track_id), [
            (*point[:4], *velocity[:3], nearby)
            for point, velocity, nearby in zip(track[1:], velocities, nearby_objects)
        ]
    return None

def main():
    start = time.time()

    # Load model
    model = YOLO("yolov8n.pt").to(device)

    # Video setup
    video_path = "SampleVids/bombay_trafficShortened.mp4"
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_HEIGHT))
    frame_diagonal = np.sqrt(width**2 + height**2)
    frame_stride = max(1, int(fps / 1.5))

    # Output setup
    new_track_dir = f"track{max([int(d[5:]) for d in os.listdir('runs/detect') if d.startswith('track')] + [1]) + 1}"
    new_track_path = os.path.join("runs/detect", new_track_dir)
    os.makedirs(new_track_path)
    output_video_path = os.path.join(new_track_path, "output.mp4")
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Tracking setup
    track_history = {}
    kalman_filters = {}
    dynamic_radius = 150
    alpha = 0.2

    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame, persist=True, tracker="bytetrack.yaml")
        
        boxes = results[0].boxes.xywh
        track_ids = results[0].boxes.id.int().cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        
        annotated_frame = results[0].plot()

        if frame_count % FRAME_CALCULATION_INTERVAL == 0:
            curr_frame_objects = torch.tensor([(box[0].item(), box[1].item(), box[2].item(), box[3].item()) for box, cls in zip(boxes, classes) if int(cls) in VEHICLE_CLASSES])
            dynamic_radius = alpha * estimate_average_spacing(curr_frame_objects) + (1 - alpha) * dynamic_radius
        
        if frame_count % frame_stride == 0:
            for box, track_id, cls in zip(boxes, track_ids, classes):
                if int(cls) in VEHICLE_CLASSES:
                    x, y, w, h = box
                    if track_id not in track_history:
                        track_history[track_id] = deque(maxlen=100)  # Limit history to prevent memory issues
                        kalman_filters[track_id] = init_kalman_filter()
                    
                    kf = kalman_filters[track_id]
                    kf.predict()
                    kf.update(np.array([x, y]))
                    smoothed_x, smoothed_y = kf.x[:2]
                    
                    track_history[track_id].append((float(smoothed_x), float(smoothed_y), float(w), float(h), int(cls)))
        
        out.write(annotated_frame)
        
        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Process trajectories
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        future_to_track = {executor.submit(process_trajectory, track_id, list(track), track_history, width, height, dynamic_radius, fps, frame_diagonal, frame_stride): track_id for track_id, track in track_history.items()}
        track_data = {track_id: track for future in concurrent.futures.as_completed(future_to_track) for track_id, track in [future.result()] if track}

    # Save track data
    with open(os.path.join(new_track_path, "track_history.json"), "w") as json_f:
        json.dump(track_data, json_f)

    print(f"Saved to {new_track_path}")
    print(f"Processing time: {time.time() - start} seconds")

    # Clean up
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()