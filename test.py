from ultralytics import YOLO
import cv2
import numpy as np
import time
import os
import json
from scipy.spatial import cKDTree
from filterpy.kalman import KalmanFilter

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

def calculate_velocity(track, frame_interval, fps, frame_diagonal):
    velocities = []
    time_interval = frame_interval / fps

    for i in range(1, len(track)):
        prev_x, prev_y, prev_w, prev_h = track[i-1]
        curr_x, curr_y, curr_w, curr_h = track[i]
        
        dx = curr_x - prev_x
        dy = curr_y - prev_y
        displacement = np.sqrt(dx**2 + dy**2)
        
        velocity = displacement / time_interval
        avg_size = (prev_w + prev_h + curr_w + curr_h) / 4
        adjusted_velocity = velocity / avg_size / frame_diagonal
        
        velocities.append((dx, dy, adjusted_velocity, curr_w, curr_h))
    
    return velocities

def count_nearby_objects(current_object, all_objects, radius):
    tree = cKDTree(all_objects)
    return len(tree.query_ball_point(current_object, radius)) - 1

start = time.time()

# Load the YOLOv8 model
model = YOLO("yolov8x.pt")

# Open the video file
video_path = "traffic3.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = {}
kalman_filters = {}
_tracker = "bytetrack.yaml"

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

frame_diagonal = np.sqrt(width**2 + height**2)
frame_stride = max(1, int(fps / 10))  # Adaptive frame stride
frame_count = 0

# Vehicle classes in COCO dataset
vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model.track(frame, persist=True, tracker=_tracker)
        
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        classes = results[0].boxes.cls.cpu().tolist()
        
        annotated_frame = results[0].plot()
        
        if frame_count % frame_stride == 0:
            for box, track_id, cls in zip(boxes, track_ids, classes):
                if int(cls) in vehicle_classes:
                    x, y, w, h = box
                    if track_id not in track_history:
                        track_history[track_id] = []
                        kalman_filters[track_id] = init_kalman_filter()
                    
                    kalman_state = update_kalman(kalman_filters[track_id], x, y)
                    smoothed_x, smoothed_y = kalman_state[0], kalman_state[1]
                    
                    track_history[track_id].append((float(smoothed_x), float(smoothed_y), float(w), float(h)))
        
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

# Calculate velocities and nearby objects
min_track_length = 5
track_data = {}
radius = 50  # Radius for nearby object detection

for track_id, track in track_history.items():
    if len(track) >= min_track_length:
        velocities = calculate_velocity(track, frame_stride, fps, frame_diagonal)
        
        nearby_objects = []
        for i, point in enumerate(track):
            all_objects = [p[:2] for tid, t in track_history.items() for p in t[max(0, i-1):i+2]]
            nearby = count_nearby_objects(point[:2], all_objects, radius)
            nearby_objects.append(nearby)
        
        track_data[str(track_id)] = [
            (x, y, w, h, dx, dy, speed, nearby)
            for (x, y, w, h), (dx, dy, speed, _, _), nearby in zip(track[1:], velocities, nearby_objects[1:])
        ]

# Save track data
with open(os.path.join(new_track_path, "track_history.json"), "w") as json_f:
    json.dump(track_data, json_f, indent=4)

print("Saved to", new_track_path)
end = time.time()
print("It took", end - start, "seconds!")