from ultralytics import YOLO
import cv2
from collections import defaultdict
import numpy as np
import time
import os
import json
from scipy.spatial import cKDTree

start = time.time()

model = YOLO("yolov8x.pt")

# video_path = "Road traffic video for object recognition-Trim.mp4"
video_path = "SampleVids/bombay_trafficShortened.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

_tracker = "bytetrack.yaml"

# new_dir = "C:\\Gouda\\Capstone\\Segmentation\\YOLOV8\\MultipleObjectTracking_Ultralytics\\Tracks"
new_dir = "runs/detect"

new_track_dir_no = 2
for d in os.listdir(new_dir):
    if d.startswith("track"):
        new_track_dir_no = max(int(d[5:]), new_track_dir_no)

new_track_dir_no += 1
new_track_dir = f"track{new_track_dir_no}"
new_track_path = os.path.join(new_dir, new_track_dir)
os.makedirs(new_track_path)

output_video_path = os.path.join(new_track_path, "output.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_stride = 20  # Number of frames to skip between each detection

i = 0

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True, tracker=_tracker)

        #its either .cuda() or.device('cuda').
        #@Shivgouda test this out...older code had .cpu()
        #im talking abt the next 2 lines
        boxes = results[0].boxes.xywh.cuda()
        track_ids = results[0].boxes.id.int().cuda().tolist()

        annotated_frame = results[0].plot()

        if i % frame_stride == 0:
            # This is code for velocity and nearby objects: the next 2 lines: they are crucial research on this
            all_positions = [(float(box[0]), float(box[1])) for box in boxes]
            tree = cKDTree(all_positions)

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]

                if len(track) > 0:
                    prev_x, prev_y = track[-1][0], track[-1][1]
                    delta_x, delta_y = float(x) - prev_x, float(y) - prev_y

                    # Calculate instantaneous velocity
                    time_diff = frame_stride / fps  # Time between current and previous detection
                    velocity = np.sqrt(delta_x**2 + delta_y**2) / time_diff
                else:
                    delta_x, delta_y = 0, 0
                    velocity = 0.0

                # Count nearby objects (excluding itself)
                nearby_objects_count = len(tree.query_ball_point((float(x), float(y)), r=50)) - 1

                # Append all calculated values to the track
                track.append((float(x), float(y), float(w), float(h), delta_x, delta_y, velocity, nearby_objects_count))

        out.write(annotated_frame)

        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        i += 1

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Save the track history to a JSON file
track_history_dict = {str(track_id): [
    {
        "x": point[0],
        "y": point[1],
        "w": point[2],
        "h": point[3],
        "delta_x": point[4],
        "delta_y": point[5],
        "velocity": point[6],
        "nearby_objects_count": point[7]
    } for point in points
] for track_id, points in track_history.items()}

with open(os.path.join(new_track_path, "track_history.json"), "w") as json_f:
    json.dump(track_history_dict, json_f, indent=4)

print("Saved to ", new_track_path)

end = time.time()
length = end - start
print("It took", length, "seconds!")
