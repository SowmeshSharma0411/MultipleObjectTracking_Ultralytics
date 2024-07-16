from ultralytics import YOLO
import cv2
from collections import defaultdict
import numpy as np
import time
import os
import json
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.yolov8 import download_yolov8s_model
from ultralytics.utils.files import increment_path

# Function to run YOLOv8 with SAHI for object detection
def detect_with_sahi(frame, detection_model):
    results = get_sliced_prediction(
        frame, detection_model, slice_height=512, slice_width=512, overlap_height_ratio=0.2, overlap_width_ratio=0.2
    )
    return results.object_prediction_list

# Initialize YOLOv8 model and move to GPU
model = YOLO("yolov8x.pt").to('cuda')
detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8", model_path="models/yolov8x.pt", confidence_threshold=0.3, device="cuda"
)

# Open the video file
video_path = "Road traffic video for object recognition-Trim.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

_tracker = "bytetrack.yaml"

new_track_dir_no = 2
for d in os.listdir("C:\\Gouda\\Capstone\\Segmentation\\YOLO\\out_vid"):
    if d.startswith("track"):
        new_track_dir_no = max(int(d[5:]), new_track_dir_no)

new_track_dir_no += 1
new_track_dir = f"track{new_track_dir_no}"
new_track_path = os.path.join("C:\\Gouda\\Capstone\\Segmentation\\YOLO\\out_vid", new_track_dir)
os.makedirs(new_track_path)

output_video_path = os.path.join(new_track_path, "output.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_stride = 30  # Number of frames to skip between each detection

i = 0

start = time.time()

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run SAHI for object detection
        object_predictions = detect_with_sahi(frame, detection_model)

        # Prepare detections for ByteTrack
        detections = []
        for obj_pred in object_predictions:
            x1, y1, x2, y2 = obj_pred.bbox.minx, obj_pred.bbox.miny, obj_pred.bbox.maxx, obj_pred.bbox.maxy
            confidence = obj_pred.score.value
            class_id = obj_pred.category.id
            detections.append([x1, y1, x2, y2, confidence, class_id])

        # Convert detections to tensor and move to GPU
        detections_tensor = torch.tensor(detections).to('cuda')

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, tracker=_tracker)  # pass the frame here

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        if i % frame_stride == 0:
            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y), float(w), float(h)))

            # Write the annotated frame to the output video
        out.write(annotated_frame)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        i += 1

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()

# Convert track_history to a regular dictionary
track_history_dict = {track_id: points for track_id, points in track_history.items()}

with open(os.path.join(new_track_path, "track_history.json"), "w") as json_f:
    json.dump(track_history_dict, json_f, indent=4)

print("Saved to ", new_track_path)

end = time.time()
length = end - start
print("It took", length, "seconds!")
