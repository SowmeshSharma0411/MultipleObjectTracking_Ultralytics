from ultralytics import YOLO
import cv2
import os
import time
import gc
import torch
import numpy as np
from threading import Thread, Lock

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

acci_model = YOLO("best.pt")
yolo_model = YOLO("best_yolo.pt")

# Open the video file
video_path = "Inputs/HitNRun3.mp4"
cap = cv2.VideoCapture(video_path)

new_track_path = "Outputs"

output_video_path = os.path.join(new_track_path, "output.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print(fps, width, height)

frame_count = 0
track_history = {}
accident_objects = {}

track_history_lock = Lock()  # Initialize a lock for track history

def run_accident_detection(frame, frame_count, annotated_frame):
    # Run the accident detection model
    results = acci_model(frame, device=device)
    if results and results[0].boxes:
        print(f"ACCIDENT detected in FRAME {frame_count}")
        for box in results[0].boxes:
            accident_bbox = box.xyxy.cpu().numpy().flatten()
            # Draw red bounding box for accidents
            cv2.rectangle(annotated_frame, 
                          (int(accident_bbox[0]), int(accident_bbox[1])), 
                          (int(accident_bbox[2]), int(accident_bbox[3])), 
                          (0, 0, 255), 2)  # Red for accidents
            cv2.putText(annotated_frame, 'ACCIDENT', 
                        (int(accident_bbox[0]), int(accident_bbox[1])-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


def run_object_tracking(frame, frame_count, annotated_frame):
    # Run YOLO model for object tracking
    results = yolo_model(frame, device=device)
    
    track_history[frame_count] = []
    for box in results[0].boxes:
        obj_id = box.id.cpu().numpy() if box.id is not None else None  # Object ID
        bbox = box.xyxy.cpu().numpy().flatten()  # Flatten to 1D array
        class_idx = int(box.cls.cpu().numpy())  # Get class index
        label = yolo_model.names[class_idx]  # Get label from YOLO model
        
        if len(bbox) == 4:
            # Append bbox, object ID, and label to track history
            track_history[frame_count].append({'id': obj_id, 'bbox': bbox, 'label': label})
            
            # Draw green bounding box for object tracking
            cv2.rectangle(annotated_frame, 
                          (int(bbox[0]), int(bbox[1])), 
                          (int(bbox[2]), int(bbox[3])), 
                          (0, 255, 0), 2)  # Green for YOLO objects
            
            # Annotate the frame with the object label
            cv2.putText(annotated_frame, label, 
                        (int(bbox[0]), int(bbox[1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


# Main processing loop
while cap.isOpened():
    success, frame = cap.read()
    if success:
        # Create a copy of the frame for annotation
        annotated_frame = frame.copy()

        # Run object tracking (YOLO) on the frame
        run_object_tracking(frame, frame_count, annotated_frame)

        # Run accident detection on the frame
        run_accident_detection(frame, frame_count, annotated_frame)
        
        # Display the annotated frame with both accident and object tracking
        cv2.imshow("Accident and Object Tracking", annotated_frame)

        # Save the annotated frame to the output video
        out.write(annotated_frame)
        
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Clean up GPU memory
torch.cuda.empty_cache()
gc.collect()

print("Saved to", new_track_path)
end = time.time()