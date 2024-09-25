from ultralytics import YOLO
import cv2
import os
import time
import gc
import torch
import numpy as np
from threading import Thread, Lock
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

acci_model = YOLO("best.pt")
yolo_model = YOLO("best_yolo.pt")
tracker = "bytetrack.yaml"

# Open the video file
video_path = "Inputs/HitNRun1.mp4"
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

track_history_lock = Lock()  # Initialize a lock for track history

frame_duration = 1 / fps  # Time for one frame
accident_tracking_duration = 5  # 5 seconds
accident_tracking_frames = int(accident_tracking_duration / frame_duration)

track_history = {}

accident_participants = {}  # To store accident participants
accident_bbox_history = {}  # To store accident bounding boxes

def check_overlap(bbox1, bbox2):
    """ Check if two bounding boxes overlap """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)

def run_accident_detection(frame, frame_count, annotated_frame):
    # Run the accident detection model
    results = acci_model(frame, device=device)
    if results and results[0].boxes:
        print(f"ACCIDENT detected in FRAME {frame_count}")
        accident_bbox_list = []
       
        for box in results[0].boxes:
            accident_bbox = box.xyxy.cpu().numpy().flatten()
            accident_bbox_list.append(accident_bbox)
           
            # Draw red bounding box for accidents
            cv2.rectangle(annotated_frame,
                          (int(accident_bbox[0]), int(accident_bbox[1])),
                          (int(accident_bbox[2]), int(accident_bbox[3])),
                          (0, 0, 255), 2)  # Red for accidents
            cv2.putText(annotated_frame, 'ACCIDENT',
                        (int(accident_bbox[0]), int(accident_bbox[1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Store the accident bounding boxes
        accident_bbox_history[frame_count] = accident_bbox_list

        # Check overlap with accident bbox
        overlapping_objects = []
        if frame_count in track_history:
            for accident_bbox in accident_bbox_list:
                # Check if there are at least two objects overlapping in the accident bbox
                for track_id1, box1 in track_history[frame_count].items():
                    for track_id2, box2 in track_history[frame_count].items():
                        if track_id1 != track_id2:
                            object_bbox1 = [box1[0] - box1[2] / 2, box1[1] - box1[3] / 2, box1[0] + box1[2] / 2, box1[1] + box1[3] / 2]
                            object_bbox2 = [box2[0] - box2[2] / 2, box2[1] - box2[3] / 2, box2[0] + box2[2] / 2, box2[1] + box2[3] / 2]

                            if check_overlap(accident_bbox, object_bbox1) and check_overlap(accident_bbox, object_bbox2):
                                overlapping_objects.append(track_id1)
                                overlapping_objects.append(track_id2)

        # Track only overlapping objects (participants) for the next 5 seconds
        if len(set(overlapping_objects)) >= 2:
            accident_participants[frame_count] = list(set(overlapping_objects))

def run_object_tracking(frame, frame_count, annotated_frame):
    # Run YOLO model for object tracking
    results = yolo_model.track(frame, device=device, persist=True, tracker=tracker)

    if results and results[0].boxes and results[0].boxes.xywh is not None:
        # Get tracked boxes and their IDs
        track_ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else []
        boxes = results[0].boxes.xywh.cpu().numpy() if results[0].boxes.xywh is not None else []

        if len(track_ids) > 0 and len(boxes) > 0:
            tracked_objects = {}

            for track_id, box in zip(track_ids, boxes):
                # Get (x, y, w, h)
                x, y, w, h = box

                # Store tracking information in track history
                with track_history_lock:
                    if frame_count not in track_history:
                        track_history[frame_count] = {}
                    track_history[frame_count][track_id] = (x, y, w, h)

                # Draw green bounding box for object tracking
                cv2.rectangle(annotated_frame,
                              (int(x - w / 2), int(y - h / 2)),
                              (int(x + w / 2), int(y + h / 2)),
                              (0, 255, 0), 2)  # Green for YOLO objects

                # Annotate the frame with the track ID instead of the object type
                cv2.putText(annotated_frame, f'ID: {track_id}',
                            (int(x - w / 2), int(y - h / 2) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        print(f"No objects detected in frame {frame_count}")


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

# Display the accident participants results
print("Accidents and their participants:")
for frame_id, object_ids in accident_participants.items():
    print(f"Frame {frame_id}: Participants IDs {object_ids}")

# Clean up GPU memory
torch.cuda.empty_cache()
gc.collect()

print("Saved to", new_track_path)
end = time.time()