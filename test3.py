import cv2
import torch
import time
from ultralytics import YOLO
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace, yaml_load
from ultralytics.engine.results import Results
from pathlib import Path

# Load the YOLOv8 model
model = YOLO("C:\\Users\\aimlc\\OneDrive\\Desktop\\Sowmesh\\MulitpleObjectTracking\\models\\runs\\detect\\train17\\weights\\best.pt")

# Load the SAHI detection model
detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path="C:\\Users\\aimlc\\OneDrive\\Desktop\\Sowmesh\\MulitpleObjectTracking\\models\\runs\\detect\\train17\\weights\\best.pt",
    confidence_threshold=0.05,
    device="cuda"
)

# Load the tracker configuration
tracker_config_path = "C:\\Users\\aimlc\\OneDrive\\Desktop\\Sowmesh\\MulitpleObjectTracking\\bytetrack.yaml"
tracker_args = yaml_load(tracker_config_path)
tracker_args = IterableSimpleNamespace(**tracker_args)

# Open the video file
video_path = "C:\\Users\\aimlc\\OneDrive\\Desktop\\Sowmesh\\MulitpleObjectTracking\\SampleVids\\traffic_vid2Shortened.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize the BYTEtrack tracker with configuration
tracker = BYTETracker(tracker_args, frame_rate=fps)  # Set frame_rate to 1 since we're processing 1 frame per second

# Initialize video writer
output_path = "output_tracked_1fps.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 1, (width, height))  # Set fps to 1 for the output video

frame_count = 0
last_process_time = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    current_time = time.time()
    if current_time - last_process_time >= 0.2:  # Process frame if 1 second has passed
        frame_count += 1
        
        # Step 1: Use SAHI to get detections on sliced images
        sliced_results = get_sliced_prediction(
            image=frame,
            detection_model=detection_model,
            slice_height = 768,
            slice_width = 768,
            overlap_height_ratio = 0.5,
            overlap_width_ratio = 0.5
        )
        
        boxes = []
        scores = []
        class_ids = []
        for object_prediction in sliced_results.object_prediction_list:
            bbox = object_prediction.bbox.to_xyxy()
            boxes.append(bbox)
            scores.append(object_prediction.score.value)
            class_ids.append(object_prediction.category.id)
        
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        class_ids_tensor = torch.tensor(class_ids, dtype=torch.int64)
        combined_data = torch.cat([boxes_tensor, scores_tensor.unsqueeze(1), class_ids_tensor.unsqueeze(1)], dim=1)
        
        results = Results(
            orig_img=frame,
            path=video_path,
            names=model.names,
            boxes=combined_data
        )
        
        # Step 2: Update tracker with detections
        det = results.boxes.cpu().numpy()
        if len(det):
            tracks = tracker.update(det, frame)
            
            # Update results with tracking information
            if len(tracks):
                idx = tracks[:, -1].astype(int)
                results = results[idx]
                results.update(boxes=torch.as_tensor(tracks[:, :-1]))
        
        # Step 3: Plot results
        plotted_frame = results.plot()
        
        # Write the frame with detections and tracking to the output video
        out.write(plotted_frame)
        
        # Display the frame (optional)
        cv2.imshow('Tracked Objects', plotted_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        last_process_time = current_time
    else:
        # If less than a second has passed, skip to the next frame
        cap.grab()

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Tracking completed. Output saved to {output_path}")