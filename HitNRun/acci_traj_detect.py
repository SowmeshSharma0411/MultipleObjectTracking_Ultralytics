from ultralytics import YOLO
import cv2
import os
import time
import gc
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

acci_model = YOLO("best.pt")

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


while cap.isOpened():
    success, frame = cap.read()
    if success:
        # Run accident detection model on the frame
        results = acci_model(frame, device=device)
        
        # Check if any accident is detected
        if results and results[0].boxes:
            print(f"Accident detected in frame {frame_count}")

        # Annotate the frame
        annotated_frame = results[0].plot()
        
        # Save the annotated frame
        out.write(annotated_frame)
        cv2.imshow("Accident Detection", annotated_frame)
        
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