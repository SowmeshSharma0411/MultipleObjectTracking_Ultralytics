from ultralytics import YOLO
import cv2
from collections import defaultdict
import numpy as np
# import torch
import time
import os
import json

# start = time.time()
# model = YOLO("yolov8n.pt")
# print(torch.cuda.is_available())

# results = model.track("SampleVids/bombay_trafficShortened.mp4", show=True, tracker="bytetrack.yaml", save=True)  # with ByteTrack
# end = time.time()
# length = end - start

# # Show the results : this can be altered however you like
# print("It took", length, "seconds!")

start= time.time()

# Load the YOLOv8 model
model = YOLO("yolov8x.pt")

# Open the video file
video_path = "SampleVids/bombay_trafficShortened.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

_tracker = "bytetrack.yaml"

new_track_dir_no = 2
for d in os.listdir("runs/detect"):
    if d.startswith("track"):
        new_track_dir_no = max(int(d[5:]), new_track_dir_no)

new_track_dir_no+=1
new_track_dir = f"track{new_track_dir_no}"
new_track_path = os.path.join("runs/detect", new_track_dir)
os.makedirs(new_track_path)

output_video_path = os.path.join(new_track_path, "output.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_stride = 20  # Number of frames to skip between each detection

i=0


# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, tracker=_tracker)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        if(i%frame_stride==0):
            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                # track.append((float(x), float(y)))  # x, y center point
                track.append((float(x), float(y), float(w), float(h)))
                # if len(track) > 30:  # retain 90 tracks for 90 frames
                #     track.pop(0)

                # Draw the tracking lines
                # points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                # cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=3)

            # Write the annotated frame to the output video
        out.write(annotated_frame)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        i+=1

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

# with open(os.path.join(new_track_path, "track_history.txt"), "w") as f:
#     for track_id, points in track_history.items():
#         f.write(f"Track ID {track_id}: {points}\n")

# Convert track_history to a regular dictionary
track_history_dict = {track_id: points for track_id, points in track_history.items()}

with open(os.path.join(new_track_path, "track_history.json"), "w") as json_f:
    json.dump(track_history_dict, json_f, indent=4)

print("Saved to ", new_track_path)

end = time.time()
length = end - start
print("It took", length, "seconds!")