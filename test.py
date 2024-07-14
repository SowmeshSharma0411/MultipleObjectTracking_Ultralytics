# # from ultralytics import YOLO
# # import torch
# # import time

# # start = time.time()
# # model = YOLO("yolov8n.pt")
# # print(torch.cuda.is_available())

# # # results = model.track("traffic2.mp4", show=True, tracker="bytetrack.yaml", save=True)  # with ByteTrack
# # results = model.track("traffic2.mp4", show=True, tracker="bytetrack.yaml", save=True, 
# #                       imgsz=640, stride=32, batch=4)  # R
# # end = time.time()
# # length = end - start

# # # Show the results : this can be altered however you like
# # print("It took", length, "seconds!")

# # from ultralytics import YOLO
# # import torch
# # import cv2
# # import time
# # from threading import Thread
# # from queue import Queue

# # # Check if CUDA is available
# # print(torch.cuda.is_available())

# # # Load YOLO model
# # model = YOLO("yolov8n.pt")

# # # Video processing parameters
# # video_path = "traffic2.mp4"
# # output_path = "output.mp4"
# # batch_size = 8  # Process frames in batches

# # # Function to read frames from video
# # def read_frames(video_path, queue, batch_size):
# #     cap = cv2.VideoCapture(video_path)
# #     while True:
# #         batch = []
# #         for _ in range(batch_size):
# #             ret, frame = cap.read()
# #             if not ret:
# #                 break
# #             batch.append(frame)
# #         if batch:
# #             queue.put(batch)
# #         else:
# #             break
# #     cap.release()
# #     queue.put(None)

# # # Function to process frames using YOLO and ByteTrack
# # def process_frames(queue, model, result_queue):
# #     while True:
# #         batch = queue.get()
# #         if batch is None:
# #             break
# #         start = time.time()
# #         results = model.track(batch, tracker="bytetrack.yaml")
# #         end = time.time()
# #         print(f"Processed {len(batch)} frames in {end - start:.2f} seconds")
# #         result_queue.put(results)
# #     result_queue.put(None)

# # # Function to display frames
# # def display_frames(result_queue, output_path):
# #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# #     out = None
# #     while True:
# #         results = result_queue.get()
# #         if results is None:
# #             break
# #         for result in results:
# #             annotated_frame = result.plot()  # Get the annotated frame
# #             if out is None:
# #                 height, width = annotated_frame.shape[:2]
# #                 out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
# #             out.write(annotated_frame)

# #             # Display the annotated frame
# #             cv2.imshow('Frame', annotated_frame)
# #             if cv2.waitKey(1) & 0xFF == ord('q'):
# #                 break

# #     if out:
# #         out.release()
# #     cv2.destroyAllWindows()

# # # Create queues to hold video frames and results
# # frame_queue = Queue(maxsize=10)
# # result_queue = Queue(maxsize=10)

# # # Start frame reading, processing, and display threads
# # reader_thread = Thread(target=read_frames, args=(video_path, frame_queue, batch_size))
# # processor_thread = Thread(target=process_frames, args=(frame_queue, model, result_queue))
# # display_thread = Thread(target=display_frames, args=(result_queue, output_path))

# # reader_thread.start()
# # processor_thread.start()
# # display_thread.start()

# # # Wait for all threads to finish
# # reader_thread.join()
# # processor_thread.join()
# # display_thread.join()


# # from ultralytics import YOLO
# # import cv2
# # import torch
# # import time
# # import numpy as np

# # # Ensure CUDA is available and being used
# # print(torch.cuda.is_available())

# # # Load the model (do this only once)
# # model = YOLO("yolov8n.pt")

# # # Open the video file
# # video = cv2.VideoCapture("traffic2.mp4")
# # fps = int(video.get(cv2.CAP_PROP_FPS))
# # width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
# # height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # # Define codec and create VideoWriter object
# # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# # out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

# # # Prepare buffer for batch processing
# # batch_size = 4  # Adjust based on your GPU memory
# # frame_buffer = []

# # start = time.time()
# # frame_count = 0

# # while True:
# #     ret, frame = video.read()
# #     if not ret:
# #         break
    
# #     frame_count += 1
# #     frame_buffer.append(frame)
    
# #     if len(frame_buffer) == batch_size or not ret:
# #         # Process the batch
# #         results = model.track(frame_buffer, persist=True, tracker="bytetrack.yaml")
        
# #         for i, result in enumerate(results):
# #             annotated_frame = result.plot()
# #             out.write(annotated_frame)
            
# #             if frame_count % 30 == 0:  # Show every 30th frame (adjust as needed)
# #                 cv2.imshow('Frame', annotated_frame)
# #                 if cv2.waitKey(1) & 0xFF == ord('q'):
# #                     break
        
# #         frame_buffer = []

# # video.release()
# # out.release()
# # cv2.destroyAllWindows()

# # end = time.time()
# # length = end - start

# # print(f"Processed {frame_count} frames in {length:.2f} seconds. FPS: {frame_count/length:.2f}")

# # from collections import defaultdict

# # import cv2
# # import numpy as np

# # from ultralytics import YOLO
# # import time

# # start= time.time()

# # # Load the YOLOv8 model
# # model = YOLO("yolov8n.pt")

# # # Open the video file
# # video_path = "bombay_trafficShortened.mp4"
# # cap = cv2.VideoCapture(video_path)

# # # Store the track history
# # track_history = defaultdict(lambda: [])

# # # Loop through the video frames
# # while cap.isOpened():
# #     # Read a frame from the video
# #     success, frame = cap.read()

# #     if success:
# #         # Run YOLOv8 tracking on the frame, persisting tracks between frames
# #         results = model.track(frame, persist=True)

# #         # Get the boxes and track IDs
# #         boxes = results[0].boxes.xywh.cpu()
# #         track_ids = results[0].boxes.id.int().cpu().tolist()

# #         # Visualize the results on the frame
# #         annotated_frame = results[0].plot()

# #         # Plot the tracks
# #         for box, track_id in zip(boxes, track_ids):
# #             x, y, w, h = box
# #             track = track_history[track_id]
# #             track.append((float(x), float(y)))  # x, y center point
# #             if len(track) > 30:  # retain 90 tracks for 90 frames
# #                 track.pop(0)

# #             # Draw the tracking lines
# #             points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
# #             cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

# #         # Display the annotated frame
# #         cv2.imshow("YOLOv8 Tracking", annotated_frame)

# #         # Break the loop if 'q' is pressed
# #         if cv2.waitKey(1) & 0xFF == ord("q"):
# #             break
# #     else:
# #         # Break the loop if the end of the video is reached
# #         break

# # # Release the video capture object and close the display window
# # cap.release()
# # cv2.destroyAllWindows()

# # end = time.time()
# # length = end - start
# # print("It took", length, "seconds!")
# # from ultralytics import YOLO
# # import cv2
# # from collections import defaultdict
# # import numpy as np
# # import time
# # import os
# # import json

# # def calculate_velocity(track, fps, window_size=5):
# #     velocities = []
# #     for i in range(1, len(track)):
# #         prev_x, prev_y, _, _ = track[i-1]
# #         curr_x, curr_y, _, _ = track[i]
        
# #         # Calculate displacement
# #         dx = curr_x - prev_x
# #         dy = curr_y - prev_y
# #         displacement = np.sqrt(dx**2 + dy**2)
        
# #         # Calculate time elapsed
# #         time_elapsed = 1 / fps  # in seconds
        
# #         # Calculate velocity (pixels per second)
# #         velocity = displacement / time_elapsed
        
# #         velocities.append(velocity)
    
# #     # Apply moving average smoothing
# #     smoothed_velocities = []
# #     for i in range(len(velocities)):
# #         window = velocities[max(0, i-window_size+1):i+1]
# #         smoothed_velocities.append(sum(window) / len(window))
    
# #     return smoothed_velocities

# # start = time.time()

# # # Load the YOLOv8 model
# # model = YOLO("yolov8n.pt")

# # # Open the video file
# # video_path = "traffic3.mp4"
# # cap = cv2.VideoCapture(video_path)

# # # Store the track history
# # track_history = defaultdict(lambda: [])
# # _tracker = "bytetrack.yaml"

# # # Create output directory
# # new_track_dir_no = max([int(d[5:]) for d in os.listdir("runs/detect") if d.startswith("track")] + [1]) + 1
# # new_track_dir = f"track{new_track_dir_no}"
# # new_track_path = os.path.join("runs/detect", new_track_dir)
# # os.makedirs(new_track_path)

# # # Set up video writer
# # output_video_path = os.path.join(new_track_path, "output.mp4")
# # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# # fps = int(cap.get(cv2.CAP_PROP_FPS))
# # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# # out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# # frame_count = 0

# # # Loop through the video frames
# # while cap.isOpened():
# #     success, frame = cap.read()
# #     if not success:
# #         break

# #     # Run YOLOv8 tracking on the frame, persisting tracks between frames
# #     results = model.track(frame, persist=True, tracker=_tracker)

# #     # Get the boxes and track IDs
# #     boxes = results[0].boxes.xywh.cpu()
# #     track_ids = results[0].boxes.id.int().cpu().tolist()

# #     # Visualize the results on the frame
# #     annotated_frame = results[0].plot()

# #     # Update track history
# #     for box, track_id in zip(boxes, track_ids):
# #         x, y, w, h = box
# #         track_history[track_id].append((float(x), float(y), float(w), float(h)))

# #     # Draw the tracking lines
# #     for track_id, track in track_history.items():
# #         if len(track) > 1:
# #             points = np.array([(int(x), int(y)) for x, y, _, _ in track]).reshape((-1, 1, 2))
# #             cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

# #     # Write the annotated frame to the output video
# #     out.write(annotated_frame)

# #     # Display the annotated frame
# #     cv2.imshow("YOLOv8 Tracking", annotated_frame)

# #     frame_count += 1

# #     if cv2.waitKey(1) & 0xFF == ord("q"):
# #         break

# # # Release resources
# # cap.release()
# # out.release()
# # cv2.destroyAllWindows()

# # # Calculate velocities for each track
# # track_velocities = {}
# # for track_id, points in track_history.items():
# #     if len(points) > 1:
# #         track_velocities[track_id] = calculate_velocity(points, fps)

# # # Save track history and velocities
# # with open(os.path.join(new_track_path, "track_history.json"), "w") as json_f:
# #     json.dump({str(k): v for k, v in track_history.items()}, json_f, indent=4)

# # with open(os.path.join(new_track_path, "track_velocities.json"), "w") as json_f:
# #     json.dump({str(k): v for k, v in track_velocities.items()}, json_f, indent=4)

# # print("Saved to", new_track_path)
# # end = time.time()
# # print("It took", end - start, "seconds!")

# from ultralytics import YOLO
# import cv2
# import numpy as np
# import time
# import os
# import json
# from filterpy.kalman import KalmanFilter

# def initialize_kalman_filter():
#     kf = KalmanFilter(dim_x=4, dim_z=2)  # State: [x, y, vx, vy], Measurement: [x, y]
#     kf.F = np.array([[1, 0, 1, 0],
#                      [0, 1, 0, 1],
#                      [0, 0, 1, 0],
#                      [0, 0, 0, 1]])  # State transition matrix
#     kf.H = np.array([[1, 0, 0, 0],
#                      [0, 1, 0, 0]])  # Measurement function
#     kf.R *= 10  # Measurement noise
#     kf.Q *= 0.1  # Process noise
#     return kf

# def compute_homography():
#     # This is a placeholder. You need to implement this based on your specific setup
#     # For example, you might use known points on the ground plane and their corresponding image coordinates
#     src_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
#     dst_points = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)  # Assuming 10x10 meter area
#     return cv2.getPerspectiveTransform(src_points, dst_points)

# def pixel_to_world(pixel_coord, homography):
#     px, py = pixel_coord
#     world_coord = cv2.perspectiveTransform(np.array([[[px, py]]], dtype=np.float32), homography)
#     return world_coord[0][0]

# def calculate_velocity(track, fps, homography):
#     kf = initialize_kalman_filter()
#     velocities = []
    
#     for i in range(len(track)):
#         x, y, _, _ = track[i]
#         world_x, world_y = pixel_to_world((x, y), homography)
        
#         if i == 0:
#             kf.x = np.array([[world_x], [world_y], [0], [0]])
#         else:
#             kf.predict()
#             kf.update(np.array([[world_x], [world_y]]))
        
#         if i > 0:
#             vx, vy = kf.x[2][0], kf.x[3][0]
#             velocity = np.sqrt(vx**2 + vy**2)
#             velocities.append(velocity)
    
#     return velocities

# start = time.time()

# # Load the YOLOv8 model
# model = YOLO("yolov8n.pt")

# # Open the video file
# video_path = "traffic3.mp4"
# cap = cv2.VideoCapture(video_path)

# # Store the track history
# track_history = {}
# _tracker = "bytetrack.yaml"

# # Create output directory
# new_track_dir_no = max([int(d[5:]) for d in os.listdir("runs/detect") if d.startswith("track")] + [1]) + 1
# new_track_dir = f"track{new_track_dir_no}"
# new_track_path = os.path.join("runs/detect", new_track_dir)
# os.makedirs(new_track_path)

# # Set up video writer
# output_video_path = os.path.join(new_track_path, "output.mp4")
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# homography = compute_homography()

# frame_count = 0

# # Loop through the video frames
# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         break

#     # Run YOLOv8 tracking on the frame, persisting tracks between frames
#     results = model.track(frame, persist=True, tracker=_tracker)

#     # Get the boxes and track IDs
#     boxes = results[0].boxes.xywh.cpu()
#     track_ids = results[0].boxes.id.int().cpu().tolist()

#     # Visualize the results on the frame
#     annotated_frame = results[0].plot()

#     # Update track history
#     for box, track_id in zip(boxes, track_ids):
#         x, y, w, h = box
#         if track_id not in track_history:
#             track_history[track_id] = []
#         track_history[track_id].append((float(x), float(y), float(w), float(h)))

#     # Draw the tracking lines
#     for track_id, track in track_history.items():
#         if len(track) > 1:
#             points = np.array([(int(x), int(y)) for x, y, _, _ in track]).reshape((-1, 1, 2))
#             cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

#     # Write the annotated frame to the output video
#     out.write(annotated_frame)

#     # Display the annotated frame
#     cv2.imshow("YOLOv8 Tracking", annotated_frame)

#     frame_count += 1

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # Release resources
# cap.release()
# out.release()
# cv2.destroyAllWindows()

# # Calculate velocities for each track
# track_velocities = {}
# for track_id, points in track_history.items():
#     if len(points) > 1:
#         track_velocities[track_id] = calculate_velocity(points, fps, homography)

# # Save track history and velocities
# with open(os.path.join(new_track_path, "track_history.json"), "w") as json_f:
#     json.dump({str(k): v for k, v in track_history.items()}, json_f, indent=4)

# with open(os.path.join(new_track_path, "track_velocities.json"), "w") as json_f:
#     json.dump({str(k): v for k, v in track_velocities.items()}, json_f, indent=4)

# print("Saved to", new_track_path)
# end = time.time()
# print("It took", end - start, "seconds!")

# from ultralytics import YOLO
# import cv2
# from collections import defaultdict
# import numpy as np
# import time
# import os
# import json

# def calculate_smooth_velocity(track, frame_interval, fps, window_size=3):
#     velocities = calculate_velocity(track, frame_interval, fps)
#     smoothed_velocities = []
#     for i in range(len(velocities)):
#         window = velocities[max(0, i-window_size+1):i+1]
#         smoothed_velocities.append(sum(window) / len(window))
#     return smoothed_velocities

# def calculate_velocity(track, frame_interval, fps):
#     velocities = []
#     time_interval = frame_interval / fps  # Convert frame interval to seconds

#     for i in range(1, len(track)):
#         prev_x, prev_y, prev_w, prev_h = track[i-1]
#         curr_x, curr_y, curr_w, curr_h = track[i]
        
#         # Calculate displacement
#         dx = curr_x - prev_x
#         dy = curr_y - prev_y
#         displacement = np.sqrt(dx**2 + dy**2)
        
#         # Calculate velocity in pixels per second
#         velocity = displacement / time_interval
        
#         # Adjust velocity based on object size (rough distance estimation)
#         avg_size = (prev_w + prev_h + curr_w + curr_h) / 4
#         adjusted_velocity = velocity / avg_size
        
#         velocities.append(adjusted_velocity)
    
#     return velocities

# start = time.time()

# # Load the YOLOv8 model
# model = YOLO("yolov8x.pt")

# # Open the video file
# video_path = "traffic3.mp4"
# cap = cv2.VideoCapture(video_path)

# # Store the track history
# track_history = defaultdict(lambda: [])
# _tracker = "bytetrack.yaml"

# # Create new directory for output
# new_track_dir_no = max([int(d[5:]) for d in os.listdir("runs/detect") if d.startswith("track")] + [1]) + 1
# new_track_dir = f"track{new_track_dir_no}"
# new_track_path = os.path.join("runs/detect", new_track_dir)
# os.makedirs(new_track_path)

# # Set up video writer
# output_video_path = os.path.join(new_track_path, "output.mp4")
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# frame_stride = 20  # Number of frames to skip between each detection
# frame_count = 0

# # Loop through the video frames
# while cap.isOpened():
#     success, frame = cap.read()
#     if success:
#         # Run YOLOv8 tracking on the frame, persisting tracks between frames
#         results = model.track(frame, persist=True, tracker=_tracker)
        
#         # Get the boxes and track IDs
#         boxes = results[0].boxes.xywh.cpu()
#         track_ids = results[0].boxes.id.int().cpu().tolist()
        
#         # Visualize the results on the frame
#         annotated_frame = results[0].plot()
        
#         if frame_count % frame_stride == 0:
#             # Update track history
#             for box, track_id in zip(boxes, track_ids):
#                 x, y, w, h = box
#                 track_history[track_id].append((float(x), float(y), float(w), float(h)))
        
#         # Write the annotated frame to the output video
#         out.write(annotated_frame)
        
#         # Display the annotated frame
#         cv2.imshow("YOLOv8 Tracking", annotated_frame)
        
#         frame_count += 1
        
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         break

# # Release resources
# cap.release()
# out.release()
# cv2.destroyAllWindows()

# # Calculate velocities
# track_velocities = {}
# for track_id, track in track_history.items():
#     if len(track) > 1:
#         track_velocities[track_id] = calculate_velocity(track, frame_stride, fps)

# # Save track history and velocities
# with open(os.path.join(new_track_path, "track_history.json"), "w") as json_f:
#     json.dump({str(k): v for k, v in track_history.items()}, json_f, indent=4)

# with open(os.path.join(new_track_path, "track_velocities.json"), "w") as json_f:
#     json.dump({str(k): v for k, v in track_velocities.items()}, json_f, indent=4)

# print("Saved to", new_track_path)
# end = time.time()
# print("It took", end - start, "seconds!")


from ultralytics import YOLO
import cv2
import numpy as np
import time
import os
import json

def calculate_smooth_velocity(track, frame_interval, fps, frame_diagonal, window_size=3):
    velocities = []
    time_interval = frame_interval / fps  # Convert frame interval to seconds

    for i in range(1, len(track)):
        prev_x, prev_y, prev_w, prev_h = track[i-1]
        curr_x, curr_y, curr_w, curr_h = track[i]
        
        # Calculate displacement
        dx = curr_x - prev_x
        dy = curr_y - prev_y
        displacement = np.sqrt(dx**2 + dy**2)
        
        # Calculate velocity in pixels per second
        velocity = displacement / time_interval
        
        # Adjust velocity based on object size and frame size
        avg_size = (prev_w + prev_h + curr_w + curr_h) / 4
        adjusted_velocity = velocity / avg_size / frame_diagonal
        
        velocities.append(adjusted_velocity)
    
    # Apply moving average smoothing
    smoothed_velocities = []
    for i in range(len(velocities)):
        window = velocities[max(0, i-window_size+1):i+1]
        smoothed_velocities.append(sum(window) / len(window))
    
    return smoothed_velocities

start = time.time()

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Open the video file
video_path = "traffic3.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = {}
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
frame_stride = 10  # Number of frames to skip between each detection
frame_count = 0

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()
    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, tracker=_tracker)
        
        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        classes = results[0].boxes.cls.cpu().tolist()
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        if frame_count % frame_stride == 0:
            # Update track history
            for box, track_id, cls in zip(boxes, track_ids, classes):
                x, y, w, h = box
                if track_id not in track_history:
                    track_history[track_id] = {"points": [], "class": cls}
                track_history[track_id]["points"].append((float(x), float(y), float(w), float(h)))
        
        # Write the annotated frame to the output video
        out.write(annotated_frame)
        
        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Calculate velocities
min_track_length = 5
track_data = {}
for track_id, track_info in track_history.items():
    if len(track_info["points"]) >= min_track_length:
        velocities = calculate_smooth_velocity(track_info["points"], frame_stride, fps, frame_diagonal)
        track_data[str(track_id)] = {
            "type": model.names[int(track_info["class"])],
            "velocities": velocities
        }

# Save track data
with open(os.path.join(new_track_path, "track_data.json"), "w") as json_f:
    json.dump(track_data, json_f, indent=4)

print("Saved to", new_track_path)
end = time.time()
print("It took", end - start, "seconds!")