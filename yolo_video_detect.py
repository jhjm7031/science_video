import torch
import cv2

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov8', 'yolov8n')  # Load the small YOLOv5 model (yolov5s)

# Set the video source: replace 'input_video.mp4' with the path to your video file or use 0 for webcam
video_path = 'D:\File\code\\test_video1.mp4'  # Change this to your video file
cap = cv2.VideoCapture(video_path)

# Check if the video file or webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties for saving the output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object to save output
output_video = 'output_video.avi'  # Change to your desired output file name
out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    # Perform object detection
    results = model(frame)  # Detect objects in the current frame
    results.render()  # Render results on the frame

    # Save the rendered frame to the output video
    output_frame = results.ims[0]
    out.write(output_frame)

    # Display the frame with detections
    cv2.imshow('YOLOv5 Object Detection', output_frame)

    # Press 'q' to quit the video display
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing complete. Output saved to:", output_video)
