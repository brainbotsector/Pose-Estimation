import cv2
import mediapipe as mp

# Initialize MediaPipe Pose Estimation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Load the video
video_path = "dancevideo.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Set the desired display width
display_width = 1080  # Adjust this value to your desired width

# Calculate the corresponding display height to maintain the original aspect ratio
display_height = int((display_width / frame_width) * frame_height)

# Define the output video path
output_video_path = 'output_video.mp4'

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (display_width, display_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe
    results = pose.process(frame_rgb)
    
    # Draw the skeleton with green neon dots if pose detected
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))
    
    # Resize the frame to match the display aspect ratio
    frame = cv2.resize(frame, (display_width, display_height))
    
    # Write the frame to the output video
    out.write(frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(f"Dance Pose Estimation Video saved to {output_video_path}")
# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
