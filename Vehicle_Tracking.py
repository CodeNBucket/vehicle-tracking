import os   
import numpy as np
import supervision as sv
from ultralytics import YOLO
import cv2


HOME = os.getcwd()

# Load Ultralytics YOLO model
model = YOLO("yolov8x.pt")

# Define class mappings
CLASS_NAMES_DICT = model.model.names
SELECTED_CLASS_NAMES = ['car', 'motorcycle', 'bus', 'truck']
SELECTED_CLASS_IDS = [
    {value: key for key, value in CLASS_NAMES_DICT.items()}[class_name]
    for class_name in SELECTED_CLASS_NAMES
]

# Download and set video source
#SOURCE_VIDEO_PATH = os.path.join(Path to your video directory) # ----- Change this part accordingly
TARGET_VIDEO_PATH = os.path.join(HOME, "result.mp4")

video = cv2.VideoCapture(SOURCE_VIDEO_PATH)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) # Setting up width and height
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
video.release()  # Close video file

# **Dynamically set the line at 1-x from the bottom**
LINE_Y = int(height * 5/8)  # Set it up 5/8 to change the height of the line dynamically
LINE_START = sv.Point(0, LINE_Y)
LINE_END = sv.Point(width, LINE_Y) # Covers the width of the video

# Initialize BYTETracker
byte_tracker = sv.ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=30,
    minimum_matching_threshold=0.8,
    frame_rate=60,
    minimum_consecutive_frames=3
)
byte_tracker.reset()

# Initialize LineZone for counting with initialized lines
line_zone = sv.LineZone(start=LINE_START, end=LINE_END)

# Create annotators
box_annotator = sv.BoxAnnotator(thickness=1)
label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.8, text_color=sv.Color.BLACK)
trace_annotator = sv.TraceAnnotator(thickness=1, trace_length=30)
line_zone_annotator = sv.LineZoneAnnotator(thickness=1, text_thickness=1, text_scale=1)

# Define callback function for processing each frame
def callback(frame: np.ndarray, index: int) -> np.ndarray:
    # Run YOLO inference
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    # Filter detections by selected class IDs
    detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]
    
    # Track detections with BYTETracker
    detections = byte_tracker.update_with_detections(detections)
    
    # Create labels
    labels = [
        f"{model.model.names[class_id]} {confidence:0.2f}"
        for confidence, class_id in zip(detections.confidence, detections.class_id)
    ]
    
    # Annotate frame
    annotated_frame = frame.copy()
    annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    
    # Update line counter
    line_zone.trigger(detections)
    
    # Return annotated frame
    return line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)

# Process the whole video
sv.process_video(
    source_path=SOURCE_VIDEO_PATH,
    target_path=TARGET_VIDEO_PATH,
    callback=callback
)

print(f"Processed video saved at: {TARGET_VIDEO_PATH}")
