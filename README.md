Vehicle Tracking with YOLOv8 and ByteTrack

A real-time vehicle detection and tracking system using YOLOv8 and ByteTrack, implemented with Supervision and OpenCV. The script processes a video, detects vehicles, tracks them with unique IDs, and counts the number of vehicles passing a designated line.

## Features
- Detects and tracks cars, motorcycles, buses, and trucks  
- Uses YOLOv8 for object detection  
- Implements ByteTrack for multi-object tracking  
- Dynamically places a tracking line in the video  
- Saves the processed video with bounding boxes, labels, and tracking traces  
- Counts vehicles crossing the tracking line  


- Make sure you have Python 3.8 or newer installed.


- pip install ultralytics==8.3.19 supervision[assets]==0.24.0 numpy opencv-python


- Changing Tracked Vehicle Types
Modify the `SELECTED_CLASS_NAMES` list in the script:

SELECTED_CLASS_NAMES = ['car', 'motorcycle', 'bus', 'truck']

To track bicycles as well, add:
SELECTED_CLASS_NAMES.append('bicycle')


- Adjusting the Tracking Line Position
LINE_Y = int(height * 5/8)  # Adjust the fraction to change height


The script will generate a processed video where:
- Vehicles are detected and tracked.
- A line counts the number of passing vehicles.
- Each vehicle is assigned a unique ID.


