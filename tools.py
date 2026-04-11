import cv2
import numpy as np
from ultralytics import YOLO
import json
import os
# 1. LOAD THE EDGE MODEL
# We load the ONNX model globally once so it doesn't reload into RAM on every function call.
print("⚙️ Loading optimized ONNX YOLO model into memory...")
model = YOLO("yolov8n.onnx", task="detect")

# 2. THE GATEKEEPER (Motion Detection)
def check_motion(frame1, frame2, threshold=25, min_contour_area=500):
    """
    Calculates pixel differences between two frames to detect motion.
    Uses virtually zero compute. If this returns False, we don't run YOLO.
    """
    # Convert frames to grayscale (cheaper to process)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce camera noise/flicker
    gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)
    gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)
    
    # Compute absolute difference between the two frames
    frame_diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Find areas of the image that changed
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            return True # Motion detected! Wake up YOLO.
            
    return False # No motion. Let YOLO sleep.

# 3. THE SPATIAL DETECTOR (ONNX Inference)
def detect_objects_in_frame(frame, target_object=None):
    """
    Runs the ultra-fast ONNX YOLO model on a single frame.
    Returns a list of detected objects.
    """
    target_object = target_object.lower().strip()
    if target_object == "people": target_object = "person"
    if target_object == "cars": target_object = "car"
    # Run inference (verbose=False keeps the terminal clean)
    results = model(frame, verbose=False)[0]
    
    detected_items = []
    
    # Parse the bounding boxes
    for box in results.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        
        # Only keep confident detections
        if confidence > 0.50: 
            detected_items.append(class_name)
            
    # If the LLM is looking for a specific object, filter the list
    if target_object:
        detected_items = [item for item in detected_items if target_object.lower() in item.lower()]
        
    return detected_items

def search_event(video_path, target_object="person", skip_seconds=2):
    """
    COARSE-TO-FINE SEARCH:
    Scans a video quickly by skipping frames. 
    Uses the Gatekeeper to avoid running YOLO on empty scenes.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    found_timestamps = []
    prev_frame = None
    
    print(f"🕵️ Searching for {target_object} in {video_path}...")

    # COARSE PASS: Check every 'skip_seconds'
    for frame_idx in range(0, total_frames, int(fps * skip_seconds)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        timestamp = frame_idx / fps
        
        # 1. THE GATEKEEPER CHECK
        if prev_frame is not None:
            if not check_motion(prev_frame, frame):
                prev_frame = frame
                continue # Skip YOLO, nothing moved!
        
        # 2. THE SPATIAL CHECK (YOLO)
        detections = detect_objects_in_frame(frame, target_object=target_object)
        
        if detections:
            print(f"📍 Found {target_object} near {timestamp:.2f}s")
            found_timestamps.append(timestamp)
            
        prev_frame = frame

    cap.release()
    return found_timestamps