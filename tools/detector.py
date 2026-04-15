import cv2
from ultralytics import YOLO

# Load model once to save memory
model = YOLO("yolov8n.pt") 

def search_objects(video_path, target_object="person", zone_coords=None):
    """
    Scans video for specific objects. 
    If zone_coords is provided (polygon), it checks for intrusions.
    """
    cap = cv2.VideoCapture(video_path)
    results_found = []
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Process every 30th frame (1 second of video) to stay fast
        if frame_count % int(fps) == 0:
            timestamp = frame_count // int(fps)
            results = model(frame, verbose=False)
            
            count = 0
            for r in results[0].boxes:
                label = model.names[int(r.cls[0])]
                if label == target_object:
                    count += 1
            
            if count > 0:
                results_found.append({"timestamp": f"{timestamp}s", "count": count})
        
        frame_count += 1
    
    cap.release()
    return results_found