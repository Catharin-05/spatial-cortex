from ultralytics import YOLO

# Use a model with tracking enabled
tracker_model = YOLO("yolov8n.pt")

def count_unique_objects(video_path, target="truck"):
    """
    Uses ByteTrack to count how many UNIQUE objects pass through the video.
    """
    results = tracker_model.track(video_path, persist=True, verbose=False)
    unique_ids = set()
    
    for r in results:
        if r.boxes.id is not None:
            for box, id in zip(r.boxes, r.boxes.id):
                cls = int(box.cls[0])
                if tracker_model.names[cls] == target:
                    unique_ids.add(int(id))
                    
    return f"Total unique {target}s detected: {len(unique_ids)}"