import cv2

from tools.inspector import analyze_state
def check_progress(video_path):
    """
    Compares the beginning and the end of the video to summarize change.
    """
    cap = cv2.VideoCapture(video_path)
    
    # Get first frame
    ret, frame_start = cap.read()
    
    # Get last frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
    ret, frame_end = cap.read()
    cap.release()

    # Use Moondream to describe the difference
    desc_start = analyze_state(video_path, 0, "Describe the state of the construction.")
    desc_end = analyze_state(video_path, 2, "Describe the state of the construction.") # Assuming 2s+ video
    
    return f"Start state: {desc_start}. End state: {desc_end}. Progress involves these changes."