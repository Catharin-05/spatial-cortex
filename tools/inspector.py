import ollama
import cv2
import base64

def analyze_state(video_path, timestamp_sec, question):
    """
    Uses Ollama (Llava) to inspect a frame. 
    No extra weights or complex libraries needed.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp_sec * fps))
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return "Error: Could not extract frame."

    # Encode frame to base64 for Ollama
    _, buffer = cv2.imencode('.jpg', frame)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    print(f"👁️ Ollama-Llava is inspecting frame at {timestamp_sec}s...")
    
    response = ollama.generate(
        model='llava',
        prompt=question,
        images=[image_base64]
    )
    
    return response['response']