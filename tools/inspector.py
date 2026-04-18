import ollama
import cv2
import base64

def analyze_state(video_path, timestamp_sec, question):
    """
    Uses Ollama (Llava) to inspect a frame. 
    Includes defensive programming against LLM hallucinations.
    """
    # --- 🛡️ THE FIX: SCRUB THE INPUT ---
    # If the LLM passes "11s", "11", or 11100, we force it into a clean number
    if isinstance(timestamp_sec, str):
        # Remove any stray "s" and strip spaces
        timestamp_sec = timestamp_sec.replace('s', '').strip()
        
    try:
        # Convert to float (e.g., 11.0)
        timestamp_sec = float(timestamp_sec)
        
        # Safeguard: If the LLM hallucinates a huge number like 11100 instead of 11
        if timestamp_sec > 1000: 
            timestamp_sec = timestamp_sec / 1000 
            
    except ValueError:
        return "Error: timestamp_sec must be a valid number."
    # -----------------------------------

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Now this math is 100% safe
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp_sec * fps))
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return f"Error: Could not extract frame at {timestamp_sec}s."

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

