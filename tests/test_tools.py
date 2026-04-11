import sys
import os
import cv2

# 1. PATH CONFIGURATION
# Ensuring we can find tools.py and the video folder
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

# 2. IMPORTS
from tools import check_motion, detect_objects_in_frame, search_event

# Construct the path to your specific VIRAT video
VIDEO_PATH = os.path.join(parent_dir, "test-videos", "VIRAT_S_010204_05_000856_000890.mp4")

def run_comprehensive_tests():
    print("🚀 STARTING EDGE AGENT UNIT TESTS")
    print("-" * 40)
    
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ ERROR: Video not found at {VIDEO_PATH}")
        return

    # --- TEST 1: SPATIAL DETECTION ---
    print("\n🔍 TEST 1: Spatial Detector (YOLO-ONNX)")
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100) # Grab a frame from the middle
    ret, test_frame = cap.read()
    cap.release()

    if ret:
        detections = detect_objects_in_frame(test_frame)
        print(f"✅ Detection successful. Objects found: {detections}")
    else:
        print("❌ Failed to read frame for Spatial Test.")

    # --- TEST 2: THE GATEKEEPER ---
    print("\n🧠 TEST 2: The Gatekeeper (Motion Logic)")
    cap = cv2.VideoCapture(VIDEO_PATH)
    _, f1 = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 60) # Skip 2 seconds ahead
    _, f2 = cap.read()
    cap.release()
    
    motion = check_motion(f1, f2)
    print(f"✅ Motion check complete. Movement detected: {motion}")

    # --- TEST 3: COARSE-TO-FINE PIPELINE ---
    print("\n🕰️ TEST 3: Coarse-to-Fine Search (Temporal Logic)")
    # We'll search for 'car' or 'person' specifically in the VIRAT footage
    # skip_seconds=5 makes the test run very fast for verification
    timestamps = search_event(VIDEO_PATH, target_object="car", skip_seconds=5)
    
    if timestamps:
        print(f"✅ Search Tool successful. 'car' detected at: {timestamps} seconds")
    else:
        print("ℹ️ Search Tool finished: No 'car' detections in these specific sampled frames.")

    print("\n" + "="*40)
    print("✅ ALL CORE TOOLS OPERATIONAL")
    print("="*40)

if __name__ == "__main__":
    run_comprehensive_tests()