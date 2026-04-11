import ollama
import json
import os
from tools import search_event

# --- CONFIGURATION ---
MODEL = "llama3.2"

# THE MISSION: Extreme strictness to prevent the AI from "guessing" what's in the video.
SYSTEM_PROMPT = """
YOU ARE THE SPATIAL-CORTEX VISION AGENT. YOU ARE CURRENTLY BLIND.
YOU HAVE NO KNOWLEDGE OF THE VIDEO CONTENT UNTIL YOU CALL A TOOL.

STRICT RULES:
1. NEVER guess timestamps or confidence scores.
2. If the user asks a question about the video, your ONLY valid response is a tool call.
3. Use 'search_event' to "see" specific objects like 'person' or 'car'.
4. Do not provide a text answer until AFTER you have tool results.

MANDATORY OUTPUT FORMAT (JSON ONLY):
{
  "tool": "search_event",
  "parameters": {
    "target_object": "person",
    "skip_seconds": 1
  }
}
"""

def run_agent_loop(user_query, video_file):
    """
    Connects the LLM 'Brain' to the Vision 'Tools'.
    """
    video_path = os.path.join("test-videos", video_file)
    
    # Check if video exists before starting
    if not os.path.exists(video_path):
        return f"Error: Video file not found at {video_path}"

    print(f"\n🧠 Agent is thinking about your request...")
    
    # PHASE 1: BRAIN DECIDES WHICH TOOL TO USE
    response = ollama.chat(
        model=MODEL,
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_query}
        ],
        format='json'
    )
    
    raw_content = response['message']['content']
    print(f"🤖 Brain's Plan: {raw_content}")
    
    try:
        decision = json.loads(raw_content)
    except json.JSONDecodeError:
        return "Error: Brain output invalid JSON. Try again."

    # PHASE 2: EXECUTE THE VISION TOOL
    if "search_event" in raw_content.lower():
        # Get target object and normalize (e.g., 'people' -> 'person')
        target = decision.get('parameters', {}).get('target_object', 'person').lower()
        if "person" in target or "people" in target: target = "person"
        if "car" in target: target = "car"
        
        print(f"🎬 [ACTION] Scanning video for '{target}'...")
        
        # This calls the YOLO + Motion logic from tools.py
        results = search_event(video_path, target_object=target, skip_seconds=1)
        
        # PHASE 3: BRAIN SUMMARIZES DATA FOR HUMAN
        print(f"📝 [REPORT] Generating human-readable summary...")
        
        # We give the raw numbers to the LLM and tell it to be a professional analyst
        summary_prompt = f"""
        User Query: "{user_query}"
        Detection Results: {results}
        
        TASK:
        1. Round timestamps to the nearest whole second.
        2. If detections are continuous (e.g., 1s, 2s, 3s), summarize them as a range (e.g., "from 1s to 20s").
        3. Provide a concise, professional answer based ONLY on the detection results.
        """
        
        final_report = ollama.chat(
            model=MODEL,
            messages=[
                {'role': 'system', 'content': "You are a professional security analyst. Be factual and helpful."},
                {'role': 'user', 'content': summary_prompt}
            ]
        )
        return final_report['message']['content']
    
    else:
        return f"Agent decided not to search. Message: {raw_content}"

if __name__ == "__main__":
    # The specific file we are testing
    target_video = "VIRAT_S_010204_05_000856_000890.mp4"
    user_question = "At what times do people appear in the footage?"
    
    print("="*50)
    print(f"SPATIAL-CORTEX AGENT ONLINE")
    print("="*50)
    
    result = run_agent_loop(user_question, target_video)
    
    print("\n" + "="*30)
    print("FINAL AGENT RESPONSE:")
    print(result)
    print("="*30)