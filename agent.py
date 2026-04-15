import json
import ollama
from typing import List, Dict
from tools.detector import search_objects
from tools.inspector import analyze_state
from tools.tracker import count_unique_objects
from tools.compare import check_progress

class SpatialCortexAgent:
    def __init__(self, video_path: str, model: str = "llama3.2"):
        self.video_path = video_path
        self.model = model
        self.max_retries = 5
        self.memory: List[Dict] = []

    def _generate_prompt(self) -> str:
        return """
        You are an autonomous Construction Site Auditor. You use a 'Cycle of Thought' to verify site safety and progress.
        
        AVAILABLE TOOLS:
        - search_objects(target_object: str): Returns timestamps where an object is detected.
        - analyze_state(timestamp_sec: int, question: str): Visual inspection of a specific frame using VQA.
        - count_unique_objects(target: str): Tracking-based counting for logistics.
        - check_progress(): High-level comparison of site changes.

        INSTRUCTIONS:
        1. Analyze the user request.
        2. If you need data, pick a tool.
        3. If you have the tool's result, decide if you need another tool or can answer.
        4. ALWAYS respond in valid JSON.

        OUTPUT FORMAT:
        {
            "thought": "Your internal reasoning",
            "tool": "tool_name or 'none'",
            "parameters": {},
            "final_answer": "Only provide this if you have sufficient data"
        }
        """

    def run(self, user_query: str):
        print(f"🎬 Initializing Agentic Flow for: '{user_query}'")
        self.memory.append({"role": "user", "content": user_query})

        for i in range(self.max_retries):
            # 1. THE REASONING STEP (The "Node")
            response = ollama.chat(
                model=self.model,
                format="json",
                messages=[
                    {"role": "system", "content": self._generate_prompt()},
                    *self.memory
                ]
            )
            
            content = json.loads(response['message']['content'])
            thought = content.get("thought", "")
            tool = content.get("tool", "none")
            
            print(f"\n🧠 [Cycle {i+1}] Thought: {thought}")

            # 2. THE TERMINATION STEP (The "Edge")
            if content.get("final_answer"):
                print(f"🏁 MISSION COMPLETE")
                return content["final_answer"]

            # 3. THE ACTION STEP (The "Action")
            if tool != "none":
                params = content.get("parameters", {})
                print(f"🛠️  Executing: {tool}({params})")
                
                observation = self._execute_tool(tool, params)
                print(f"👁️  Observation: {observation}")
                
                # Update memory with the loop's finding
                self.memory.append({"role": "assistant", "content": json.dumps(content)})
                self.memory.append({"role": "user", "content": f"Observation: {observation}"})
            else:
                break

        return "Agent failed to reach a conclusion within max cycles."

    def _execute_tool(self, name: str, params: Dict):
        try:
            if name == "search_objects":
                return search_objects(self.video_path, **params)
            elif name == "analyze_state":
                return analyze_state(self.video_path, **params)
            elif name == "count_unique_objects":
                return count_unique_objects(self.video_path, **params)
            elif name == "check_progress":
                return check_progress(self.video_path)
            return "Tool not found."
        except Exception as e:
            return f"Tool Execution Error: {str(e)}"

# --- BOOTSTRAP ---
if __name__ == "__main__":
    # Ensure your video exists in the directory
    site_agent = SpatialCortexAgent(r"test-videos\8598744-uhd_3840_2160_30fps.mp4")
    
    # Example Complex Query
    task = "Find where the cement truck is, check if the driver is wearing a helmet, and tell me if the unloading is finished."
    
    final_report = site_agent.run(task)
    print("\n" + "="*50)
    print("SITE AUDIT REPORT:")
    print(final_report)