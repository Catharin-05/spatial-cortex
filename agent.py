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
        5. For 'search_objects', you must ONLY use basic COCO classes (e.g., 'person', 'truck', 'car').
        6. If the user asks for a 'cement truck', search for a 'truck' first.
        7. Once you find the timestamp of a 'truck', use 'analyze_state' to ask the VQA model: "Is this truck a cement truck?"

        CRITICAL JSON RULES:
        - If you need to use a tool, DO NOT include the "final_answer" key at all.
        - If you have all the info and are ready to answer the user, put your report in "final_answer" and set "tool" to "none".

        OUTPUT FORMAT EXAMPLE (When using a tool):
        {
            "thought": "I need to find a truck first.",
            "tool": "search_objects",
            "parameters": {"target_object": "truck"}
        }
        
        OUTPUT FORMAT EXAMPLE (When finished):
        {
            "thought": "I have all the data.",
            "tool": "none",
            "parameters": {},
            "final_answer": "The cement truck was found at..."
        }
        """

    def run(self, user_query: str):
        print(f"🎬 Initializing Agentic Flow for: '{user_query}'")
        self.memory.append({"role": "user", "content": user_query})

        for i in range(self.max_retries):
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

            final_answer = content.get("final_answer")
            if final_answer and str(final_answer).strip().lower() not in ["none", "null", ""]:
                print(f"🏁 MISSION COMPLETE")
                return final_answer

            if tool != "none":
                params = content.get("parameters", {})
                print(f"🛠️  Executing: {tool}({params})")
                
                observation = self._execute_tool(tool, params)
                print(f"👁️  Observation: {observation}")
                
                self.memory.append({"role": "assistant", "content": json.dumps(content)})
                self.memory.append({"role": "user", "content": f"Observation: {observation}"})
            else:
                print("⚠️ Agent chose no tool but provided no final answer. Forcing loop exit.")
                break

        return "Agent failed to reach a conclusion within max cycles."

    def run_stream(self, user_query: str):
        """Yields execution steps for UI integration."""
        self.memory.append({"role": "user", "content": user_query})
        yield {"type": "info", "content": f"Initializing Agentic Flow for: '{user_query}'"}

        for i in range(self.max_retries):
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
            
            yield {"type": "thought", "cycle": i+1, "content": thought}

            final_answer = content.get("final_answer")
            if final_answer and str(final_answer).strip().lower() not in ["none", "null", ""]:
                yield {"type": "final", "content": final_answer}
                return

            if tool != "none":
                params = content.get("parameters", {})
                yield {"type": "tool", "tool": tool, "params": params}
                
                observation = self._execute_tool(tool, params)
                yield {"type": "observation", "content": observation}
                
                self.memory.append({"role": "assistant", "content": json.dumps(content)})
                self.memory.append({"role": "user", "content": f"Observation: {observation}"})
            else:
                yield {"type": "error", "content": "Agent chose no tool but provided no final answer. Loop exited."}
                break

        yield {"type": "error", "content": "Agent failed to reach a conclusion within max cycles."}

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
    site_agent = SpatialCortexAgent(r"test-videos\8598744-uhd_3840_2160_30fps.mp4")
    task = "Find where the cement truck is, check if the driver is wearing a helmet, and tell me if the unloading is finished."
    final_report = site_agent.run(task)
    print("\n" + "="*50)
    print("SITE AUDIT REPORT:")
    print(final_report)