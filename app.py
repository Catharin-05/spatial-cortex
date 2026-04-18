import streamlit as st
import os
from agent import SpatialCortexAgent

# Streamlit Page Config
st.set_page_config(page_title="Spatial Cortex UI", layout="wide")
st.title("🏗️ Spatial Cortex: Site Auditor")

# Sidebar for inputs
with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Upload Construction Video (MP4)", type=["mp4"], key="video_uploader")
    query = st.text_area("Audit Query", value="Find where the truck is and check if it is a cement truck.", key="audit_query_input")
    run_button = st.button("Run Audit", key="run_audit_btn")

if uploaded_file and run_button:
    # Save the uploaded video to disk so OpenCV can read it
    video_path = "temp_ui_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())
    
    # Layout: Video on left, Logs on right
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Source Feed")
        st.video(video_path)
        
    with col2:
        st.subheader("Agent Execution Log")
        log_container = st.container()
        
        agent = SpatialCortexAgent(video_path)
        
        with log_container:
            for step in agent.run_stream(query):
                if step["type"] == "info":
                    st.info(f"🎬 {step['content']}")
                
                elif step["type"] == "thought":
                    st.markdown(f"**🧠 [Cycle {step['cycle']}] Thought:** {step['content']}")
                
                elif step["type"] == "tool":
                    st.code(f"🛠️ Executing: {step['tool']}({step['params']})")
                
                elif step["type"] == "observation":
                    st.markdown(f"> 👁️ **Observation:** {step['content']}")
                    st.divider()
                
                elif step["type"] == "final":
                    st.success("🏁 MISSION COMPLETE")
                    st.subheader("Site Audit Report")
                    st.write(step["content"])
                
                elif step["type"] == "error":
                    st.error(step["content"])
                    
    # Cleanup temporary video file
    if os.path.exists(video_path):
        os.remove(video_path)
elif run_button and not uploaded_file:
    st.error("Please upload a video file first.")