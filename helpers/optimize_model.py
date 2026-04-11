from ultralytics import YOLO

print("🔄 Downloading base PyTorch YOLOv8 Nano model...")
# This automatically downloads the standard yolov8n.pt file from the internet
model = YOLO("yolov8n.pt") 

print("⚙️ Converting model to ONNX format for Edge deployment...")
# Export the model to ONNX format. 
# dynamic=False locks the input size, making inference faster on your laptop CPU.
# Force opset=17 to ensure compatibility with onnxruntime
success = model.export(format="onnx", dynamic=False, opset=17)

if success:
    print("✅ Successfully exported to yolov8n.onnx!")
else:
    print("❌ Export failed.")