import cv2
import time
import torch
from ultralytics import YOLO

def process_video(source=0, model_path='yolo11n.pt', conf_threshold=0.5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = YOLO(model_path)
    model.to(device)
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    frame_count = 0
    total_time = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        
        inference_start = time.time()
        results = model(frame, conf=conf_threshold)
        inference_end = time.time()
        inference_time = inference_end - inference_start
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = f"{model.names[cls]} {conf:.2f}"
                
                if model.names[cls] == 'person':
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        end_time = time.time()
        frame_time = end_time - start_time
        total_time += frame_time
        frame_count += 1
        
        fps = frame_count / total_time if total_time > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Inference time: {inference_time * 1000:.4f} ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cv2.imshow('YOLO Person Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video(source='/data/6387-191695740_small.mp4', model_path='yolo11n.pt', conf_threshold=0.5)