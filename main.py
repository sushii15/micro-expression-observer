import cv2
import mediapipe as mp
import time
import os
from feature_extraction import FeatureExtractor
from detector import EventDetector
from report_generator import ReportGenerator

def main():
    # Setup
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Webcam not found.")
        return

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    extractor = FeatureExtractor()
    detector = EventDetector(buffer_duration=3.0) # 3s calibration
    
    print("--------------------------------------------------")
    print("   Micro-Expression Observation System")
    print("--------------------------------------------------")
    print("Keep face still for 3 seconds to calibrate.")
    print("Press 'q' to quit and generate report.")
    
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        current_time = time.time() - start_time
        status_text = "Calibrating..."
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # 1. Feature Extraction
            features = extractor.extract(landmarks, w, h)
            
            if features:
                # 2. Detection
                events = detector.update(features, current_time)
                
                if detector.calibration_done:
                    status_text = "Monitoring"
                    # Visual Feedback
                    if events:
                        cv2.putText(frame, "EVENT DETECTED", (50, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        print(f"[{current_time:.2f}s] Event: {events[-1].au_type}")

        # UI
        color = (0, 255, 255) if not detector.calibration_done else (0, 255, 0)
        cv2.putText(frame, f"Status: {status_text}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                   
        cv2.imshow("Micro-Expression Observer", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Report
    print("\nGenerating Report...")
    gen = ReportGenerator(style="plain") # Default to plain
    report = gen.generate(detector.event_log, current_time)
    
    with open("report.txt", "w") as f:
        f.write(report)
        
    print(report)
    print("\nReport saved to report.txt")

if __name__ == "__main__":
    main()
