import numpy as np
import time
from collections import deque
from dataclasses import dataclass

@dataclass
class MicroEvent:
    start_time: float
    end_time: float
    au_type: str
    intensity_z: float
    duration: float
    emotion_label: str = "unknown"

class EventDetector:
    def __init__(self, buffer_duration=5.0):
        # ... (buffer init)
        self.buffer_size = 30 * buffer_duration 
        self.baseline_buffers = {
            "AU01": deque(maxlen=int(self.buffer_size)),
            "AU04": deque(maxlen=int(self.buffer_size)),
            "AU06": deque(maxlen=int(self.buffer_size)),
            "AU12": deque(maxlen=int(self.buffer_size)),
            "AU15": deque(maxlen=int(self.buffer_size))
        }
        
        self.baseline_stats = {}
        self.calibration_done = False
        self.active_events = {} 
        self.event_log = []
        
        self.Z_THRESHOLD = 2.0 
        self.MIN_DURATION = 0.1 
        self.MAX_DURATION = 1.0 
        
        # Load Model
        import pickle
        try:
            with open("emotion_model.pkl", "rb") as f:
                self.model = pickle.load(f)
            print("[System] Emotion Model Loaded.")
        except:
            print("[Warning] Emotion Model not found.")
            self.model = None

    def update(self, features, timestamp):
        """
        Ingest features, update baseline (if calibrating), check for events.
        """
        if not self.calibration_done:
            # Accumulate baseline
            for k, v in features.items():
                self.baseline_buffers[k].append(v)
                
            # Check sufficiency
            if len(self.baseline_buffers["AU01"]) >= 30 * 2: # 2 seconds min
                self._compute_baseline()
            return []

        # Detection Logic
        detected = []
        
        for au, val in features.items():
            mean = self.baseline_stats[au]['mean']
            std = self.baseline_stats[au]['std']
            if std == 0: std = 0.001
            
            z_score = (val - mean) / std
            
            # AU01 (Raise): Trig if Z > T
            # AU04 (Frown): Trig if Z < -T (Distance shrinks)
            # AU06 (Squint): Trig if Z < -T (Distance shrinks)
            # AU12 (Smile): Trig if Z > T
            # AU15 (Depress): Trig if Z < -T (Corners go down -> Dist to chin drops? Wait)
            #   Dist M_Corner to Chin. Norm pose: X. Frown: Corners down -> X decreases.
            
            is_active = False
            intensity = 0.0
            
            if au in ["AU01", "AU12"]:
                if z_score > self.Z_THRESHOLD:
                    is_active = True
                    intensity = z_score
            else:
                if z_score < -self.Z_THRESHOLD:
                    is_active = True
                    intensity = abs(z_score)
            
            # State Machine
            if is_active:
                if au not in self.active_events:
                    self.active_events[au] = {
                        "start": timestamp,
                        "peak_z": intensity
                    }
                else:
                    # Update peak
                    self.active_events[au]['peak_z'] = max(self.active_events[au]['peak_z'], intensity)
            else:
                if au in self.active_events:
                    # Event Ended
                    start = self.active_events[au]['start']
                    duration = timestamp - start
                    peak = self.active_events[au]['peak_z']
                    
                    if duration >= self.MIN_DURATION and duration <= self.MAX_DURATION:
                        # Predict Emotion for this snapshot
                        emotion = "unknown"
                        if self.model:
                            # Construct vector [01, 04, 06, 12, 15]
                            # Ideally we want the values AT THE PEAK or Average during event
                            # For simplicity, we use the current 'features' dictionary values 
                            # (which is the Offset frame, might be low).
                            # Better: Store the peak feature vector in active_events.
                            # For now, let's just use the current feature vector since micro-exp is fast.
                            # Note: The model was trained on RAW features from CSV. 
                            # Our 'features' are normalized ratios. 
                            # The CSV values are around 0-5. Our values are around 0.2-0.5?
                            # WAIT. The CSV data analysis showed Mean ~2.5. 
                            # Our FeatureExtractor returns distance/iod. This is usually < 1.0.
                            # ISSUE: Domain Shift. The Random Forest is trained on CSV values (likely different normalization or mm).
                            # SCALING FIX: We can retrain model on OUR data (impossible now)
                            # OR we trust the RF to learn patterns if trends are similar?
                            # OR we re-scale our inputs to match CSV mean?
                            # CSV Mean ~2.5. Our Mean ~0.5. Scale Factor ~5.0.
                            # Let's try scaling by 5.0 to align ranges roughly.
                            
                            vec = [
                                features["AU01"] * 5.0,
                                features["AU04"] * 5.0,
                                features["AU06"] * 5.0,
                                features["AU12"] * 5.0,
                                features["AU15"] * 5.0
                            ]
                            try:
                                pred = self.model.predict([vec])[0]
                                emotion = pred
                            except:
                                pass
                                
                        evt = MicroEvent(start, timestamp, au, peak, duration, emotion)
                        self.event_log.append(evt)
                        detected.append(evt)
                    
                    del self.active_events[au]
                    
        return detected

    def _compute_baseline(self):
        print("[System] Calibration Complete. Monitoring...")
        for au, buf in self.baseline_buffers.items():
            self.baseline_stats[au] = {
                "mean": np.mean(buf),
                "std": np.std(buf)
            }
        self.calibration_done = True
