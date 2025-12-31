import numpy as np
import mediapipe as mp

class FeatureExtractor:
    """
    Computes facial feature metrics from MediaPipe landmarks.
    Metrics are normalized by face dimensions to be scale-invariant.
    """
    
    def __init__(self):
        # MediaPipe Canonical Face Mesh Indices
        
        # Reference: Face Height (Top Head to Chin) or IOD (Inter-Ocular Distance)
        # IOD: Left Eye Outer (33) to Right Eye Outer (263)
        self.IDX_LEFT_EYE_OUTER = 33
        self.IDX_RIGHT_EYE_OUTER = 263
        
        # AU01: Inner Brow Raise
        # Left Brow Inner (55) to Left Eye Inner (133)
        # Right Brow Inner (285) to Right Eye Inner (362)
        self.IDX_BROW_INNER_L = 55
        self.IDX_EYE_INNER_L = 133
        self.IDX_BROW_INNER_R = 285
        self.IDX_EYE_INNER_R = 362
        
        # AU04: Brow Lowerer (Corrugator)
        # Distance between Brow Inners (55, 285) or Brow to Nose 
        # Typically Frown = Brows sticking together and down.
        # We'll use Distance between Brow Inners (decreases with frown)
        # AND Distance Brow to Eye (decreases with lower).
        # Metric: Average Vertical Distance Brow-Eye (similar to AU01 but tracked downwards)
        # Let's use Glabella (10) to Brow Inner distance?
        # Simpler: Just track AU01 metric. If it Go DOWN, it's AU04? 
        # No, AU01 is Frontalis (Up). AU04 is Corrugator (In/Down). 
        # We'll monitor Inter-Brow Distance (55 to 285) for "In".
        
        # AU06: Cheek Raise (Orbicularis Oculi) - Squint
        # Cheek Top (111 approx) to Eye Bottom (145)
        self.IDX_CHEEK_L = 111
        self.IDX_EYE_BOTTOM_L = 145
        self.IDX_CHEEK_R = 340 # Symmetrical approx
        self.IDX_EYE_BOTTOM_R = 374
        
        # AU12: Lip Corner Puller (Zygomaticus) - Smile
        # Mouth Left (61) to Mouth Right (291) - Horizontal Dist increases
        self.IDX_MOUTH_L = 61
        self.IDX_MOUTH_R = 291
        
        # AU15: Lip Corner Depressor (DAO) - Frown
        # Mouth Corner (61) to Chin (152) or vertical difference relative to Mouth Center (0)
        # We'll use Vertical position of Corner vs Center.
        self.IDX_MOUTH_CENTER_UP = 0
        self.IDX_MOUTH_CENTER_LOW = 17
        self.IDX_CHIN = 152

    def extract(self, landmarks, image_w, image_h):
        """
        Input: list of NormalizedLandmark (x, y, z).
        Output: dict of raw feature values.
        """
        points = np.array([(lm.x * image_w, lm.y * image_h) for lm in landmarks])
        
        # 1. Scale Factor (IOD)
        iod = np.linalg.norm(points[self.IDX_LEFT_EYE_OUTER] - points[self.IDX_RIGHT_EYE_OUTER])
        if iod == 0: return None
        
        # Helper Dist
        def dist(i1, i2):
            return np.linalg.norm(points[i1] - points[i2])
            
        def dist_y(i1, i2):
            return abs(points[i1][1] - points[i2][1])
            
        # 2. Features
        
        # AU01 Proxy: Brow Inner Height (Larger = Raise)
        brow_l = dist(self.IDX_BROW_INNER_L, self.IDX_EYE_INNER_L)
        brow_r = dist(self.IDX_BROW_INNER_R, self.IDX_EYE_INNER_R)
        au01_raw = (brow_l + brow_r) / (2 * iod)
        
        # AU04 Proxy: Inter-Brow Dist (Smaller = Frown/Squeeze) 
        # Note: We negate it so "Higher Value" = "More Action" to match ML convention?
        # Or we keep it raw. Let's keep raw density.
        # Startle/Frown often generally squeezes face.
        inter_brow = dist(self.IDX_BROW_INNER_L, self.IDX_BROW_INNER_R)
        au04_raw = inter_brow / iod 
        # Note: AU04 triggers when this DECREASES.
        
        # AU06 Proxy: Cheek-Eye Compresion (Smaller = Squint)
        cheek_l = dist(self.IDX_CHEEK_L, self.IDX_EYE_BOTTOM_L)
        cheek_r = dist(self.IDX_CHEEK_R, self.IDX_EYE_BOTTOM_R)
        au06_raw = (cheek_l + cheek_r) / (2 * iod)
        # AU06 triggers when this DECREASES (cheek goes up).
        
        # AU12 Proxy: Mouth Width (Larger = Smile)
        mouth_width = dist(self.IDX_MOUTH_L, self.IDX_MOUTH_R)
        au12_raw = mouth_width / iod
        # AU12 triggers when this INCREASES.
        
        # AU15 Proxy: Mouth Corner Drop
        # Dist from Corner to Chin (Smaller = Dropped/Sad?) 
        # Actually corner moves DOWN, so closer to chin.
        corner_chin_l = dist(self.IDX_MOUTH_L, self.IDX_CHIN)
        corner_chin_r = dist(self.IDX_MOUTH_R, self.IDX_CHIN)
        au15_raw = (corner_chin_l + corner_chin_r) / (2 * iod)
        # AU15 triggers when this DECREASES (corners go down/closer to chin).
        
        return {
            "AU01": au01_raw, # Raise > High
            "AU04": au04_raw, # Frown > Low
            "AU06": au06_raw, # Squint > Low
            "AU12": au12_raw, # Smile > High
            "AU15": au15_raw  # Depress > Low
        }
