import pandas as pd
import numpy as np

def analyze_dataset(csv_path):
    """
    Reads the dataset and computes statistics for calibration.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

    # Relevant Action Units
    au_columns = [
        'AU01_inner_brow_raise', 
        'AU04_brow_lower', 
        'AU06_cheek_raise', 
        'AU12_lip_corner_pull', 
        'AU15_lip_corner_depress'
    ]
    
    stats = {}
    
    print("--- DATASET ANALYSIS ---")
    print(f"Total Rows: {len(df)}")
    
    for au in au_columns:
        if au not in df.columns:
            print(f"Warning: {au} not found in CSV.")
            continue
            
        data = df[au]
        
        # Calculate Stats
        mean_val = data.mean()
        std_val = data.std()
        max_val = data.max()
        p75 = np.percentile(data, 75)
        p90 = np.percentile(data, 90)
        p95 = np.percentile(data, 95)
        
        stats[au] = {
            "mean": mean_val,
            "std": std_val,
            "high_threshold": p90, # We use Top 10% as likely "Active" events
            "max": max_val
        }
        
        print(f"\n{au}:")
        print(f"  Mean: {mean_val:.2f} | Std: {std_val:.2f}")
        print(f"  75th: {p75:.2f} | 90th: {p90:.2f} (Proposed Threshold)")
        
    # Analyze Duration? 
    # Use 'timestamp' and 'video_clip_id'. 
    # If we assume rows are sequential frames? The timestamp delta is helpful.
    # Let's check avg timestamp delta
    if 'timestamp' in df.columns and len(df) > 1:
        # Sort by clip and timestamp just in case
        # But CSV looks mixed. 
        pass
        
    return stats

if __name__ == "__main__":
    analyze_dataset("dataset.csv")
