import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

def train_model():
    print("Loading dataset...")
    try:
        df = pd.read_csv("dataset.csv")
    except Exception as e:
        print(f"Error: {e}")
        return

    # Features and Target
    feature_cols = ['AU01_inner_brow_raise', 'AU04_brow_lower', 'AU06_cheek_raise', 'AU12_lip_corner_pull', 'AU15_lip_corner_depress']
    target_col = 'emotion_predicted'
    
    # Check for missing
    df = df.dropna(subset=feature_cols + [target_col])
    
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"Classes: {y.unique()}")
    
    # Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model
    clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    print("Training complete. Evaluation:")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Save
    with open("emotion_model.pkl", "wb") as f:
        pickle.dump(clf, f)
    print("Model saved to emotion_model.pkl")

if __name__ == "__main__":
    train_model()
