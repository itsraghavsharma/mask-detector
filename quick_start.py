#!/usr/bin/env python3
"""
Quick start script for RTSP mask detection with default camera
"""

import sys
import os
from rtsp_mask_detector import RTSPMaskDetector, CameraConfig, setup_logging

def main():
    # Default RTSP URL from your specification
    RTSP_URL = "rtsp://admin:Krishna%40429@192.168.1.25:554/Streaming/Channels/101"
    
    # Model path
    MODEL_PATH = "runs/detect/train/weights/best.pt"
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please ensure the trained model weights are available.")
        sys.exit(1)
    
    # Setup logging
    setup_logging("INFO")
    
    # Create camera configuration
    camera_config = CameraConfig(
        rtsp_url=RTSP_URL,
        camera_id="Main_Camera",
        focal_length_mm=50.0,
        sensor_width_mm=36.0,
        frame_width=1920,
        frame_height=1080,
        known_face_width_cm=15.0
    )
    
    # Initialize and run
    try:
        print("=" * 60)
        print("RTSP Mask Detection System - Quick Start")
        print("=" * 60)
        print(f"Camera URL: {RTSP_URL}")
        print(f"Model: {MODEL_PATH}")
        print("\nStarting detection system...")
        print("Press 'q' in the camera window to quit.")
        print("=" * 60)
        
        detector = RTSPMaskDetector([camera_config], MODEL_PATH)
        detector.initialize()
        detector.run()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
