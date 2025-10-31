# RTSP Mask Detection System

A production-ready, multi-threaded RTSP streaming system for real-time face mask detection with distance calculation.

## Features

- **Multi-Camera Support**: Simultaneously monitor 2+ RTSP camera streams
- **Distance Calculation**: Calculate person distance from camera using face bounding box
- **High Performance**: Multi-threaded architecture for efficient processing
- **Robust Error Handling**: Automatic reconnection on stream failures
- **Industry Compliant**: Production-ready code with logging, configuration management, and error recovery
- **Real-time Display**: Live visualization with FPS, detection count, and distance information

## Installation

1. Install dependencies:
```bash
pip install -r requirements_rtsp.txt
```

2. Ensure you have the trained model weights at:
   - `runs/detect/train/weights/best.pt`

## Quick Start

### Single Camera (Quick Start)
```bash
python rtsp_mask_detector.py --rtsp "rtsp://admin:Krishna%40429@192.168.1.25:554/Streaming/Channels/101"
```

### Multiple Cameras (Using Config File)

1. Create a configuration file `config.json` (or copy from `config.example.json`):
```json
{
  "cameras": [
    {
      "rtsp_url": "rtsp://admin:Krishna%40429@192.168.1.25:554/Streaming/Channels/101",
      "camera_id": "Camera_1",
      "focal_length_mm": 50.0,
      "sensor_width_mm": 36.0,
      "frame_width": 1920,
      "frame_height": 1080,
      "known_face_width_cm": 15.0
    },
    {
      "rtsp_url": "rtsp://admin:password@192.168.1.26:554/Streaming/Channels/101",
      "camera_id": "Camera_2",
      "focal_length_mm": 50.0,
      "sensor_width_mm": 36.0,
      "frame_width": 1920,
      "frame_height": 1080,
      "known_face_width_cm": 15.0
    }
  ]
}
```

2. Run with configuration:
```bash
python rtsp_mask_detector.py --config config.json
```

## Configuration Parameters

### Camera Configuration

- **rtsp_url**: RTSP stream URL (URL-encoded credentials if needed)
- **camera_id**: Unique identifier for the camera
- **focal_length_mm**: Camera focal length in millimeters (for distance calculation)
- **sensor_width_mm**: Camera sensor width in millimeters (for distance calculation)
- **frame_width**: Camera resolution width in pixels
- **frame_height**: Camera resolution height in pixels
- **known_face_width_cm**: Average face width in centimeters (default: 15cm, used for distance estimation)

### Distance Calculation

Distance is calculated using the formula:
```
distance = (known_face_width_cm × focal_length_pixels) / bbox_width_pixels
```

The focal length in pixels is calculated as:
```
focal_length_pixels = (focal_length_mm × frame_width) / sensor_width_mm
```

**Calibration Tips**:
- Measure the actual focal length of your camera lens
- Adjust `known_face_width_cm` based on your use case (average is 14-16cm)
- For more accurate distance, calibrate using a known reference object at a known distance

## Command Line Options

```bash
python rtsp_mask_detector.py [OPTIONS]

Options:
  --model PATH       Path to YOLO model weights (default: runs/detect/train/weights/best.pt)
  --config PATH      Path to camera configuration JSON file
  --rtsp URL         Single RTSP URL for quick start
  --log-level LEVEL  Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)
```

## Architecture

### Multi-Threading Design

1. **Camera Handler Threads**: Each camera runs in its own thread with:
   - Independent frame capture loop
   - YOLO inference processing
   - Automatic reconnection on failures

2. **Display Thread**: Separate thread for visualization:
   - Rate-limited display (10 FPS default)
   - Non-blocking result queue
   - Keyboard input handling

3. **Main Thread**: Orchestrates camera handlers and monitors health

### Performance Features

- **Thread-Safe Operations**: All shared resources protected with locks
- **Queue-Based Communication**: Non-blocking result queues prevent frame drops
- **Buffer Optimization**: Minimal buffer size for low latency
- **Efficient Inference**: Shared YOLO model instance across all cameras

## Detection Classes

The model detects three mask states:
- **with_mask**: Person wearing mask correctly (Green box)
- **without_mask**: Person not wearing mask (Red box)
- **mask_weared_incorrect**: Person wearing mask incorrectly (Orange box)

## Display Information

Each camera window shows:
- **FPS**: Current frames per second
- **Camera ID**: Identifier for the camera
- **Detection Count**: Number of people detected in frame
- **Bounding Boxes**: Color-coded by mask status
- **Distance**: Estimated distance from camera in meters

## Error Handling & Robustness

- **Automatic Reconnection**: Cameras automatically reconnect on stream loss
- **Graceful Degradation**: System continues running if one camera fails
- **Logging**: Comprehensive logging to both file and console
- **Exception Handling**: All critical operations wrapped in try-catch blocks
- **Resource Cleanup**: Proper cleanup on shutdown

## Troubleshooting

### Connection Issues

1. **Check RTSP URL**: Ensure URL is correct and credentials are URL-encoded
2. **Network Connectivity**: Verify camera is accessible from your machine
3. **RTSP Protocol**: Some cameras may require specific RTSP options (add to OpenCV VideoCapture)

### Distance Accuracy

1. **Calibrate Camera Parameters**: Update `focal_length_mm` and `sensor_width_mm` from camera specs
2. **Adjust Face Width**: Fine-tune `known_face_width_cm` based on your demographics
3. **Camera Angle**: Distance calculation assumes front-facing view; angles affect accuracy

### Performance Issues

1. **Reduce Display FPS**: Lower `max_display_fps` in DisplayManager
2. **Lower Resolution**: Reduce `frame_width` and `frame_height` in config
3. **GPU Support**: Ensure CUDA is properly configured for PyTorch

## Logging

Logs are saved to: `mask_detector_YYYYMMDD_HHMMSS.log`

Log levels:
- **DEBUG**: Detailed debugging information
- **INFO**: General information (default)
- **WARNING**: Warning messages
- **ERROR**: Error messages

## Example Usage Scenarios

### Scenario 1: Single Entry Point Monitoring
```bash
python rtsp_mask_detector.py --rtsp "rtsp://admin:pass@192.168.1.25:554/stream1"
```

### Scenario 2: Multi-Building Surveillance
```json
{
  "cameras": [
    {"rtsp_url": "rtsp://...", "camera_id": "Building_A_Entrance"},
    {"rtsp_url": "rtsp://...", "camera_id": "Building_B_Entrance"},
    {"rtsp_url": "rtsp://...", "camera_id": "Building_C_Entrance"}
  ]
}
```

### Scenario 3: Custom Camera Calibration
```json
{
  "rtsp_url": "rtsp://...",
  "camera_id": "Main_Entrance",
  "focal_length_mm": 8.0,  // Wide-angle lens
  "sensor_width_mm": 6.17,  // APS-C sensor
  "frame_width": 1280,
  "frame_height": 720,
  "known_face_width_cm": 14.5
}
```

## Exit

Press `q` in any camera window to exit the application.

## License

Same as the parent Mask-Detector project.
