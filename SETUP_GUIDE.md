# Setup Guide - RTSP Mask Detection System

## Quick Setup (5 minutes)

### Step 1: Install Dependencies
```bash
pip install -r requirements_rtsp.txt
```

### Step 2: Verify Model
Ensure your trained model exists at:
```
runs/detect/train/weights/best.pt
```

### Step 3: Run with Default RTSP URL
```bash
python quick_start.py
```

**OR** using the main script:
```bash
python rtsp_mask_detector.py --rtsp "rtsp://admin:Krishna%40429@192.168.1.25:554/Streaming/Channels/101"
```

## Advanced Setup

### Multiple Cameras

1. **Copy example config:**
```bash
cp config.example.json config.json
```

2. **Edit config.json** with your camera URLs:
```json
{
  "cameras": [
    {
      "rtsp_url": "rtsp://user:pass@192.168.1.25:554/stream",
      "camera_id": "Camera_1",
      ...
    },
    {
      "rtsp_url": "rtsp://user:pass@192.168.1.26:554/stream",
      "camera_id": "Camera_2",
      ...
    }
  ]
}
```

3. **Run:**
```bash
python rtsp_mask_detector.py --config config.json
```

## RTSP URL Format

Your RTSP URL should follow this format:
```
rtsp://username:password@ip_address:port/path
```

**Important:** If your password contains special characters (like `@`, `%`, `#`), they must be URL-encoded:
- `@` → `%40`
- `%` → `%25`
- `#` → `%23`

Example:
- Password: `Krishna@429` → Use: `Krishna%40429` in URL
- Your URL: `rtsp://admin:Krishna%40429@192.168.1.25:554/Streaming/Channels/101`

## Distance Calibration

For accurate distance measurement, calibrate your camera parameters:

1. **Find Camera Specifications:**
   - Focal length (usually in mm, e.g., 8mm, 50mm)
   - Sensor size (e.g., 1/2.7", 1/3", APS-C, Full Frame)

2. **Common Sensor Sizes:**
   - Full Frame: 36mm × 24mm
   - APS-C: 23.6mm × 15.6mm
   - 1/2.7": 6.17mm × 4.63mm
   - 1/3": 4.8mm × 3.6mm

3. **Update config.json:**
```json
{
  "focal_length_mm": 8.0,      // Your camera's focal length
  "sensor_width_mm": 6.17,      // Your camera's sensor width
  "frame_width": 1920,          // Camera resolution width
  "frame_height": 1080,         // Camera resolution height
  "known_face_width_cm": 15.0   // Average face width (14-16cm typical)
}
```

## Testing Distance Accuracy

1. Place a person at a known distance (e.g., 2 meters)
2. Note the displayed distance
3. Adjust `known_face_width_cm` to match:
   - If displayed distance is too high → increase `known_face_width_cm`
   - If displayed distance is too low → decrease `known_face_width_cm`

## Troubleshooting

### "Cannot connect to camera"
- Verify RTSP URL is correct
- Check network connectivity: `ping <camera_ip>`
- Test with VLC: `vlc rtsp://...` to verify stream works
- Check camera credentials

### "Model file not found"
- Ensure model is at: `runs/detect/train/weights/best.pt`
- Or specify custom path: `--model path/to/weights.pt`

### Low FPS / Performance Issues
- Reduce display FPS (edit `DisplayManager` in code)
- Lower camera resolution in config
- Use GPU acceleration (ensure CUDA is installed)
- Process fewer cameras simultaneously

### Distance calculation seems wrong
- Calibrate camera parameters (see Distance Calibration section)
- Ensure person is facing camera (not at an angle)
- Check that `focal_length_mm` and `sensor_width_mm` match your camera

## Performance Tips

1. **For maximum performance:**
   - Use GPU (CUDA-enabled PyTorch)
   - Lower camera resolution (640x480 instead of 1920x1080)
   - Reduce number of displayed windows

2. **For multiple cameras:**
   - Stagger camera connections (already built-in)
   - Monitor system resources (CPU, GPU, network)
   - Consider running on separate machines for many cameras

## System Requirements

- **Minimum:**
  - CPU: 4 cores
  - RAM: 8GB
  - Python 3.8+

- **Recommended:**
  - GPU: NVIDIA GPU with CUDA support
  - RAM: 16GB+
  - Fast network connection for RTSP streams

## Security Notes

- **Never commit config.json** with real credentials to version control
- Use environment variables for sensitive information (future enhancement)
- Keep RTSP credentials secure
- Restrict camera network access if possible

## Next Steps

- Integrate with alerting system (email, SMS, webhooks)
- Add database logging for compliance
- Implement web dashboard
- Add analytics and reporting
- Export detections to CSV/JSON

For more details, see `RTSP_README.md`.
