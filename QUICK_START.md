# Quick Start Guide - Mask Detection System

## Prerequisites

1. **Python 3.8+** installed
2. **Virtual Environment** (recommended)
3. **Trained YOLO Model** at `runs/detect/train/weights/best.pt`

## Step 1: Setup Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

## Step 2: Install Dependencies

```bash
# Install all required packages
pip install -r requirements_rtsp.txt
```

**Note:** If you encounter issues with `dlib` or `face-recognition`, see `FACE_RECOGNITION_SETUP.md` for detailed installation instructions.

## Step 3: Prepare Camera Configuration

### Option A: Add Cameras via Web UI (Recommended)

1. Run the system (see Step 4)
2. Open browser to `http://localhost:5000`
3. Go to **Cameras** tab
4. Click **Add Camera** button
5. Enter:
   - **Camera ID**: e.g., `Camera_1`
   - **RTSP URL**: e.g., `rtsp://admin:password@192.168.1.25:554/Streaming/Channels/101`
   - Adjust advanced settings if needed
6. Click **Add Camera**

You can add up to **4 cameras** via the UI.

### Option B: Use Config File

Create a `config.json` file:

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
    }
  ]
}
```

Or use the example config:

```bash
cp config.example.json config.json
# Edit config.json with your RTSP URLs
```

## Step 4: Run the System

### Basic Command (Headless + Web UI)

```bash
# Make sure you're in the project directory and venv is activated
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows

# Run with web dashboard
python rtsp_mask_detector.py --web
```

### Full Command with Options

```bash
python rtsp_mask_detector.py \
    --model runs/detect/train/weights/best.pt \
    --config config.json \
    --web \
    --web-port 5000 \
    --headless \
    --event-interval 30
```

### Command Line Options

- `--model PATH`: Path to YOLO model (default: `runs/detect/train/weights/best.pt`)
- `--config PATH`: Path to camera config JSON file (optional if using UI)
- `--rtsp URL`: Quick start with single RTSP URL (creates Camera_1)
- `--web`: Enable web dashboard (recommended)
- `--web-port PORT`: Web server port (default: 5000)
- `--headless`: Disable GUI windows (required on macOS, recommended for servers)
- `--event-interval SECONDS`: Minimum seconds between event captures (default: 30)
- `--disable-spatial-dedup`: Disable spatial deduplication
- `--log-level LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

### Examples

**Single RTSP URL (Quick Start):**
```bash
python rtsp_mask_detector.py \
    --rtsp "rtsp://admin:pass@192.168.1.25:554/stream" \
    --web \
    --headless
```

**Multiple Cameras from Config:**
```bash
python rtsp_mask_detector.py \
    --config config.json \
    --web \
    --headless
```

**With Custom Event Interval:**
```bash
python rtsp_mask_detector.py \
    --web \
    --headless \
    --event-interval 60  # Capture events every 60 seconds
```

## Step 5: Access Web Dashboard

1. Open your browser
2. Navigate to: **http://localhost:5000**
3. You'll see the dashboard with tabs:
   - **Live Feed**: Real-time camera streams
   - **Events**: Mask-off event log
   - **Face Management**: Add/edit known faces
   - **Cameras**: Manage cameras (add/remove up to 4)
   - **Analytics**: Compliance statistics and trends

## Step 6: Add Known Faces (Optional)

1. Go to **Face Management** tab
2. Click **Add Person**
3. Enter person's name
4. Upload multiple face images (recommended: 3-5 images)
5. Click **Add Person**

The system will recognize these people when they remove their masks.

## Stopping the System

Press `Ctrl+C` in the terminal to stop the system gracefully.

## Troubleshooting

### Model Not Found
- Make sure `runs/detect/train/weights/best.pt` exists
- Or specify path with `--model PATH`

### OpenCV Display Errors (macOS)
- Always use `--headless` flag on macOS
- Use web dashboard instead

### RTSP Connection Issues
- Verify RTSP URL is correct
- Check camera is accessible on network
- Test RTSP URL with VLC player first
- Ensure camera credentials are correct (URL-encoded if needed)

### Port Already in Use
- Change port: `--web-port 8080`
- Or stop other process using port 5000

### Face Recognition Not Working
- See `FACE_RECOGNITION_SETUP.md` for detailed setup
- Ensure `dlib` and `face-recognition` are installed correctly

## Files Created During Runtime

- `cameras.json`: Camera configurations (auto-saved)
- `events.db`: SQLite database of mask-off events
- `events/`: Directory with event snapshots
- `known_faces/`: Directory with known face images
- `mask_detector_*.log`: Log files

## Features

✅ **Multi-Camera Support**: Up to 4 RTSP cameras simultaneously  
✅ **Real-Time Detection**: Live mask detection on video streams  
✅ **Face Recognition**: Identify known persons  
✅ **Event Logging**: Automatic snapshots and database logging  
✅ **Web Dashboard**: Modern, professional UI  
✅ **Analytics**: Time-based statistics and trends  
✅ **Distance Calculation**: Measure person distance from camera  
✅ **Auto-Reconnection**: Robust handling of camera disconnections  
✅ **Event Throttling**: Prevents duplicate event flooding  

## Need Help?

- Check logs in `mask_detector_*.log` files
- Enable debug logging: `--log-level DEBUG`
- Review `FACE_RECOGNITION_SETUP.md` for face recognition setup
- Review `SETUP_GUIDE.md` for detailed setup instructions

