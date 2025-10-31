# Face Recognition Setup Guide

The system now includes face recognition to identify offenders when they remove their masks.

## Installation

### Step 1: Install Dependencies

```bash
pip install face-recognition dlib
```

**Note:** On macOS, if you encounter issues with dlib, you may need:
```bash
brew install cmake
pip install dlib
```

### Step 2: Create Known Faces Directory

Create a directory structure for known faces:

```bash
mkdir -p known_faces
```

### Step 3: Add Person Directories

For each person you want to recognize, create a subdirectory:

```bash
mkdir known_faces/John_Doe
mkdir known_faces/Jane_Smith
# etc.
```

### Step 4: Add Face Images

Place **multiple face images** (JPG, PNG, BMP) of each person in their directory:

```
known_faces/
  ├── John_Doe/
  │   ├── photo1.jpg
  │   ├── photo2.jpg
  │   └── photo3.jpg
  └── Jane_Smith/
      ├── photo1.jpg
      └── photo2.jpg
```

**Tips for best results:**
- Use 3-5 images per person
- Images should show the person's face clearly
- Different angles and lighting conditions help
- Face should be clearly visible (not too small, not blurry)

### Step 5: Run the System

The face recognizer will automatically load known faces on startup:

```bash
python rtsp_mask_detector.py --rtsp "rtsp://..." --web
```

You should see in the logs:
```
Loaded 3 face(s) for John_Doe
Loaded 2 face(s) for Jane_Smith
Face recognition initialized: 5 faces from 2 people
```

## How It Works

1. **Detection**: When someone is detected without a mask, the system crops their face
2. **Recognition**: The face is compared against all known faces
3. **Identification**: If a match is found (within tolerance), the person's name is recorded
4. **Database**: The recognized name is saved in the SQLite database with the event

## Recognition Tolerance

The default tolerance is 0.6 (lower = stricter):
- **0.4-0.5**: Very strict (fewer false positives, might miss some matches)
- **0.6** (default): Balanced
- **0.7-0.8**: More lenient (might have false positives)

You can adjust tolerance by modifying the `FaceRecognizer` initialization in the code.

## Viewing Recognized Faces

Recognized names appear in:
- **Database**: `events.db` - `recognized_name` column
- **Web Dashboard**: Identity field shows recognized name or "Unknown"
- **Logs**: `🚨 Recognized offender: John_Doe (camera: Camera_1)`

## Adding New Faces

1. Create a new directory: `known_faces/NewPerson/`
2. Add face images to that directory
3. **Option A**: Restart the application (faces will auto-load)
4. **Option B**: Call `face_recognizer.reload_known_faces()` programmatically

## Troubleshooting

### "face_recognition library not available"
- Install: `pip install face-recognition dlib`
- If dlib fails, install cmake first: `brew install cmake` (macOS)

### "No faces found in image"
- Ensure images contain clear faces
- Try using images with only one face visible
- Images should be front-facing or slightly angled

### Low recognition accuracy
- Add more training images per person (3-5 recommended)
- Use images with good lighting
- Ensure faces are clear and not too small
- Adjust tolerance if needed

### Recognition too slow
- Face recognition runs on CPU by default
- Consider reducing number of known faces if performance is an issue
- The recognition happens after capture, so it doesn't slow down detection

## Example Usage

```bash
# Start with face recognition
python rtsp_mask_detector.py \
  --rtsp "rtsp://admin:pass@192.168.1.25:554/stream" \
  --web \
  --web-port 8080 \
  --event-interval 30.0
```

When a known person removes their mask:
- Event is captured with their name
- Logged as: `🚨 Recognized offender: John_Doe`
- Saved to database with `recognized_name` field populated
- Displayed in web dashboard

