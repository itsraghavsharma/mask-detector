# Performance Optimizations for Multi-Camera Support

## Overview
The system has been optimized to handle 4 cameras simultaneously with high FPS (target: 25-30 FPS per camera).

## Key Optimizations Implemented

### 1. Frame Skipping (`process_every_n_frames`)
- **Default**: Process every 2nd frame (`process_every_n_frames: 2`)
- **Benefit**: Reduces inference load by 50% while maintaining smooth video
- **Trade-off**: Minimal detection accuracy impact (objects are detected on every other frame)
- **Adjustable**: Can be set per camera in config

### 2. Inference Resolution Reduction (`inference_size`)
- **Default**: Resize frames to 640px for inference (`inference_size: 640`)
- **Benefit**: ~4x faster inference vs full resolution (1920x1080)
- **Accuracy**: Bounding boxes are scaled back to original resolution
- **Trade-off**: Slight accuracy reduction on very small faces, but minimal

### 3. Confidence Threshold Filtering (`conf_threshold`)
- **Default**: 0.25 (filters low-confidence detections early)
- **Benefit**: Reduces post-processing overhead
- **Adjustable**: Increase for fewer false positives, decrease for more sensitivity

### 4. Optimized Queue Management
- **Queue Size**: Scales with camera count (50 frames per camera)
- **Behavior**: Ring buffer - drops oldest when full (non-blocking)
- **Benefit**: Prevents memory buildup and frame blocking

### 5. Batch Tensor Operations
- **Optimization**: Convert all detections to numpy in batch (not loop)
- **Benefit**: Faster tensor-to-numpy conversion

### 6. Frame Reference vs Copy
- **Optimization**: Use frame references instead of copying
- **Benefit**: Reduces memory allocation overhead

### 7. OpenCV RTSP Optimizations
- Buffer size: Set to 1 (minimal buffering)
- FPS: Request 30 FPS if camera supports
- RGB conversion: Explicitly set

### 8. Staggered Camera Connections
- **Optimization**: Start cameras with 0.2s delay between each
- **Benefit**: Reduces network congestion and resource contention

### 9. Display FPS Optimization
- **Increased**: From 10 FPS to 15 FPS display refresh
- **Benefit**: Smoother video playback

## Expected Performance

### With 4 Cameras:
- **Target FPS per camera**: 25-30 FPS
- **Processing**: Every 2nd frame (effective 12-15 detections/sec)
- **Inference time**: ~30-40ms per frame (at 640px)
- **Total inference load**: ~120-160ms across 4 cameras

### Performance Scaling:

| Cameras | FPS/Camera | Total FPS | Inference Load |
|---------|------------|-----------|----------------|
| 1       | 30         | 30        | ~40ms         |
| 2       | 28-30      | 56-60     | ~80ms         |
| 3       | 26-28      | 78-84     | ~120ms        |
| 4       | 24-26      | 96-104    | ~160ms        |

## Configuration Options

### Per-Camera Settings (in config.json or UI):

```json
{
  "cameras": [
    {
      "rtsp_url": "...",
      "camera_id": "Camera_1",
      "process_every_n_frames": 2,    // Process every 2nd frame
      "inference_size": 640,           // Resize to 640px for inference
      "conf_threshold": 0.25           // Confidence threshold
    }
  ]
}
```

### Tuning Guidelines:

**For Higher FPS (Lower Accuracy):**
- Increase `process_every_n_frames` to 3 or 4
- Decrease `inference_size` to 512 or 416
- Increase `conf_threshold` to 0.35

**For Higher Accuracy (Lower FPS):**
- Set `process_every_n_frames` to 1 (process all frames)
- Increase `inference_size` to 832 or 1024
- Decrease `conf_threshold` to 0.20

**For Balanced Performance (Recommended):**
- `process_every_n_frames`: 2
- `inference_size`: 640
- `conf_threshold`: 0.25

## Hardware Recommendations

### Minimum Requirements:
- **CPU**: 4+ cores (Intel i5 or AMD Ryzen 5 equivalent)
- **RAM**: 8GB
- **GPU**: Optional but recommended (CUDA-capable for faster inference)

### Recommended:
- **CPU**: 6+ cores (Intel i7 or AMD Ryzen 7)
- **RAM**: 16GB
- **GPU**: NVIDIA GPU with 4GB+ VRAM (10x faster inference)

## Monitoring Performance

Check logs for performance metrics:
```
Started camera handler for Camera_1 (process_every=2 frames, inference_size=640)
```

Monitor FPS in web dashboard or logs to ensure targets are met.

## Troubleshooting Low FPS

1. **Check CPU/GPU Usage**: Use `htop` or `nvidia-smi`
2. **Reduce Processing Frequency**: Increase `process_every_n_frames`
3. **Lower Inference Resolution**: Decrease `inference_size` to 512
4. **Check Network**: Ensure RTSP streams aren't bandwidth-limited
5. **Reduce Camera Resolution**: Lower `frame_width`/`frame_height` in config

## Advanced: GPU Acceleration

If you have CUDA-capable GPU:
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

YOLO will automatically use GPU if available, providing 5-10x speedup.

