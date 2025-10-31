"""
RTSP Mask Detection System with Distance Calculation
Production-ready multi-camera mask detection system with RTSP streaming support.
"""

import cv2
import numpy as np
import threading
import time
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from queue import Queue, Empty
from ultralytics import YOLO
import argparse
import json
import os
from datetime import datetime
import sqlite3
from pathlib import Path
import base64
from io import BytesIO
try:
    from flask import Flask, Response, render_template_string, jsonify, request
    from werkzeug.utils import secure_filename
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# Face recognition support
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False


@dataclass
class CameraConfig:
    """Configuration for a single camera"""
    rtsp_url: str
    camera_id: str
    focal_length_mm: float = 50.0  # Default focal length in mm
    sensor_width_mm: float = 36.0  # Default sensor width in mm (full frame)
    frame_width: int = 1920  # Camera resolution width
    frame_height: int = 1080  # Camera resolution height
    known_face_width_cm: float = 15.0  # Average face width in cm for distance calculation
    frame_timeout: float = 10.0  # Seconds without frames before reconnecting (sleep/wake detection)
    max_consecutive_failures: int = 5  # Max failures before triggering reconnection
    # Performance optimization parameters
    process_every_n_frames: int = 3  # Process every Nth frame (1 = all frames, 3 = every 3rd frame for better FPS)
    inference_size: int = 512  # Resize frame to this size for inference (faster - optimized for multi-camera)
    conf_threshold: float = 0.25  # Minimum confidence threshold for detections


@dataclass
class DetectionResult:
    """Result of a detection"""
    frame: np.ndarray
    detections: List[Dict]
    timestamp: float
    camera_id: str
    fps: float
    # raw fields to help event logging
    
class EventLogger:
    """Logs mask-off events with snapshots to SQLite and disk."""
    
    def __init__(self, db_path: str = "events.db", image_dir: str = "events", 
                 min_capture_interval: float = 30.0, enable_spatial_dedup: bool = True):
        """
        Args:
            db_path: Path to SQLite database
            image_dir: Directory to save event images
            min_capture_interval: Minimum seconds between captures per camera (default: 30s)
            enable_spatial_dedup: Enable spatial deduplication (skip if same location recently)
        """
        self.db_path = db_path
        self.image_dir = image_dir
        self.min_capture_interval = min_capture_interval
        self.enable_spatial_dedup = enable_spatial_dedup
        Path(self.image_dir).mkdir(parents=True, exist_ok=True)
        self._init_db()
        self.lock = threading.Lock()
        
        # Throttling: track last capture time per camera
        self.last_capture_time = {}  # camera_id -> timestamp
        
        # Spatial deduplication: track recent bbox locations per camera
        # Format: camera_id -> [(bbox_center_x, bbox_center_y, timestamp), ...]
        self.recent_locations = {}  # camera_id -> list of recent (x, y, timestamp)
        self.spatial_dedup_radius = 50.0  # pixels - same person if within this radius
        self.spatial_dedup_window = 60.0  # seconds - check locations within this window
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    class TEXT NOT NULL,
                    confidence REAL,
                    distance_m REAL,
                    image_path TEXT NOT NULL,
                    recognized_name TEXT,
                    is_false_claim INTEGER DEFAULT 0
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_name ON events(recognized_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_class ON events(class)")
            # Add is_false_claim column if it doesn't exist (for existing databases)
            try:
                conn.execute("ALTER TABLE events ADD COLUMN is_false_claim INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass  # Column already exists
        finally:
            conn.commit()
            conn.close()
    
    def save_snapshot(self, frame: np.ndarray, camera_id: str) -> str:
        ts_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f"{camera_id}_{ts_str}.jpg"
        out_path = os.path.join(self.image_dir, filename)
        cv2.imwrite(out_path, frame)
        return out_path
    
    def should_capture_event(self, camera_id: str, bbox: Optional[List[float]] = None) -> bool:
        """
        Check if we should capture this event based on throttling rules.
        
        Args:
            camera_id: Camera identifier
            bbox: Bounding box [x1, y1, x2, y2] for spatial deduplication
            
        Returns:
            True if event should be captured, False if throttled
        """
        current_time = time.time()
        
        # Check time-based throttling (minimum interval between captures)
        last_time = self.last_capture_time.get(camera_id, 0)
        time_since_last = current_time - last_time
        if time_since_last < self.min_capture_interval:
            return False
        
        # Spatial deduplication: skip if same person detected recently in same location
        if self.enable_spatial_dedup and bbox is not None:
            # Calculate bbox center
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            
            # Check recent locations for this camera
            if camera_id in self.recent_locations:
                recent = self.recent_locations[camera_id]
                # Remove old locations (outside time window)
                cutoff_time = current_time - self.spatial_dedup_window
                recent[:] = [loc for loc in recent if loc[2] > cutoff_time]
                
                # Check if any recent location is within radius
                for loc_x, loc_y, loc_time in recent:
                    distance = np.sqrt((center_x - loc_x)**2 + (center_y - loc_y)**2)
                    if distance < self.spatial_dedup_radius:
                        # Same person detected recently in same area - skip
                        return False
            
            # Add this location to recent list
            if camera_id not in self.recent_locations:
                self.recent_locations[camera_id] = []
            self.recent_locations[camera_id].append((center_x, center_y, current_time))
        
        # Update last capture time
        self.last_capture_time[camera_id] = current_time
        return True
    
    def log_event(self, camera_id: str, cls: str, confidence: float, distance_m: Optional[float], image_path: str, recognized_name: Optional[str] = None):
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute(
                    "INSERT INTO events (ts, camera_id, class, confidence, distance_m, image_path, recognized_name, is_false_claim) VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
                    (
                        datetime.now().isoformat(timespec='seconds'),
                        camera_id,
                        cls,
                        float(confidence) if confidence is not None else None,
                        float(distance_m) if distance_m is not None else None,
                        image_path,
                        recognized_name
                    )
                )
            finally:
                conn.commit()
                conn.close()
    
    def mark_false_claim(self, event_id: int):
        """Mark an event as false claim"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute("UPDATE events SET is_false_claim = 1 WHERE id = ?", (event_id,))
            finally:
                conn.commit()
                conn.close()


class CameraManager:
    """Manages camera configurations in SQLite database"""
    
    def __init__(self, db_path: str = "cameras.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()
    
    def _init_db(self):
        """Initialize camera database table"""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cameras (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT UNIQUE NOT NULL,
                    rtsp_url TEXT NOT NULL,
                    focal_length_mm REAL DEFAULT 50.0,
                    sensor_width_mm REAL DEFAULT 36.0,
                    frame_width INTEGER DEFAULT 1920,
                    frame_height INTEGER DEFAULT 1080,
                    known_face_width_cm REAL DEFAULT 15.0,
                    frame_timeout REAL DEFAULT 10.0,
                    max_consecutive_failures INTEGER DEFAULT 5,
                    process_every_n_frames INTEGER DEFAULT 3,
                    inference_size INTEGER DEFAULT 512,
                    conf_threshold REAL DEFAULT 0.25,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cameras_id ON cameras(camera_id)")
            conn.commit()
        finally:
            conn.close()
    
    def save_camera(self, config: CameraConfig) -> bool:
        """Save or update a camera configuration"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                now = datetime.now().isoformat()
                # Check if camera exists
                existing = conn.execute(
                    "SELECT id FROM cameras WHERE camera_id = ?", (config.camera_id,)
                ).fetchone()
                
                if existing:
                    # Update existing
                    conn.execute(
                        """
                        UPDATE cameras SET
                            rtsp_url = ?,
                            focal_length_mm = ?,
                            sensor_width_mm = ?,
                            frame_width = ?,
                            frame_height = ?,
                            known_face_width_cm = ?,
                            frame_timeout = ?,
                            max_consecutive_failures = ?,
                            process_every_n_frames = ?,
                            inference_size = ?,
                            conf_threshold = ?,
                            updated_at = ?
                        WHERE camera_id = ?
                        """,
                        (
                            config.rtsp_url,
                            config.focal_length_mm,
                            config.sensor_width_mm,
                            config.frame_width,
                            config.frame_height,
                            config.known_face_width_cm,
                            config.frame_timeout,
                            config.max_consecutive_failures,
                            getattr(config, 'process_every_n_frames', 3),
                            getattr(config, 'inference_size', 512),
                            getattr(config, 'conf_threshold', 0.25),
                            now,
                            config.camera_id
                        )
                    )
                else:
                    # Insert new
                    conn.execute(
                        """
                        INSERT INTO cameras (
                            camera_id, rtsp_url, focal_length_mm, sensor_width_mm,
                            frame_width, frame_height, known_face_width_cm,
                            frame_timeout, max_consecutive_failures,
                            process_every_n_frames, inference_size, conf_threshold,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            config.camera_id,
                            config.rtsp_url,
                            config.focal_length_mm,
                            config.sensor_width_mm,
                            config.frame_width,
                            config.frame_height,
                            config.known_face_width_cm,
                            config.frame_timeout,
                            config.max_consecutive_failures,
                            getattr(config, 'process_every_n_frames', 3),
                            getattr(config, 'inference_size', 512),
                            getattr(config, 'conf_threshold', 0.25),
                            now,
                            now
                        )
                    )
                conn.commit()
                return True
            except sqlite3.IntegrityError as e:
                logging.error(f"Camera {config.camera_id} already exists: {e}")
                return False
            except Exception as e:
                logging.error(f"Error saving camera {config.camera_id}: {e}")
                return False
            finally:
                conn.close()
    
    def delete_camera(self, camera_id: str) -> bool:
        """Delete a camera configuration"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute("DELETE FROM cameras WHERE camera_id = ?", (camera_id,))
                conn.commit()
                return cursor.rowcount > 0
            except Exception as e:
                logging.error(f"Error deleting camera {camera_id}: {e}")
                return False
            finally:
                conn.close()
    
    def load_all_cameras(self) -> List[CameraConfig]:
        """Load all camera configurations from database"""
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute("SELECT * FROM cameras ORDER BY camera_id").fetchall()
            configs = []
            for row in rows:
                # Handle both new format (with performance params) and old format
                config = CameraConfig(
                    camera_id=row[1],
                    rtsp_url=row[2],
                    focal_length_mm=row[3] or 50.0,
                    sensor_width_mm=row[4] or 36.0,
                    frame_width=row[5] or 1920,
                    frame_height=row[6] or 1080,
                    known_face_width_cm=row[7] or 15.0,
                    frame_timeout=row[8] or 10.0,
                    max_consecutive_failures=row[9] or 5,
                    process_every_n_frames=row[10] if len(row) > 10 and row[10] is not None else 3,
                    inference_size=row[11] if len(row) > 11 and row[11] is not None else 512,
                    conf_threshold=row[12] if len(row) > 12 and row[12] is not None else 0.25
                )
                configs.append(config)
            return configs
        except Exception as e:
            logging.error(f"Error loading cameras from database: {e}")
            return []
        finally:
            conn.close()
    
    def get_camera(self, camera_id: str) -> Optional[CameraConfig]:
        """Get a specific camera configuration"""
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute("SELECT * FROM cameras WHERE camera_id = ?", (camera_id,)).fetchone()
            if not row:
                return None
            
            return CameraConfig(
                camera_id=row[1],
                rtsp_url=row[2],
                focal_length_mm=row[3] or 50.0,
                sensor_width_mm=row[4] or 36.0,
                frame_width=row[5] or 1920,
                frame_height=row[6] or 1080,
                known_face_width_cm=row[7] or 15.0,
                frame_timeout=row[8] or 10.0,
                max_consecutive_failures=row[9] or 5,
                process_every_n_frames=row[10] if len(row) > 10 and row[10] is not None else 3,
                inference_size=row[11] if len(row) > 11 and row[11] is not None else 512,
                conf_threshold=row[12] if len(row) > 12 and row[12] is not None else 0.25
            )
        except Exception as e:
            logging.error(f"Error loading camera {camera_id}: {e}")
            return None
        finally:
            conn.close()


class FaceRecognizer:
    """Face recognition for identifying offenders using face_recognition library"""
    
    def __init__(self, known_faces_dir: str = "known_faces", tolerance: float = 0.6):
        """
        Args:
            known_faces_dir: Directory containing known face images
                           Structure: known_faces/PersonName/image1.jpg, image2.jpg, etc.
            tolerance: Lower values make recognition more strict (default: 0.6)
        """
        self.known_faces_dir = known_faces_dir
        self.tolerance = tolerance
        self.known_face_encodings = []
        self.known_face_names = []
        self.lock = threading.Lock()
        self.load_known_faces()
    
    def load_known_faces(self):
        """Load known faces from directory structure"""
        if not FACE_RECOGNITION_AVAILABLE:
            logging.warning("face_recognition library not available. Install with: pip install face-recognition")
            logging.warning("Face recognition will be disabled.")
            return
        
        if not os.path.exists(self.known_faces_dir):
            logging.info(f"Known faces directory '{self.known_faces_dir}' not found.")
            logging.info("To enable face recognition:")
            logging.info("  1. Create directory: known_faces/")
            logging.info("  2. Create subdirectories: known_faces/PersonName/")
            logging.info("  3. Add face images to each person's directory")
            logging.info("  4. Restart the application")
            return
        
        self.known_face_encodings = []
        self.known_face_names = []
        loaded_count = 0
        
        # Walk through known faces directory
        for person_name in sorted(os.listdir(self.known_faces_dir)):
            person_dir = os.path.join(self.known_faces_dir, person_name)
            if not os.path.isdir(person_dir):
                continue
            
            image_count = 0
            
            # Load images for this person
            for filename in sorted(os.listdir(person_dir)):
                if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                
                image_path = os.path.join(person_dir, filename)
                try:
                    # Load image using face_recognition
                    image = face_recognition.load_image_file(image_path)
                    
                    # Find face encodings (can have multiple faces in one image)
                    encodings = face_recognition.face_encodings(image)
                    
                    if len(encodings) > 0:
                        # Use first face encoding found
                        self.known_face_encodings.append(encodings[0])
                        self.known_face_names.append(person_name)
                        image_count += 1
                        loaded_count += 1
                    else:
                        logging.debug(f"No face found in {image_path}")
                        
                except Exception as e:
                    logging.warning(f"Error processing {image_path}: {e}")
                    continue
            
            if image_count > 0:
                logging.info(f"Loaded {image_count} face(s) for {person_name}")
        
        if loaded_count > 0:
            logging.info(f"Face recognition initialized: {loaded_count} faces from {len(set(self.known_face_names))} people")
        else:
            logging.info("No faces loaded. Face recognition will return 'Unknown'")
    
    def recognize_face(self, face_image: np.ndarray) -> Optional[str]:
        """
        Recognize a face in the given image
        
        Args:
            face_image: BGR image (from OpenCV) or RGB image array
            
        Returns:
            Recognized person name, or None if not recognized
        """
        if not FACE_RECOGNITION_AVAILABLE or len(self.known_face_encodings) == 0:
            return None
        
        try:
            with self.lock:
                # Convert BGR to RGB (face_recognition uses RGB)
                if len(face_image.shape) == 3:
                    rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                else:
                    rgb_image = face_image
                
                # Find face locations and encodings
                face_locations = face_recognition.face_locations(rgb_image)
                face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
                
                if len(face_encodings) == 0:
                    return None
                
                # Use first face found
                face_encoding = face_encodings[0]
                
                # Compare with known faces
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, 
                    face_encoding, 
                    tolerance=self.tolerance
                )
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, 
                    face_encoding
                )
                
                # Find best match
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    # Check if distance is acceptable (lower is better)
                    distance = face_distances[best_match_index]
                    # Lower tolerance means stricter matching
                    if distance <= self.tolerance:
                        recognized_name = self.known_face_names[best_match_index]
                        logging.debug(f"Recognized: {recognized_name} (distance: {distance:.3f})")
                        return recognized_name
                
                return None
                
        except Exception as e:
            logging.warning(f"Error during face recognition: {e}")
            return None
    
    def reload_known_faces(self):
        """Reload known faces from directory (useful for adding new faces without restart)"""
        logging.info("Reloading known faces...")
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()


class DistanceCalculator:
    """Calculate distance from camera using face bounding box"""
    
    def __init__(self, camera_config: CameraConfig):
        self.config = camera_config
        # Calculate focal length in pixels
        self.focal_length_pixels = (camera_config.focal_length_mm * 
                                   camera_config.frame_width) / camera_config.sensor_width_mm
    
    def calculate_distance(self, bbox_width_pixels: float) -> Optional[float]:
        """
        Calculate distance using the formula:
        distance = (known_width * focal_length) / pixel_width
        
        Args:
            bbox_width_pixels: Width of bounding box in pixels
            
        Returns:
            Distance in centimeters, or None if calculation fails
        """
        if bbox_width_pixels <= 0:
            return None
        
        try:
            # Convert known face width from cm to pixels (approximate)
            # distance_cm = (known_width_cm * focal_length_pixels) / pixel_width
            distance_cm = (self.config.known_face_width_cm * self.focal_length_pixels) / bbox_width_pixels
            
            # Clamp distance to reasonable values (0.5m to 10m)
            distance_cm = max(50, min(1000, distance_cm))
            
            return distance_cm
        except Exception as e:
            logging.error(f"Distance calculation error: {e}")
            return None


class CameraHandler:
    """Thread-safe camera handler with robust reconnection logic"""
    
    def __init__(self, config: CameraConfig, model: YOLO, result_queue: Queue):
        self.config = config
        self.model = model
        self.result_queue = result_queue
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0
        self.lock = threading.Lock()
        self.distance_calc = DistanceCalculator(config)
        
        # Performance optimization parameters (optimized defaults for multi-camera)
        self.process_every_n = getattr(config, 'process_every_n_frames', 3)
        self.inference_size = getattr(config, 'inference_size', 512)
        self.conf_threshold = getattr(config, 'conf_threshold', 0.25)
        self.frame_skip_counter = 0
        self.last_detection_frame = None  # Store last frame for display even if not processed
        
        # Robustness parameters from config
        self.last_frame_time = time.time()
        self.frame_timeout = config.frame_timeout
        self.consecutive_failures = 0
        self.max_consecutive_failures = config.max_consecutive_failures
        self.reconnect_attempts = 0
        self.max_reconnect_delay = 60.0  # Maximum delay between reconnect attempts (exponential backoff)
        self.connection_healthy = False
        self.total_reconnects = 0
        
    def connect(self, max_retries: int = 5, retry_delay: float = 2.0, use_backoff: bool = False) -> bool:
        """Connect to RTSP stream with enhanced retry logic and exponential backoff"""
        for attempt in range(max_retries):
            try:
                # Calculate delay with exponential backoff if enabled
                if use_backoff and attempt > 0:
                    delay = min(retry_delay * (2 ** attempt), self.max_reconnect_delay)
                    logging.info(f"Waiting {delay:.1f}s before retry (exponential backoff)...")
                    time.sleep(delay)
                elif attempt > 0:
                    time.sleep(retry_delay)
                
                logging.info(f"Connecting to camera {self.config.camera_id} (attempt {attempt + 1}/{max_retries})...")
                
                # Clean up existing connection
                self._cleanup_connection()
                
                # Create new connection with timeout settings
                self.cap = cv2.VideoCapture(self.config.rtsp_url)
                
                # Set OpenCV properties for optimized RTSP handling
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer for lowest latency
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                # Additional optimizations for performance
                self.cap.set(cv2.CAP_PROP_FPS, 30)  # Request 30 FPS if camera supports it
                try:
                    self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)  # Ensure RGB conversion
                except:
                    pass
                
                if self.cap.isOpened():
                    # Test read with timeout
                    test_start = time.time()
                    ret, frame = self.cap.read()
                    test_duration = time.time() - test_start
                    
                    if ret and frame is not None:
                        # Check if frame is valid (has dimensions)
                        if frame.size > 0 and frame.shape[0] > 0 and frame.shape[1] > 0:
                            logging.info(f"Successfully connected to camera {self.config.camera_id} (test read took {test_duration:.2f}s)")
                            self.connection_healthy = True
                            self.consecutive_failures = 0
                            self.reconnect_attempts = 0
                            self.last_frame_time = time.time()
                            return True
                        else:
                            logging.warning(f"Received empty frame from camera {self.config.camera_id}")
                            self._cleanup_connection()
                    else:
                        logging.warning(f"Failed to read test frame from camera {self.config.camera_id}")
                        self._cleanup_connection()
                else:
                    logging.warning(f"Failed to open camera {self.config.camera_id}")
                    self._cleanup_connection()
                    
            except cv2.error as e:
                error_msg = str(e).lower()
                if 'timeout' in error_msg or 'timed out' in error_msg:
                    logging.warning(f"Connection timeout to camera {self.config.camera_id} (attempt {attempt + 1})")
                elif 'network' in error_msg or 'connection refused' in error_msg:
                    logging.warning(f"Network error connecting to camera {self.config.camera_id} (attempt {attempt + 1})")
                else:
                    logging.error(f"OpenCV error during connection attempt {attempt + 1}: {e}")
                self._cleanup_connection()
            except OSError as e:
                # Network/system errors
                if 'Network is unreachable' in str(e) or 'Connection refused' in str(e):
                    logging.warning(f"Network unreachable for camera {self.config.camera_id} (attempt {attempt + 1})")
                else:
                    logging.error(f"OS error during connection attempt {attempt + 1}: {e}")
                self._cleanup_connection()
            except Exception as e:
                logging.error(f"Unexpected error during connection attempt {attempt + 1}: {e}", exc_info=True)
                self._cleanup_connection()
        
        logging.error(f"Failed to connect to camera {self.config.camera_id} after {max_retries} attempts")
        self.connection_healthy = False
        return False
    
    def _cleanup_connection(self):
        """Safely cleanup camera connection"""
        try:
            if self.cap:
                with self.lock:
                    if self.cap.isOpened():
                        self.cap.release()
                    self.cap = None
        except Exception as e:
            logging.warning(f"Error during connection cleanup: {e}")
    
    def disconnect(self):
        """Disconnect from camera"""
        try:
            with self.lock:
                if self.cap:
                    if self.cap.isOpened():
                        self.cap.release()
                    self.cap = None
            self.connection_healthy = False
        except Exception as e:
            logging.warning(f"Error during disconnection: {e}")
    
    def _process_frame(self, frame: np.ndarray) -> DetectionResult:
        """Process frame through YOLO model with optimizations"""
        try:
            original_shape = frame.shape[:2]
            original_height, original_width = original_shape
            
            # Resize frame for inference (much faster) - maintain aspect ratio
            if self.inference_size and (original_width > self.inference_size or original_height > self.inference_size):
                scale = self.inference_size / max(original_width, original_height)
                inference_width = int(original_width * scale)
                inference_height = int(original_height * scale)
                inference_frame = cv2.resize(frame, (inference_width, inference_height), interpolation=cv2.INTER_LINEAR)
            else:
                inference_frame = frame
                scale = 1.0
            
            # Run inference with confidence threshold and optimized settings
            results = self.model(
                inference_frame, 
                verbose=False,
                conf=self.conf_threshold,  # Filter low-confidence detections early
                imgsz=self.inference_size,  # Optimize for this size
                half=False,  # Use FP32 for stability (FP16 can be faster but may have issues)
                device='',  # Let YOLO choose device automatically
                max_det=100  # Limit max detections for speed
            )
            
            # Parse detections and scale bboxes back to original resolution
            detections = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                # Batch convert to numpy (faster than loop)
                boxes_xyxy = boxes.xyxy.cpu().numpy()
                boxes_conf = boxes.conf.cpu().numpy()
                boxes_cls = boxes.cls.cpu().numpy().astype(int)
                
                for i in range(len(boxes)):
                    if boxes_conf[i] < self.conf_threshold:
                        continue
                    
                    # Scale bbox back to original resolution
                    box = boxes_xyxy[i] / scale
                    conf = float(boxes_conf[i])
                    cls = boxes_cls[i]
                    class_name = self.model.names[cls]
                    
                    # Clamp to original frame bounds
                    box[0] = max(0, min(box[0], original_width))
                    box[1] = max(0, min(box[1], original_height))
                    box[2] = max(0, min(box[2], original_width))
                    box[3] = max(0, min(box[3], original_height))
                    
                    # Calculate bounding box dimensions
                    bbox_width = box[2] - box[0]
                    bbox_height = box[3] - box[1]
                    
                    # Calculate distance using original resolution bbox
                    distance_cm = self.distance_calc.calculate_distance(bbox_width)
                    distance_m = distance_cm / 100.0 if distance_cm else None
                    
                    detection = {
                        'class': class_name,
                        'confidence': conf,
                        'bbox': [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                        'distance_cm': distance_cm,
                        'distance_m': distance_m,
                        'bbox_width': float(bbox_width),
                        'bbox_height': float(bbox_height)
                    }
                    detections.append(detection)
            
            # Update FPS
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:
                self.fps = self.frame_count / (current_time - self.last_fps_time)
                self.frame_count = 0
                self.last_fps_time = current_time
            
            # Use original frame for display (not resized inference frame)
            return DetectionResult(
                frame=frame,  # No copy - use reference for better performance
                detections=detections,
                timestamp=time.time(),
                camera_id=self.config.camera_id,
                fps=self.fps
            )
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            return DetectionResult(
                frame=frame,
                detections=[],
                timestamp=time.time(),
                camera_id=self.config.camera_id,
                fps=self.fps
            )
    
    def _capture_loop(self):
        """Main capture loop with robust error handling and auto-reconnection"""
        while self.running:
            try:
                # Check if connection is still valid
                connection_valid = False
                with self.lock:
                    if self.cap and self.cap.isOpened():
                        connection_valid = True
                
                if not connection_valid:
                    logging.warning(f"Camera {self.config.camera_id} connection lost, attempting reconnection...")
                    self._reconnect_with_backoff()
                    continue
                
                # Read frame with timeout handling
                frame_read_start = time.time()
                ret = False
                frame = None
                
                try:
                    with self.lock:
                        if self.cap and self.cap.isOpened():
                            ret, frame = self.cap.read()
                except Exception as e:
                    logging.error(f"Exception during frame read for {self.config.camera_id}: {e}")
                    ret = False
                    frame = None
                
                # Check if frame read took too long (possible hang)
                frame_read_duration = time.time() - frame_read_start
                if frame_read_duration > 5.0:
                    logging.warning(f"Frame read took {frame_read_duration:.2f}s (possible hang) for {self.config.camera_id}")
                    self.consecutive_failures += 1
                    if self.consecutive_failures >= self.max_consecutive_failures:
                        logging.error(f"Too many slow reads for {self.config.camera_id}, reconnecting...")
                        self._reconnect_with_backoff()
                    continue
                
                if not ret or frame is None:
                    self.consecutive_failures += 1
                    if self.consecutive_failures >= self.max_consecutive_failures:
                        logging.warning(f"Too many consecutive read failures for {self.config.camera_id}, reconnecting...")
                        self._reconnect_with_backoff()
                    else:
                        logging.debug(f"Failed to read frame from camera {self.config.camera_id} (failure {self.consecutive_failures}/{self.max_consecutive_failures})")
                        time.sleep(0.1)
                    continue
                
                # Validate frame
                if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
                    logging.warning(f"Received invalid frame from {self.config.camera_id}")
                    self.consecutive_failures += 1
                    time.sleep(0.1)
                    continue
                
                # Check for frame timeout (laptop sleep detection) BEFORE updating timestamp
                current_time = time.time()
                time_since_last_frame = current_time - self.last_frame_time
                if time_since_last_frame > self.frame_timeout:
                    logging.warning(f"No frames received for {time_since_last_frame:.1f}s from {self.config.camera_id} (possible sleep/wake), reconnecting...")
                    self._reconnect_with_backoff()
                    continue
                
                # Frame is valid - reset failure counter and update timestamp
                self.consecutive_failures = 0
                self.last_frame_time = current_time
                self.connection_healthy = True
                
                # Frame skipping optimization: process every Nth frame
                self.frame_skip_counter += 1
                should_process = (self.frame_skip_counter % self.process_every_n == 0)
                
                if should_process:
                    # Process frame through model
                    result = self._process_frame(frame)
                else:
                    # Skip detection but still provide frame for display
                    # Use last detection results if available, otherwise empty
                    self.last_detection_frame = frame
                    # Update FPS counter even for skipped frames
                    self.frame_count += 1
                    current_time_check = time.time()
                    if current_time_check - self.last_fps_time >= 1.0:
                        self.fps = self.frame_count / (current_time_check - self.last_fps_time)
                        self.frame_count = 0
                        self.last_fps_time = current_time_check
                    
                    # Create result with no detections but with frame for display
                    result = DetectionResult(
                        frame=frame,
                        detections=[],  # No new detections on skipped frames
                        timestamp=time.time(),
                        camera_id=self.config.camera_id,
                        fps=self.fps
                    )
                
                # Add to queue (non-blocking, drop if full for better performance)
                try:
                    self.result_queue.put_nowait(result)
                except:
                    # Queue full, drop oldest entry and add new one (ring buffer behavior)
                    try:
                        # Try to get one item without blocking
                        self.result_queue.get_nowait()
                        # Now put the new one
                        self.result_queue.put_nowait(result)
                    except:
                        # If still can't, just skip this frame
                        pass
                
            except KeyboardInterrupt:
                logging.info(f"Capture loop interrupted for {self.config.camera_id}")
                break
            except Exception as e:
                logging.error(f"Unexpected error in capture loop for camera {self.config.camera_id}: {e}", exc_info=True)
                self.consecutive_failures += 1
                
                # If too many errors, try reconnecting
                if self.consecutive_failures >= self.max_consecutive_failures:
                    logging.error(f"Too many errors for {self.config.camera_id}, attempting reconnection...")
                    self._reconnect_with_backoff()
                else:
                    time.sleep(0.5)  # Brief pause before retry
    
    def _reconnect_with_backoff(self):
        """Reconnect with exponential backoff and health checks"""
        self.connection_healthy = False
        self.reconnect_attempts += 1
        self.total_reconnects += 1
        
        logging.info(f"Reconnection attempt {self.reconnect_attempts} for camera {self.config.camera_id}")
        
        # Cleanup existing connection
        self._cleanup_connection()
        
        # Wait before reconnecting (exponential backoff)
        if self.reconnect_attempts > 1:
            backoff_delay = min(2.0 * (2 ** (self.reconnect_attempts - 2)), self.max_reconnect_delay)
            logging.info(f"Waiting {backoff_delay:.1f}s before reconnection (backoff)...")
            time.sleep(backoff_delay)
        
        # Attempt reconnection
        if self.connect(max_retries=3, retry_delay=2.0, use_backoff=True):
            logging.info(f"Successfully reconnected to camera {self.config.camera_id} (total reconnects: {self.total_reconnects})")
            self.reconnect_attempts = 0  # Reset on successful connection
        else:
            logging.warning(f"Reconnection failed for camera {self.config.camera_id}, will retry...")
            # Will retry on next loop iteration
    
    def start(self):
        """Start camera capture in separate thread with enhanced connection handling"""
        if self.running:
            logging.warning(f"Camera {self.config.camera_id} is already running")
            return
        
        # Reset connection state
        self.consecutive_failures = 0
        self.reconnect_attempts = 0
        self.last_frame_time = time.time()
        self.frame_skip_counter = 0
        
        if not self.connect(max_retries=5, retry_delay=2.0, use_backoff=False):
            logging.error(f"Cannot start camera {self.config.camera_id}: initial connection failed")
            logging.info(f"Will continue attempting connection in background...")
            # Don't return - let the capture loop handle reconnection
        
        self.running = True
        self.thread = threading.Thread(
            target=self._capture_loop, 
            daemon=True, 
            name=f"CameraHandler-{self.config.camera_id}"
        )
        # Set thread priority if available (platform specific)
        self.thread.start()
        logging.info(f"Started camera handler for {self.config.camera_id} (process_every={self.process_every_n} frames, inference_size={self.inference_size})")
    
    def get_health_status(self) -> Dict:
        """Get current health status of the camera handler"""
        time_since_last_frame = time.time() - self.last_frame_time
        is_connected = False
        try:
            with self.lock:
                is_connected = self.cap is not None and self.cap.isOpened()
        except:
            pass
        
        return {
            'running': self.running,
            'connection_healthy': self.connection_healthy,
            'time_since_last_frame': time_since_last_frame,
            'consecutive_failures': self.consecutive_failures,
            'total_reconnects': self.total_reconnects,
            'fps': self.fps,
            'is_connected': is_connected
        }
    
    def stop(self):
        """Stop camera capture"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self.disconnect()
        logging.info(f"Stopped camera handler for {self.config.camera_id}")


class DisplayManager:
    """Manages display windows for multiple cameras"""
    
    def __init__(self, max_display_fps: float = 10.0, enabled: bool = True):
        self.max_display_fps = max_display_fps
        self.last_display_time = {}
        self.min_display_interval = 1.0 / max_display_fps
        self.enabled = enabled
        self.disabled_due_to_error = False
    
    def should_display(self, camera_id: str) -> bool:
        """Check if we should display this frame (rate limiting)"""
        current_time = time.time()
        last_time = self.last_display_time.get(camera_id, 0)
        
        if current_time - last_time >= self.min_display_interval:
            self.last_display_time[camera_id] = current_time
            return True
        return False
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw detections on frame with distance information"""
        display_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            class_name = det['class']
            conf = det['confidence']
            distance_m = det.get('distance_m')
            
            # Choose color based on mask status
            if class_name == 'with_mask':
                color = (0, 255, 0)  # Green
            elif class_name == 'without_mask':
                color = (0, 0, 255)  # Red
            else:  # mask_weared_incorrect
                color = (0, 165, 255)  # Orange
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = f"{class_name}: {conf:.2f}"
            if distance_m:
                label += f" | {distance_m:.2f}m"
            
            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(display_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return display_frame
    
    def display_result(self, result: DetectionResult):
        """Display detection result"""
        if not self.enabled or self.disabled_due_to_error:
            return
        if not self.should_display(result.camera_id):
            return
        
        # Draw detections
        display_frame = self.draw_detections(result.frame, result.detections)
        
        # Add FPS and timestamp
        fps_text = f"FPS: {result.fps:.1f}"
        cv2.putText(display_frame, fps_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        camera_text = f"Camera: {result.camera_id}"
        cv2.putText(display_frame, camera_text, (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        detection_count = len(result.detections)
        count_text = f"Detections: {detection_count}"
        cv2.putText(display_frame, count_text, (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frame
        try:
            window_name = f"Mask Detection - {result.camera_id}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, display_frame)
        except Exception as e:
            logging.error(f"Display error (disabling display): {e}")
            self.disabled_due_to_error = True


class WebStreamManager:
    """Manages web streaming via Flask"""
    
    HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Mask Detection - Live Stream</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #1a1a1a;
            color: white;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: #4CAF50;
        }
        .cameras {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(640px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .camera-box {
            background: #2a2a2a;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .camera-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #4CAF50;
        }
        .camera-stats {
            font-size: 14px;
            color: #aaa;
            margin-bottom: 10px;
        }
        img {
            width: 100%;
            height: auto;
            border-radius: 4px;
            border: 2px solid #4CAF50;
        }
        .status {
            text-align: center;
            padding: 10px;
            background: #2a2a2a;
            border-radius: 4px;
            margin-bottom: 20px;
        }
        .status.online { color: #4CAF50; }
        .status.offline { color: #f44336; }
        .events {
            margin-top: 30px;
        }
        .events h2 { color: #4CAF50; }
        .event-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
        }
        .event-card {
            background: #2a2a2a;
            border-radius: 8px;
            padding: 10px;
        }
        .event-meta { color: #aaa; font-size: 12px; margin: 6px 0; }
        .event-img { width: 100%; border-radius: 4px; border: 1px solid #444; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎭 Mask Detection System - Live Feed</h1>
        <div class="status online">● System Online</div>
        <div class="cameras" id="cameras">
            {% for camera_id in camera_ids %}
            <div class="camera-box">
                <div class="camera-title">📹 {{ camera_id }}</div>
                <div class="camera-stats" id="stats-{{ camera_id }}">Loading...</div>
                <img src="/video_feed/{{ camera_id }}" alt="{{ camera_id }}" />
            </div>
            {% endfor %}
        </div>

        <div class="events">
            <h2>🚨 Recent Mask-Off Events</h2>
            <div id="events" class="event-grid"></div>
        </div>
    </div>
    <script>
        // Update stats periodically
        setInterval(async () => {
            {% for camera_id in camera_ids %}
            try {
                const response = await fetch('/stats/{{ camera_id }}');
                const data = await response.json();
                document.getElementById('stats-{{ camera_id }}').innerHTML = 
                    `FPS: ${data.fps.toFixed(1)} | Detections: ${data.detections} | Status: ${data.status}`;
            } catch (e) {
                document.getElementById('stats-{{ camera_id }}').innerHTML = 'Status: Offline';
            }
            {% endfor %}
        }, 1000);

        // Load recent events
        async function loadEvents() {
            try {
                const res = await fetch('/events?limit=12');
                const events = await res.json();
                const container = document.getElementById('events');
                container.innerHTML = events.map(ev => `
                    <div class="event-card">
                        <img class="event-img" src="${ev.image_url}" />
                        <div class="event-meta">${ev.ts} • ${ev.camera_id}</div>
                        <div class="event-meta">Class: ${ev.class} | Conf: ${(ev.confidence||0).toFixed(2)} | Dist: ${ev.distance_m ? ev.distance_m.toFixed(2)+'m' : 'n/a'}</div>
                        <div class="event-meta">Identity: ${ev.recognized_name || 'Unknown'}</div>
                    </div>
                `).join('');
            } catch (e) {}
        }
        setInterval(loadEvents, 2000);
        loadEvents();
    </script>
</body>
</html>
    """
    
    def __init__(self, camera_ids: List[str], port: int = 5000, event_logger: Optional[EventLogger] = None, face_recognizer: Optional['FaceRecognizer'] = None, detector_instance: Optional['RTSPMaskDetector'] = None):
        if not FLASK_AVAILABLE:
            raise ImportError("Flask is required for web streaming. Install with: pip install flask")
        
        self.camera_ids = camera_ids
        self.port = port
        self.app = Flask(__name__)
        self.latest_frames = {cam_id: None for cam_id in camera_ids}
        self.frame_locks = {cam_id: threading.Lock() for cam_id in camera_ids}
        self.stats = {cam_id: {'fps': 0.0, 'detections': 0, 'status': 'offline'} for cam_id in camera_ids}
        self.event_db_path = "events.db"
        self.event_image_dir = "events"
        self.event_logger = event_logger
        self.face_recognizer = face_recognizer
        self.known_faces_dir = "known_faces"
        self.detector_instance = detector_instance  # Reference to RTSPMaskDetector for camera management
        
        # Load HTML template from file
        template_path = os.path.join(os.path.dirname(__file__), 'web_template.html')
        if os.path.exists(template_path):
            with open(template_path, 'r') as f:
                self.HTML_TEMPLATE = f.read()
        else:
            # Fallback to simple template
            self.HTML_TEMPLATE = "<html><body>Template not found</body></html>"
        
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return render_template_string(self.HTML_TEMPLATE, camera_ids=self.camera_ids)
        
        @self.app.route('/video_feed/<camera_id>')
        def video_feed(camera_id):
            def generate():
                while True:
                    with self.frame_locks.get(camera_id, threading.Lock()):
                        frame = self.latest_frames.get(camera_id)
                    
                    if frame is not None:
                        # Encode frame as JPEG
                        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if ret:
                            frame_bytes = buffer.tobytes()
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    time.sleep(0.033)  # ~30 FPS for web stream
            
            return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @self.app.route('/stats/<camera_id>')
        def get_stats(camera_id):
            stats = self.stats.get(camera_id, {'fps': 0.0, 'detections': 0, 'status': 'offline'})
            return jsonify(stats)
        
        @self.app.route('/events')
        def list_events():
            limit = int(request.args.get('limit', 100)) if 'limit' in request.args else 100
            class_filter = request.args.get('class', '')
            conn = sqlite3.connect(self.event_db_path)
            try:
                query = "SELECT id, ts, camera_id, class, confidence, distance_m, image_path, recognized_name, is_false_claim FROM events WHERE 1=1"
                params = []
                
                if class_filter:
                    query += " AND class = ?"
                    params.append(class_filter)
                
                query += " ORDER BY id DESC LIMIT ?"
                params.append(limit)
                
                rows = conn.execute(query, params).fetchall()
                result = []
                for row in rows:
                    result.append({
                        'id': row[0],
                        'ts': row[1],
                        'camera_id': row[2],
                        'class': row[3],
                        'confidence': row[4],
                        'distance_m': row[5],
                        'image_url': f"/event_image/{row[0]}",
                        'recognized_name': row[7],
                        'is_false_claim': bool(row[8]) if len(row) > 8 else False
                    })
                return jsonify(result)
            finally:
                conn.close()
        
        @self.app.route('/events/<int:event_id>/false-claim', methods=['POST'])
        def mark_false_claim(event_id: int):
            if self.event_logger:
                self.event_logger.mark_false_claim(event_id)
                return jsonify({'success': True})
            return jsonify({'success': False, 'error': 'Event logger not available'}), 500
        
        # Face Management Endpoints
        @self.app.route('/faces', methods=['GET'])
        def list_faces():
            """List all registered faces"""
            if not os.path.exists(self.known_faces_dir):
                return jsonify([])
            
            faces = []
            for person_name in sorted(os.listdir(self.known_faces_dir)):
                person_dir = os.path.join(self.known_faces_dir, person_name)
                if not os.path.isdir(person_dir):
                    continue
                
                images = []
                for filename in sorted(os.listdir(person_dir)):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        image_path = os.path.join(person_dir, filename)
                        # Serve image via URL
                        images.append(f"/face_image/{person_name}/{filename}")
                
                if images:
                    faces.append({
                        'name': person_name,
                        'images': images,
                        'image_count': len(images)
                    })
            
            return jsonify(faces)
        
        @self.app.route('/faces', methods=['POST'])
        def add_face():
            """Add new person with face images or add images to existing person"""
            if not self.face_recognizer:
                return jsonify({'success': False, 'error': 'Face recognizer not available'}), 500
            
            if 'name' not in request.form:
                return jsonify({'success': False, 'error': 'Name required'}), 400
            
            person_name = request.form['name'].strip()
            if not person_name:
                return jsonify({'success': False, 'error': 'Name cannot be empty'}), 400
            
            # Create person directory (will not error if exists)
            person_dir = os.path.join(self.known_faces_dir, person_name)
            os.makedirs(person_dir, exist_ok=True)
            
            # Save uploaded images
            if 'images' not in request.files:
                return jsonify({'success': False, 'error': 'No images provided'}), 400
            
            files = request.files.getlist('images')
            saved_count = 0
            
            for file in files:
                if file.filename == '':
                    continue
                if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                
                # Save file with unique timestamp prefix
                safe_filename = secure_filename(file.filename)
                filename = f"{int(time.time() * 1000000)}_{safe_filename}"
                filepath = os.path.join(person_dir, filename)
                file.save(filepath)
                saved_count += 1
            
            if saved_count == 0:
                return jsonify({'success': False, 'error': 'No valid images saved'}), 400
            
            # Reload face recognizer
            if self.face_recognizer:
                self.face_recognizer.reload_known_faces()
                logging.info(f"Reloaded face recognizer after adding {saved_count} image(s) for {person_name}")
            
            return jsonify({'success': True, 'saved_count': saved_count})
        
        @self.app.route('/faces/<path:person_name>', methods=['DELETE'])
        def delete_face(person_name: str):
            """Delete a person and all their images"""
            person_dir = os.path.join(self.known_faces_dir, person_name)
            if not os.path.exists(person_dir):
                return jsonify({'success': False, 'error': 'Person not found'}), 404
            
            # Delete directory
            import shutil
            try:
                shutil.rmtree(person_dir)
                
                # Reload face recognizer
                if self.face_recognizer:
                    self.face_recognizer.reload_known_faces()
                
                return jsonify({'success': True})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/face_image/<path:person_name>/<path:filename>')
        def serve_face_image(person_name: str, filename: str):
            """Serve face images"""
            image_path = os.path.join(self.known_faces_dir, person_name, filename)
            if not os.path.exists(image_path) or not os.path.isfile(image_path):
                return Response(status=404)
            
            try:
                frame = cv2.imread(image_path)
                if frame is None:
                    return Response(status=404)
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if not ret:
                    return Response(status=500)
                return Response(buffer.tobytes(), mimetype='image/jpeg')
            except Exception:
                return Response(status=500)
        
        # Analytics Endpoint
        @self.app.route('/analytics')
        def get_analytics():
            """Get comprehensive analytics including violations, trends, and time-based metrics"""
            conn = sqlite3.connect(self.event_db_path)
            try:
                # Get all violation events (without_mask and incorrect) excluding false claims
                rows = conn.execute(
                    """SELECT id, ts, recognized_name, class, camera_id, confidence, distance_m 
                       FROM events 
                       WHERE is_false_claim = 0 
                       AND recognized_name IS NOT NULL 
                       AND class IN ('without_mask', 'mask_weared _incorrect')
                       ORDER BY ts DESC"""
                ).fetchall()
                
                # Person-level analytics
                person_stats = {}
                hourly_distribution = {i: 0 for i in range(24)}  # 0-23 hours
                daily_distribution = {}  # Date -> count
                camera_distribution = {}
                distance_stats = {'min': float('inf'), 'max': 0, 'avg': 0, 'count': 0}
                recent_violations = []
                
                total_violations = 0
                distance_sum = 0
                
                for row in rows:
                    event_id, ts, name, cls, camera_id, conf, dist = row
                    
                    # Parse timestamp
                    try:
                        dt = datetime.fromisoformat(ts)
                        hour = dt.hour
                        date_key = dt.date().isoformat()
                    except:
                        hour = 0
                        date_key = ts.split('T')[0] if 'T' in ts else ts.split(' ')[0]
                    
                    # Person stats
                    if name not in person_stats:
                        person_stats[name] = {
                            'total_violations': 0,
                            'first_seen': ts,
                            'last_seen': ts,
                            'camera_ids': set(),
                            'avg_confidence': 0,
                            'conf_sum': 0,
                            'conf_count': 0,
                            'distances': []
                        }
                    
                    person = person_stats[name]
                    person['total_violations'] += 1
                    person['last_seen'] = max(person['last_seen'], ts)
                    person['first_seen'] = min(person['first_seen'], ts)
                    person['camera_ids'].add(camera_id)
                    if conf:
                        person['conf_sum'] += conf
                        person['conf_count'] += 1
                    if dist:
                        person['distances'].append(dist)
                    
                    # Hourly distribution
                    hourly_distribution[hour] = hourly_distribution.get(hour, 0) + 1
                    
                    # Daily distribution
                    daily_distribution[date_key] = daily_distribution.get(date_key, 0) + 1
                    
                    # Camera distribution
                    camera_distribution[camera_id] = camera_distribution.get(camera_id, 0) + 1
                    
                    # Distance stats
                    if dist:
                        distance_stats['min'] = min(distance_stats['min'], dist)
                        distance_stats['max'] = max(distance_stats['max'], dist)
                        distance_sum += dist
                        distance_stats['count'] += 1
                    
                    total_violations += 1
                    
                    # Recent violations (last 10)
                    if len(recent_violations) < 10:
                        recent_violations.append({
                            'id': event_id,
                            'name': name,
                            'ts': ts,
                            'class': cls,
                            'camera_id': camera_id
                        })
                
                # Calculate averages and finalize person stats
                person_results = []
                for name, stats in person_stats.items():
                    # Calculate days since first and last violation
                    try:
                        first_dt = datetime.fromisoformat(stats['first_seen'])
                        last_dt = datetime.fromisoformat(stats['last_seen'])
                        days_active = (last_dt - first_dt).days + 1
                        violations_per_day = stats['total_violations'] / max(days_active, 1)
                    except:
                        days_active = 1
                        violations_per_day = stats['total_violations']
                    
                    avg_conf = stats['conf_sum'] / max(stats['conf_count'], 1) if stats['conf_count'] > 0 else 0
                    avg_dist = sum(stats['distances']) / len(stats['distances']) if stats['distances'] else None
                    
                    person_results.append({
                        'name': name,
                        'total_violations': stats['total_violations'],
                        'first_seen': stats['first_seen'],
                        'last_seen': stats['last_seen'],
                        'days_active': days_active,
                        'violations_per_day': round(violations_per_day, 2),
                        'cameras_seen_in': list(stats['camera_ids']),
                        'camera_count': len(stats['camera_ids']),
                        'avg_confidence': round(avg_conf, 2),
                        'avg_distance_m': round(avg_dist, 2) if avg_dist else None,
                        'severity': 'high' if stats['total_violations'] > 10 else ('medium' if stats['total_violations'] > 5 else 'low')
                    })
                
                # Sort by total violations (descending)
                person_results.sort(key=lambda x: x['total_violations'], reverse=True)
                
                # Calculate distance average
                if distance_stats['count'] > 0:
                    distance_stats['avg'] = round(distance_sum / distance_stats['count'], 2)
                    distance_stats['min'] = round(distance_stats['min'], 2) if distance_stats['min'] != float('inf') else 0
                    distance_stats['max'] = round(distance_stats['max'], 2)
                
                # Find peak violation hour
                peak_hour = max(hourly_distribution.items(), key=lambda x: x[1]) if hourly_distribution else (0, 0)
                
                # Find peak violation day
                peak_day = max(daily_distribution.items(), key=lambda x: x[1]) if daily_distribution else (None, 0)
                
                # Find most problematic camera
                worst_camera = max(camera_distribution.items(), key=lambda x: x[1]) if camera_distribution else (None, 0)
                
                # Time-based trends (last 7 days)
                sorted_days = sorted(daily_distribution.items(), reverse=True)[:7]
                trend_data = [{'date': day, 'count': count} for day, count in sorted_days]
                
                return jsonify({
                    'summary': {
                        'total_violations': total_violations,
                        'unique_offenders': len(person_stats),
                        'peak_hour': peak_hour[0],
                        'peak_hour_count': peak_hour[1],
                        'peak_day': peak_day[0],
                        'peak_day_count': peak_day[1],
                        'worst_camera': worst_camera[0],
                        'worst_camera_count': worst_camera[1],
                        'avg_distance_m': distance_stats['avg'],
                        'min_distance_m': distance_stats['min'],
                        'max_distance_m': distance_stats['max']
                    },
                    'persons': person_results,
                    'hourly_distribution': hourly_distribution,
                    'daily_distribution': {k: v for k, v in sorted(daily_distribution.items(), reverse=True)[:14]},  # Last 14 days
                    'camera_distribution': camera_distribution,
                    'trend_data': trend_data,
                    'recent_violations': recent_violations
                })
            finally:
                conn.close()
        
        @self.app.route('/event_image/<int:event_id>')
        def event_image(event_id: int):
            conn = sqlite3.connect(self.event_db_path)
            try:
                row = conn.execute("SELECT image_path FROM events WHERE id = ?", (event_id,)).fetchone()
                if not row:
                    return Response(status=404)
                image_path = row[0]
            finally:
                conn.close()
            
            try:
                frame = cv2.imread(image_path)
                if frame is None:
                    return Response(status=404)
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if not ret:
                    return Response(status=500)
                return Response(buffer.tobytes(), mimetype='image/jpeg')
            except Exception:
                return Response(status=500)
        
        # Camera Management Endpoints
        @self.app.route('/api/cameras', methods=['GET'])
        def list_cameras():
            """List all configured cameras"""
            cameras = []
            if self.detector_instance:
                for config in self.detector_instance.camera_configs:
                    handler = next((h for h in self.detector_instance.camera_handlers if h.config.camera_id == config.camera_id), None)
                    status = 'online' if handler and handler.running and handler.connection_healthy else 'offline'
                    cameras.append({
                        'camera_id': config.camera_id,
                        'rtsp_url': config.rtsp_url,
                        'focal_length_mm': config.focal_length_mm,
                        'sensor_width_mm': config.sensor_width_mm,
                        'frame_width': config.frame_width,
                        'frame_height': config.frame_height,
                        'known_face_width_cm': config.known_face_width_cm,
                        'status': status
                    })
            return jsonify({'cameras': cameras, 'count': len(cameras), 'max_cameras': 4})
        
        @self.app.route('/api/cameras', methods=['POST'])
        def add_camera():
            """Add a new camera (max 4 cameras)"""
            if not self.detector_instance:
                return jsonify({'success': False, 'error': 'Detector instance not available'}), 500
            
            data = request.json
            rtsp_url = data.get('rtsp_url', '').strip()
            camera_id = data.get('camera_id', '').strip()
            
            if not rtsp_url or not camera_id:
                return jsonify({'success': False, 'error': 'RTSP URL and Camera ID are required'}), 400
            
            # Check current camera count
            if len(self.detector_instance.camera_handlers) >= 4:
                return jsonify({'success': False, 'error': 'Maximum of 4 cameras allowed'}), 400
            
            # Create camera config with performance optimizations
            config = CameraConfig(
                rtsp_url=rtsp_url,
                camera_id=camera_id,
                focal_length_mm=float(data.get('focal_length_mm', 50.0)),
                sensor_width_mm=float(data.get('sensor_width_mm', 36.0)),
                frame_width=int(data.get('frame_width', 1920)),
                frame_height=int(data.get('frame_height', 1080)),
                known_face_width_cm=float(data.get('known_face_width_cm', 15.0)),
                frame_timeout=float(data.get('frame_timeout', 10.0)),
                max_consecutive_failures=int(data.get('max_consecutive_failures', 5)),
                # Performance optimizations (default values for multi-camera - optimized for 4 cameras)
                process_every_n_frames=int(data.get('process_every_n_frames', 3)),  # Process every 3rd frame
                inference_size=int(data.get('inference_size', 512)),  # Resize to 512px for inference (faster)
                conf_threshold=float(data.get('conf_threshold', 0.25))  # Confidence threshold
            )
            
            success = self.detector_instance.add_camera(config)
            if success:
                return jsonify({'success': True, 'message': f'Camera {camera_id} added successfully'})
            else:
                return jsonify({'success': False, 'error': f'Failed to add camera {camera_id}. It may already exist.'}), 400
        
        @self.app.route('/api/cameras/<path:camera_id>', methods=['DELETE'])
        def remove_camera(camera_id: str):
            """Remove a camera"""
            if not self.detector_instance:
                return jsonify({'success': False, 'error': 'Detector instance not available'}), 500
            
            success = self.detector_instance.remove_camera(camera_id)
            if success:
                return jsonify({'success': True, 'message': f'Camera {camera_id} removed successfully'})
            else:
                return jsonify({'success': False, 'error': f'Camera {camera_id} not found'}), 404
    
    def add_camera(self, camera_id: str):
        """Add a camera to web streaming"""
        if camera_id not in self.camera_ids:
            self.camera_ids.append(camera_id)
            self.latest_frames[camera_id] = None
            self.frame_locks[camera_id] = threading.Lock()
            self.stats[camera_id] = {'fps': 0.0, 'detections': 0, 'status': 'offline'}
            logging.info(f"Camera {camera_id} added to web streaming")
    
    def remove_camera(self, camera_id: str):
        """Remove a camera from web streaming"""
        if camera_id in self.camera_ids:
            self.camera_ids.remove(camera_id)
            if camera_id in self.latest_frames:
                del self.latest_frames[camera_id]
            if camera_id in self.frame_locks:
                del self.frame_locks[camera_id]
            if camera_id in self.stats:
                del self.stats[camera_id]
            logging.info(f"Camera {camera_id} removed from web streaming")
    
    def update_frame(self, result: DetectionResult, display_manager: DisplayManager):
        """Update frame for a camera"""
        if result.camera_id in self.latest_frames:
            display_frame = display_manager.draw_detections(result.frame, result.detections)
            
            # Add text overlays
            fps_text = f"FPS: {result.fps:.1f}"
            cv2.putText(display_frame, fps_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            camera_text = f"Camera: {result.camera_id}"
            cv2.putText(display_frame, camera_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            detection_count = len(result.detections)
            count_text = f"Detections: {detection_count}"
            cv2.putText(display_frame, count_text, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            with self.frame_locks[result.camera_id]:
                self.latest_frames[result.camera_id] = display_frame
            
            # Update stats
            self.stats[result.camera_id] = {
                'fps': result.fps,
                'detections': detection_count,
                'status': 'online'
            }
    
    def start(self):
        """Start Flask server in a separate thread"""
        def run_flask():
            self.app.run(host='0.0.0.0', port=self.port, threaded=True, debug=False)
        
        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()
        logging.info(f"Web streaming server started on http://localhost:{self.port}")


class RTSPMaskDetector:
    """Main application class for RTSP mask detection"""
    
    def __init__(self, camera_configs: List[CameraConfig], model_path: str, headless: bool = False, web_mode: bool = False, web_port: int = 5000):
        self.camera_configs = camera_configs
        self.model_path = model_path
        self.model: Optional[YOLO] = None
        self.camera_handlers: List[CameraHandler] = []
        self.result_queue = Queue(maxsize=50 * len(camera_configs))  # Scale queue size with camera count
        self.display_manager = DisplayManager(max_display_fps=15.0, enabled=not headless and not web_mode)  # Increased FPS for smoother display
        self.running = False
        self.display_thread: Optional[threading.Thread] = None
        self.web_mode = web_mode
        self.web_stream_manager: Optional[WebStreamManager] = None
        self.event_logger = EventLogger(
            min_capture_interval=30.0,  # Default, can be overridden
            enable_spatial_dedup=True  # Default, can be overridden
        )
        self.face_recognizer = FaceRecognizer()
        self.camera_manager = CameraManager()  # SQLite-based camera storage
        
        if web_mode:
            if not FLASK_AVAILABLE:
                raise ImportError("Flask is required for web mode. Install with: pip install flask")
            camera_ids = [config.camera_id for config in camera_configs]
            self.web_stream_manager = WebStreamManager(camera_ids, port=web_port, 
                                                       event_logger=self.event_logger,
                                                       face_recognizer=self.face_recognizer,
                                                       detector_instance=self)
    
    def initialize(self):
        """Initialize model and camera handlers"""
        try:
            logging.info(f"Loading YOLO model from {self.model_path}")
            self.model = YOLO(self.model_path)
            logging.info("Model loaded successfully")
            
            # Create camera handlers
            for config in self.camera_configs:
                handler = CameraHandler(config, self.model, self.result_queue)
                self.camera_handlers.append(handler)
            
        except Exception as e:
            logging.error(f"Initialization error: {e}")
            raise
    
    def _display_loop(self):
        """Display loop running in separate thread"""
        while self.running:
            try:
                try:
                    result = self.result_queue.get(timeout=0.1)
                    
                    # Event capture: log snapshots for 'without_mask' (with throttling)
                    for det in result.detections:
                        if det.get('class') == 'without_mask':
                            # Check if we should capture (throttling + spatial dedup)
                            bbox = det.get('bbox', [])
                            if not self.event_logger.should_capture_event(result.camera_id, bbox):
                                # Throttled - skip this detection
                                continue
                            
                            x1, y1, x2, y2 = map(int, det['bbox'])
                            # Safe crop
                            x1 = max(0, x1); y1 = max(0, y1)
                            x2 = min(result.frame.shape[1] - 1, x2)
                            y2 = min(result.frame.shape[0] - 1, y2)
                            if x2 > x1 and y2 > y1:
                                face_crop = result.frame[y1:y2, x1:x2]
                                
                                # Perform face recognition
                                recognized_name = None
                                if self.face_recognizer.known_face_encodings:
                                    recognized_name = self.face_recognizer.recognize_face(face_crop)
                                    if recognized_name:
                                        logging.info(f"🚨 Recognized offender: {recognized_name} (camera: {result.camera_id})")
                                
                                snap_path = self.event_logger.save_snapshot(face_crop, result.camera_id)
                                self.event_logger.log_event(
                                    camera_id=result.camera_id,
                                    cls='without_mask',
                                    confidence=det.get('confidence', 0.0),
                                    distance_m=det.get('distance_m'),
                                    image_path=snap_path,
                                    recognized_name=recognized_name
                                )
                            # Only process first valid detection per frame to limit I/O
                            break

                    # Update web stream if enabled
                    if self.web_mode and self.web_stream_manager:
                        self.web_stream_manager.update_frame(result, self.display_manager)
                    
                    # Update GUI display if enabled
                    if self.display_manager.enabled and not self.display_manager.disabled_due_to_error:
                        self.display_manager.display_result(result)
                        
                except Empty:
                    continue
                
                # Handle keyboard input
                if self.display_manager.enabled and not self.display_manager.disabled_due_to_error:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.running = False
                        break
            except Exception as e:
                logging.error(f"Error in display loop: {e}")
                time.sleep(0.1)
    
    def run(self):
        """Run the detection system"""
        self.running = True
        
        # Start web server if enabled
        if self.web_mode and self.web_stream_manager:
            self.web_stream_manager.start()
            camera_ids = [config.camera_id for config in self.camera_configs]
            logging.info(f"Web interface available at http://localhost:{self.web_stream_manager.port}")
        
        # Start all camera handlers with staggered timing
        for idx, handler in enumerate(self.camera_handlers):
            handler.start()
            time.sleep(0.2 * (idx + 1))  # Stagger connections to avoid network congestion
        
        # Start display thread
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()
        
        if self.web_mode:
            logging.info("Mask detection system started. View at http://localhost:{}".format(self.web_stream_manager.port if self.web_stream_manager else 5000))
        else:
            logging.info("Mask detection system started. Press 'q' to quit.")
        
        try:
            # Main loop - monitor camera handlers with health checks
            last_health_check = time.time()
            health_check_interval = 30.0  # Perform detailed health check every 30 seconds
            
            while self.running:
                current_time = time.time()
                
                # Detailed health check for all cameras
                if current_time - last_health_check >= health_check_interval:
                    for handler in self.camera_handlers:
                        # Check if handler thread is alive
                        if handler.thread and not handler.thread.is_alive():
                            logging.warning(f"Camera handler thread for {handler.config.camera_id} died, restarting...")
                            handler.stop()
                            time.sleep(1)
                            handler.start()
                        
                        # Check connection health
                        if handler.running and not handler.connection_healthy:
                            time_since_frame = time.time() - handler.last_frame_time
                            if time_since_frame > handler.frame_timeout:
                                logging.warning(f"Camera {handler.config.camera_id} appears unhealthy (no frames for {time_since_frame:.1f}s)")
                                # Trigger reconnection in capture loop
                    
                    last_health_check = current_time
                
                # Quick check for stopped handlers
                for handler in self.camera_handlers:
                    if not handler.running:
                        # Check if it should be running (not intentionally stopped)
                        if handler.thread and handler.thread.is_alive():
                            # Thread is alive but handler says not running - might be reconnecting
                            continue
                        else:
                            # Handler completely stopped - restart if needed
                            logging.info(f"Restarting stopped camera handler: {handler.config.camera_id}")
                            handler.start()
                            time.sleep(0.5)
                
                time.sleep(5)  # Check every 5 seconds
        except KeyboardInterrupt:
            logging.info("Received keyboard interrupt")
        finally:
            self.stop()
    
    def add_camera(self, config: CameraConfig) -> bool:
        """Add a new camera dynamically (max 4 cameras)"""
        if len(self.camera_handlers) >= 4:
            logging.warning(f"Cannot add camera: Maximum of 4 cameras allowed. Current: {len(self.camera_handlers)}")
            return False
        
        # Check if camera_id already exists
        if any(h.config.camera_id == config.camera_id for h in self.camera_handlers):
            logging.warning(f"Camera with ID {config.camera_id} already exists")
            return False
        
        try:
            # Save to SQLite database
            if not self.camera_manager.save_camera(config):
                logging.error(f"Failed to save camera {config.camera_id} to database")
                return False
            
            handler = CameraHandler(config, self.model, self.result_queue)
            self.camera_handlers.append(handler)
            self.camera_configs.append(config)
            handler.start()
            
            # Update web stream manager
            if self.web_mode and self.web_stream_manager:
                self.web_stream_manager.add_camera(config.camera_id)
            
            logging.info(f"Camera {config.camera_id} added successfully")
            return True
        except Exception as e:
            logging.error(f"Error adding camera {config.camera_id}: {e}")
            return False
    
    def remove_camera(self, camera_id: str) -> bool:
        """Remove a camera dynamically"""
        handler = None
        for h in self.camera_handlers:
            if h.config.camera_id == camera_id:
                handler = h
                break
        
        if not handler:
            logging.warning(f"Camera {camera_id} not found")
            return False
        
        try:
            handler.stop()
            self.camera_handlers.remove(handler)
            self.camera_configs = [c for c in self.camera_configs if c.camera_id != camera_id]
            
            # Remove from SQLite database
            if not self.camera_manager.delete_camera(camera_id):
                logging.warning(f"Camera {camera_id} removed from runtime but not found in database")
            
            # Update web stream manager
            if self.web_mode and self.web_stream_manager:
                self.web_stream_manager.remove_camera(camera_id)
            
            logging.info(f"Camera {camera_id} removed successfully")
            return True
        except Exception as e:
            logging.error(f"Error removing camera {camera_id}: {e}")
            return False
    
    def save_all_cameras(self):
        """Save all current camera configurations to SQLite database"""
        for config in self.camera_configs:
            self.camera_manager.save_camera(config)
        logging.info(f"Saved {len(self.camera_configs)} camera(s) to database")
    
    def stop(self):
        """Stop the detection system"""
        logging.info("Stopping mask detection system...")
        self.running = False
        
        # Stop all camera handlers
        for handler in self.camera_handlers:
            handler.stop()
        
        # Wait for display thread
        if self.display_thread:
            self.display_thread.join(timeout=2.0)
        
        cv2.destroyAllWindows()
        logging.info("System stopped")


def load_config(config_path: str) -> List[CameraConfig]:
    """Load camera configurations from JSON file with performance defaults"""
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        configs = []
        for cam_data in config_data.get('cameras', []):
            # Set performance defaults if not present
            if 'process_every_n_frames' not in cam_data:
                cam_data['process_every_n_frames'] = 3
            if 'inference_size' not in cam_data:
                cam_data['inference_size'] = 512
            if 'conf_threshold' not in cam_data:
                cam_data['conf_threshold'] = 0.25
            
            config = CameraConfig(**cam_data)
            configs.append(config)
        
        logging.info(f"Loaded {len(configs)} camera(s) from {config_path}")
        return configs
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        raise


def create_default_config(rtsp_url: str, output_path: str = "config.json"):
    """Create default configuration file with performance optimizations"""
    config = {
        "cameras": [
            {
                "rtsp_url": rtsp_url,
                "camera_id": "Camera_1",
                "focal_length_mm": 50.0,
                "sensor_width_mm": 36.0,
                "frame_width": 1920,
                "frame_height": 1080,
                "known_face_width_cm": 15.0,
                "process_every_n_frames": 3,  # Performance defaults
                "inference_size": 512,
                "conf_threshold": 0.25
            }
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logging.info(f"Created default config file: {output_path}")


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'mask_detector_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )


def main():
    parser = argparse.ArgumentParser(description='RTSP Mask Detection System')
    parser.add_argument('--model', type=str, 
                       default='runs/detect/train/weights/best.pt',
                       help='Path to YOLO model weights')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to camera configuration JSON file')
    parser.add_argument('--rtsp', type=str, default=None,
                       help='Single RTSP URL (for quick start)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--headless', action='store_true',
                       help='Disable GUI windows (useful on headless/CI/Mac without display access)')
    parser.add_argument('--web', action='store_true',
                       help='Enable web streaming interface (view in browser)')
    parser.add_argument('--web-port', type=int, default=5000,
                       help='Port for web interface (default: 5000)')
    parser.add_argument('--event-interval', type=float, default=30.0,
                       help='Minimum seconds between event captures per camera (default: 30.0)')
    parser.add_argument('--disable-spatial-dedup', action='store_true',
                       help='Disable spatial deduplication (capture events even if same location)')
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    
    # Initialize camera manager (SQLite-based)
    camera_manager = CameraManager()
    
    # Load or create configuration
    # Priority: SQLite DB (cameras.db) > --config JSON > --rtsp > default config.json
    if args.config:
        if not os.path.exists(args.config):
            logging.error(f"Config file not found: {args.config}")
            return
        camera_configs = load_config(args.config)
        logging.info(f"Loaded cameras from {args.config}")
        # Also save to database for persistence
        for config in camera_configs:
            camera_manager.save_camera(config)
    elif args.rtsp:
        # Quick start with single RTSP URL
        config = CameraConfig(
            rtsp_url=args.rtsp,
            camera_id="Camera_1",
            process_every_n_frames=3,  # Performance defaults
            inference_size=512,
            conf_threshold=0.25
        )
        camera_configs = [config]
        camera_manager.save_camera(config)  # Save to database
        logging.info("Using RTSP URL from command line (saved to database)")
    else:
        # Try SQLite database first (UI-saved cameras)
        camera_configs = camera_manager.load_all_cameras()
        if camera_configs:
            logging.info(f"Loaded {len(camera_configs)} camera(s) from SQLite database (cameras.db)")
        elif os.path.exists("config.json"):
            # Fallback to JSON config
            camera_configs = load_config("config.json")
            logging.info("Loaded cameras from config.json")
            # Migrate to database
            for config in camera_configs:
                camera_manager.save_camera(config)
            logging.info("Migrated cameras from config.json to database")
        elif os.path.exists("cameras.json"):
            # Fallback to old cameras.json
            camera_configs = load_config("cameras.json")
            logging.info("Loaded cameras from cameras.json (migrating to database)")
            # Migrate to database
            for config in camera_configs:
                camera_manager.save_camera(config)
            logging.info("Migrated cameras from cameras.json to database")
        else:
            # Create default config
            default_rtsp = "rtsp://admin:Krishna%40429@192.168.1.25:554/Streaming/Channels/101"
            create_default_config(default_rtsp)
            camera_configs = load_config("config.json")
            # Save to database
            for config in camera_configs:
                camera_manager.save_camera(config)
            logging.info("Created default configuration and saved to database")
    
    # Validate model path
    if not os.path.exists(args.model):
        logging.error(f"Model file not found: {args.model}")
        return
    
    # Initialize and run detector
    try:
        detector = RTSPMaskDetector(
            camera_configs, 
            args.model, 
            headless=args.headless,
            web_mode=args.web,
            web_port=args.web_port
        )
        # Configure event logger with CLI options
        detector.event_logger.min_capture_interval = args.event_interval
        detector.event_logger.enable_spatial_dedup = not args.disable_spatial_dedup
        if args.event_interval > 0:
            logging.info(f"Event capture interval set to {args.event_interval}s per camera")
        if detector.event_logger.enable_spatial_dedup:
            logging.info("Spatial deduplication enabled (will skip same person in same location)")
        
        detector.initialize()
        detector.run()
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)


if __name__ == "__main__":
    main()

