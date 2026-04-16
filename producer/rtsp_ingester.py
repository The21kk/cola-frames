import cv2
import time
import logging
from threading import Thread
from redis_broker.stream_manager import RedisStreamManager
from producer.frame_serializer import FrameSerializer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RTSPIngester:
    """
    Captures frames from an RTSP stream and ingests them into Redis Streams.
    Runs in a daemon thread to avoid blocking the main application.
    """

    def __init__(self, camera_id: str, rtsp_url: str, fps_target: int = 5):
        """
        Initialize RTSP ingester.
        
        Args:
            camera_id: Unique camera identifier
            rtsp_url: RTSP stream URL (e.g., "rtsp://camera_ip:554/stream")
            fps_target: Target frames per second (default 5)
        """
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.fps_target = fps_target
        self.stream_manager = RedisStreamManager()
        self.is_running = False
        self.thread = None
        self.frame_count = 0

    def start(self):
        """
        Starts the ingestion in a daemon thread.
        """
        if self.is_running:
            logger.warning(f"Ingester for {self.camera_id} is already running")
            return
        
        self.is_running = True
        self.thread = Thread(target=self._ingest_loop, daemon=True)
        self.thread.start()
        logger.info(f"[START] Ingester for camera '{self.camera_id}' started (target FPS: {self.fps_target})")

    def stop(self):
        """
        Stops the ingestion thread.
        """
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info(f"[STOP] Ingester for camera '{self.camera_id}' stopped (processed {self.frame_count} frames)")

    def _ingest_loop(self):
        """
        Main loop: captures frames from RTSP and sends to Redis.
        """
        cap = cv2.VideoCapture(self.rtsp_url)
        frame_delay = 1.0 / self.fps_target
        
        if not cap.isOpened():
            logger.error(f"[ERROR] Failed to open RTSP stream: {self.rtsp_url}")
            self.is_running = False
            return
        
        # Get actual FPS from camera if available
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"[INFO] Camera {self.camera_id} actual FPS: {actual_fps}")
        
        connection_errors = 0
        MAX_CONSECUTIVE_ERRORS = 10
        
        while self.is_running:
            try:
                ret, frame = cap.read()
                
                if not ret:
                    connection_errors += 1
                    if connection_errors >= MAX_CONSECUTIVE_ERRORS:
                        logger.error(f"[ERROR] Too many consecutive read failures for {self.camera_id}")
                        break
                    time.sleep(1)
                    continue
                
                connection_errors = 0  # Reset on successful read
                
                # Serialize frame
                frame_b64 = FrameSerializer.encode_frame_to_base64(frame)
                metadata = FrameSerializer.create_metadata(
                    self.camera_id,
                    self.fps_target,
                    (frame.shape[1], frame.shape[0])
                )
                
                # Add to Redis Stream
                try:
                    self.stream_manager.add_frame_to_stream(
                        self.camera_id,
                        frame_b64,
                        metadata
                    )
                    self.frame_count += 1
                    
                    # Log every 100 frames
                    if self.frame_count % 100 == 0:
                        stream_len = self.stream_manager.get_stream_length(self.camera_id)
                        logger.info(f"[PROGRESS] Camera {self.camera_id}: {self.frame_count} frames sent, "
                                   f"stream length: {stream_len}")
                
                except Exception as e:
                    logger.error(f"[ERROR] Failed to add frame to Redis: {e}")
                
                time.sleep(frame_delay)
            
            except Exception as e:
                logger.error(f"[ERROR] Unexpected error in ingest loop: {e}")
                time.sleep(1)
        
        cap.release()
        logger.info(f"[INFO] RTSP stream closed for camera {self.camera_id}")
