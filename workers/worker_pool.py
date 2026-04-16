"""
Worker Pool: Multi-detector thread pool for parallel inference.

Manages:
- Multiple detector instances (Worker 1: YOLOv8+ViT, Worker 2: Faster R-CNN+RT-DETR)
- Frame consumption from Redis streams
- Parallel inference on GPU
- Detection publishing to Redis detection streams
"""

import logging
import threading
import queue
import time
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from redis import Redis
from producer.frame_serializer import FrameDeserializer
from redis_broker.stream_manager import RedisStreamManager

from config.settings import (
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB,
    DEVICE_WORKER_1,
    DEVICE_WORKER_2,
    DETECTION_BATCH_SIZE,
    DETECTIONS_STREAM_PREFIX,
    MAX_DETECTIONS_STREAM_LENGTH,
)
from workers.yolo_vit_detector import YOLOVitDetector
from workers.frcnn_rtdetr_detector import FasterRCNNRtdetrDetector


logger = logging.getLogger(__name__)


class WorkerPool:
    """
    Manages multiple detection workers running inference in parallel.
    
    Features:
    - Worker registry (self-discovery in Redis)
    - Thread-safe frame consumption from Redis streams
    - Batch processing for GPU efficiency
    - Detection result publishing to Redis
    - Graceful shutdown and cleanup
    """

    def __init__(
        self,
        num_workers: int = 2,
        batch_size: int = DETECTION_BATCH_SIZE,
        use_gpu: bool = True,
    ):
        """
        Initialize worker pool.
        
        Args:
            num_workers: Number of detector workers to create
            batch_size: Frames to batch for inference
            use_gpu: Enable GPU acceleration
        """
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.use_gpu = use_gpu

        # Redis connection
        self.redis_client = Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)
        self.stream_manager = RedisStreamManager()
        self.frame_deserializer = FrameDeserializer()

        # Worker instances
        self.workers: Dict[str, object] = {}
        self._initialize_workers()

        # Threading
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.running = False
        self.stop_event = threading.Event()

        logger.info(f"WorkerPool initialized with {num_workers} workers")

    def _initialize_workers(self) -> None:
        """Initialize detector instances."""
        try:
            logger.info("Initializing Worker 1: YOLOv8 + Vision Transformer...")
            self.workers["yolo_vit"] = YOLOVitDetector(
                device=DEVICE_WORKER_1 if self.use_gpu else "cpu",
                batch_size=self.batch_size,
                use_fp16=True,
            )

            logger.info("Initializing Worker 2: Faster R-CNN + RT-DETR Lite...")
            self.workers["frcnn_rtdetr"] = FasterRCNNRtdetrDetector(
                device=DEVICE_WORKER_2 if self.use_gpu else "cpu",
                batch_size=self.batch_size,
                use_fp16=True,
            )

            logger.info(f"Successfully initialized {len(self.workers)} workers")

        except Exception as e:
            logger.error(f"Failed to initialize workers: {e}")
            self.cleanup()
            raise

    def start(self) -> None:
        """Start worker pool for consuming frames and running inference."""
        self.running = True
        self.stop_event.clear()

        logger.info("Starting WorkerPool...")

        # Get list of camera streams
        camera_ids = self._get_active_cameras()

        if not camera_ids:
            logger.warning("No active cameras found in Redis")
            return

        # Submit detection tasks for all cameras
        futures = []
        for worker_name, detector in self.workers.items():
            for camera_id in camera_ids:
                future = self.executor.submit(
                    self._process_camera_stream,
                    camera_id,
                    worker_name,
                    detector,
                )
                futures.append(future)

        # Monitor futures
        for future in as_completed(futures, timeout=None):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Worker task failed: {e}")

        logger.info("WorkerPool stopped")

    def stop(self) -> None:
        """Stop worker pool gracefully."""
        logger.info("Stopping WorkerPool...")
        self.running = False
        self.stop_event.set()
        self.executor.shutdown(wait=True)
        self.cleanup()

    def _process_camera_stream(
        self,
        camera_id: str,
        worker_name: str,
        detector: object,
    ) -> None:
        """
        Process frames from a camera stream using a detector.
        
        Args:
            camera_id: Camera identifier (e.g., "camera:1")
            worker_name: Detector name (e.g., "yolo_vit")
            detector: Detector instance
        """
        logger.info(f"[{worker_name}] Starting stream processor for {camera_id}")

        frame_buffer = []
        last_process_time = time.time()

        while self.running and not self.stop_event.is_set():
            try:
                # Get latest frame from Redis stream
                frame_data = self.stream_manager.get_latest_frame(camera_id)

                if frame_data is None:
                    time.sleep(0.01)  # Avoid busy-wait
                    continue

                # Deserialize frame
                try:
                    frame_record = self.frame_deserializer.decode(frame_data)
                except Exception as e:
                    logger.warning(f"Failed to deserialize frame: {e}")
                    continue

                frame = frame_record["frame"]
                frame_id = frame_record.get("id", "")
                timestamp = frame_record.get("timestamp", time.time())

                # Buffer frames for batch processing
                frame_buffer.append((frame, frame_id, timestamp))

                # Process batch when buffer is full or timeout
                if len(frame_buffer) >= self.batch_size or (time.time() - last_process_time > 1.0):
                    self._process_batch(camera_id, worker_name, detector, frame_buffer)
                    frame_buffer = []
                    last_process_time = time.time()

            except Exception as e:
                logger.error(f"[{worker_name}] Error processing {camera_id}: {e}")
                time.sleep(0.1)

    def _process_batch(
        self,
        camera_id: str,
        worker_name: str,
        detector: object,
        frame_buffer: List[tuple],
    ) -> None:
        """
        Run inference on a batch of frames.
        
        Args:
            camera_id: Camera identifier
            worker_name: Detector name
            detector: Detector instance
            frame_buffer: List of (frame, frame_id, timestamp) tuples
        """
        try:
            frames = [f[0] for f in frame_buffer]

            # Run batch inference
            if len(frames) == 1:
                detections_list = [detector.detect(frames[0])]
            else:
                detections_list = detector.detect_batch(frames)

            # Publish detections to Redis
            for (frame, frame_id, timestamp), detections in zip(frame_buffer, detections_list):
                self._publish_detections(
                    camera_id, worker_name, frame_id, timestamp, detections
                )

        except Exception as e:
            logger.error(f"[{worker_name}] Batch processing failed for {camera_id}: {e}")

    def _publish_detections(
        self,
        camera_id: str,
        worker_name: str,
        frame_id: str,
        timestamp: float,
        detections: Dict,
    ) -> None:
        """
        Publish detection results to Redis stream.
        
        Args:
            camera_id: Camera identifier
            worker_name: Detector name
            frame_id: Frame identifier
            timestamp: Frame timestamp
            detections: Detection dictionary
        """
        try:
            # Prepare detection record
            detection_record = {
                "camera_id": camera_id,
                "worker_name": worker_name,
                "frame_id": frame_id,
                "timestamp": timestamp,
                "num_detections": detections.get("num_detections", 0),
                "execution_time_ms": detections.get("execution_time_ms", 0),
                "boxes": detections.get("boxes", []).tolist() if len(detections.get("boxes", [])) > 0 else [],
                "confidences": detections.get("confidences", []).tolist() if len(detections.get("confidences", [])) > 0 else [],
                "class_ids": detections.get("class_ids", []).tolist() if len(detections.get("class_ids", [])) > 0 else [],
            }

            # Create stream key with worker identifier
            stream_key = f"{DETECTIONS_STREAM_PREFIX}{worker_name}:{camera_id}"

            # Add to stream with LIFO trimming
            self.stream_manager.add_frame_to_stream(
                stream_key,
                detection_record,
                max_length=MAX_DETECTIONS_STREAM_LENGTH,
            )

            logger.debug(
                f"[{worker_name}] Published {detection_record['num_detections']} detections "
                f"for {camera_id} (exec time: {detection_record['execution_time_ms']:.1f}ms)"
            )

        except Exception as e:
            logger.error(f"Failed to publish detections to Redis: {e}")

    def _get_active_cameras(self) -> List[str]:
        """Get list of active cameras from Redis."""
        try:
            camera_ids = self.stream_manager.get_all_camera_ids()
            logger.info(f"Found {len(camera_ids)} active cameras")
            return camera_ids
        except Exception as e:
            logger.error(f"Failed to get active cameras: {e}")
            return []

    def get_worker_status(self) -> Dict:
        """Get status of all workers."""
        status = {}
        for worker_name, detector in self.workers.items():
            status[worker_name] = detector.get_model_info()
        return status

    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up WorkerPool...")
        for worker_name, detector in self.workers.items():
            try:
                detector.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up {worker_name}: {e}")
        logger.info("WorkerPool cleanup complete")
