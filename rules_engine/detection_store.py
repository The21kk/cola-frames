"""
Detection Storage: Store detection results in Redis streams.

Each detector worker publishes detections to its own stream:
- detections:yolo_vit:camera:1
- detections:frcnn_rtdetr:camera:1

Enables:
- Audit trail of all detections
- Individual model performance tracking
- Consensus voting across workers
"""

import logging
import json
from typing import Dict, List, Optional
import numpy as np
from redis import Redis

# Detection store uses Redis directly for custom streams
from config.settings import (
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB,
    DETECTIONS_STREAM_PREFIX,
    MAX_DETECTIONS_STREAM_LENGTH,
)


logger = logging.getLogger(__name__)


class DetectionStore:
    """Store and retrieve detection results from Redis streams."""

    def __init__(self):
        """Initialize detection storage."""
        self.redis_client = Redis(
            host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False
        )
        # Initialize Redis client for detection storage

    def store_detection(
        self,
        camera_id: str,
        worker_name: str,
        frame_id: str,
        timestamp: float,
        boxes: np.ndarray,
        confidences: np.ndarray,
        class_ids: np.ndarray,
        execution_time_ms: float,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Store detection results in Redis stream.
        
        Args:
            camera_id: Camera identifier
            worker_name: Detector worker name (e.g., "yolo_vit", "frcnn_rtdetr")
            frame_id: Frame identifier
            timestamp: Frame timestamp
            boxes: Detection bounding boxes (N, 4)
            confidences: Detection confidence scores (N,)
            class_ids: Detection class IDs (N,)
            execution_time_ms: Inference execution time
            metadata: Additional metadata dictionary
            
        Returns:
            Redis stream ID
        """
        try:
            # Convert numpy arrays to lists for JSON serialization
            detection_record = {
                "camera_id": camera_id,
                "worker_name": worker_name,
                "frame_id": frame_id,
                "timestamp": timestamp,
                "num_detections": len(boxes),
                "execution_time_ms": execution_time_ms,
                "boxes": boxes.tolist() if len(boxes) > 0 else [],
                "confidences": confidences.tolist() if len(confidences) > 0 else [],
                "class_ids": class_ids.tolist() if len(class_ids) > 0 else [],
            }

            if metadata:
                detection_record.update(metadata)

            # Create stream key
            stream_key = f"{DETECTIONS_STREAM_PREFIX}{worker_name}:{camera_id}"

            # Add to stream with LIFO trimming
            # Convert detection record to Redis stream format
            stream_data = {
                b"detections": json.dumps(detection_record).encode()
            }
            stream_id = self.redis_client.xadd(
                stream_key,
                stream_data,
                maxlen=MAX_DETECTIONS_STREAM_LENGTH,
                approximate=False
            )

            logger.debug(
                f"Stored {len(boxes)} detections for {camera_id} "
                f"via {worker_name} (stream: {stream_key})"
            )

            return stream_id

        except Exception as e:
            logger.error(f"Failed to store detection: {e}")
            raise

    def get_latest_detections(
        self,
        camera_id: str,
        worker_name: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict]:
        """
        Retrieve latest detections from stream(s).
        
        Args:
            camera_id: Camera identifier
            worker_name: Specific worker to query (if None, query all workers)
            limit: Number of latest detections to retrieve
            
        Returns:
            List of detection records
        """
        try:
            detections = []

            if worker_name:
                stream_key = f"{DETECTIONS_STREAM_PREFIX}{worker_name}:{camera_id}"
                stream_data = self.redis_client.xrevrange(stream_key, count=limit)
                detections.extend(self._parse_stream_data(stream_data))
            else:
                # Query all worker streams for this camera
                worker_names = ["yolo_vit", "frcnn_rtdetr"]
                for wname in worker_names:
                    stream_key = f"{DETECTIONS_STREAM_PREFIX}{wname}:{camera_id}"
                    stream_data = self.redis_client.xrevrange(stream_key, count=limit)
                    detections.extend(self._parse_stream_data(stream_data))

            return sorted(detections, key=lambda x: x["timestamp"], reverse=True)[:limit]

        except Exception as e:
            logger.error(f"Failed to retrieve detections: {e}")
            return []

    def get_detection_stats(
        self,
        camera_id: str,
        worker_name: Optional[str] = None,
    ) -> Dict:
        """
        Get statistics on detections for a camera.
        
        Args:
            camera_id: Camera identifier
            worker_name: Specific worker (if None, aggregate across all)
            
        Returns:
            Statistics dictionary
        """
        try:
            stats = {
                "camera_id": camera_id,
                "worker_stats": {},
            }

            worker_names = [worker_name] if worker_name else ["yolo_vit", "frcnn_rtdetr"]

            for wname in worker_names:
                stream_key = f"{DETECTIONS_STREAM_PREFIX}{wname}:{camera_id}"
                stream_length = self.redis_client.xlen(stream_key)

                # Get latest 100 detections for stats
                stream_data = self.redis_client.xrevrange(stream_key, count=100)
                detections = self._parse_stream_data(stream_data)

                if detections:
                    confidences = [d["confidences"] for d in detections if d["confidences"]]
                    flat_confidences = [c for conf_list in confidences for c in conf_list]

                    stats["worker_stats"][wname] = {
                        "stream_length": stream_length,
                        "total_detections": sum(d["num_detections"] for d in detections),
                        "avg_detections_per_frame": sum(d["num_detections"] for d in detections) / len(detections) if detections else 0,
                        "avg_confidence": np.mean(flat_confidences) if flat_confidences else 0,
                        "avg_execution_time_ms": np.mean([d["execution_time_ms"] for d in detections]) if detections else 0,
                    }

            return stats

        except Exception as e:
            logger.error(f"Failed to get detection stats: {e}")
            return {}

    def _parse_stream_data(self, stream_data: List[tuple]) -> List[Dict]:
        """Parse Redis stream data into detection records."""
        detections = []
        for stream_id, data in stream_data:
            try:
                # Data has single key "detections" with JSON-encoded full record
                if b"detections" in data:
                    record_json = json.loads(data[b"detections"].decode())
                    record_json["stream_id"] = stream_id.decode()
                    detections.append(record_json)
                else:
                    # Fallback for old format with individual fields
                    record = {k.decode(): v.decode() for k, v in data.items()}
                    record["timestamp"] = float(record.get("timestamp", 0))
                    record["num_detections"] = int(record.get("num_detections", 0))
                    record["execution_time_ms"] = float(record.get("execution_time_ms", 0))
                    record["boxes"] = json.loads(record.get("boxes", "[]"))
                    record["confidences"] = json.loads(record.get("confidences", "[]"))
                    record["class_ids"] = json.loads(record.get("class_ids", "[]"))
                    detections.append(record)
            except Exception as e:
                logger.warning(f"Failed to parse detection record: {e}")
                continue

        return detections

    def delete_detections_stream(self, camera_id: str, worker_name: Optional[str] = None) -> None:
        """
        Delete detection streams (cleanup).
        
        Args:
            camera_id: Camera identifier
            worker_name: Specific worker (if None, delete all)
        """
        try:
            worker_names = [worker_name] if worker_name else ["yolo_vit", "frcnn_rtdetr"]

            for wname in worker_names:
                stream_key = f"{DETECTIONS_STREAM_PREFIX}{wname}:{camera_id}"
                self.redis_client.delete(stream_key)
                logger.info(f"Deleted stream: {stream_key}")

        except Exception as e:
            logger.error(f"Failed to delete detection streams: {e}")
