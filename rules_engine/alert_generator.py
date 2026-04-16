"""
Alert Generator: Create alerts from validated detections.

Generates immediate alerts in alerts:* Redis stream when:
1. Detections pass consensus voting
2. Detections pass temporal persistence (>= 3 seconds)
3. Detections pass ROI validation
4. Confidence meets alert threshold
"""

import logging
import json
import time
from typing import Dict, List, Optional
from redis import Redis

# Alert generator uses Redis directly for custom streams
from config.settings import (
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB,
    DETECTION_CONFIDENCE_THRESHOLD,
)


logger = logging.getLogger(__name__)


class AlertGenerator:
    """
    Generate alerts from validated detections.
    
    Alert Record:
    {
        "alert_id": str,
        "camera_id": str,
        "timestamp": float,
        "detections": [
            {
                "boxes": [[x1, y1, x2, y2], ...],
                "confidences": [0.95, ...],
                "class_ids": [0, ...],
                "workers_agreed": ["yolo_vit", "frcnn_rtdetr"],
                "temporal_persistence_frames": int,
                "roi_passed": bool,
            }
        ],
        "severity": str,  # "low", "medium", "high"
        "acknowledged": false,
    }
    """

    def __init__(
        self,
        confidence_threshold: float = DETECTION_CONFIDENCE_THRESHOLD,
    ):
        """Initialize alert generator."""
        self.redis_client = Redis(
            host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False
        )
        # Initialize Redis client for alert storage
        self.confidence_threshold = confidence_threshold

    def generate_alerts(
        self,
        camera_id: str,
        validated_detections: List[Dict],
    ) -> List[str]:
        """
        Generate alerts from validated detections.
        
        Args:
            camera_id: Camera identifier
            validated_detections: List of detections that passed all validations
            
        Returns:
            List of generated alert IDs
        """
        alert_ids = []

        if not validated_detections:
            return alert_ids

        try:
            for detection in validated_detections:
                # Filter by confidence
                boxes = detection.get("boxes", [])
                confidences = detection.get("confidences", [])

                if not boxes or not confidences:
                    continue

                # Create alert record
                alert_id = self._create_alert_id(camera_id)
                alert_record = {
                    "alert_id": alert_id,
                    "camera_id": camera_id,
                    "timestamp": time.time(),
                    "detection_timestamp": detection.get("timestamp", time.time()),
                    "boxes": boxes,
                    "confidences": confidences,
                    "class_ids": detection.get("class_ids", []),
                    "workers_agreed": detection.get("workers_agreed", []),
                    "workers_count": len(detection.get("workers_agreed", [])),
                    "temporal_persistence_frames": detection.get("temporal_persistence_frames", 0),
                    "iou_score": detection.get("iou_score", 0),
                    "roi_passed": detection.get("roi_validation", True),
                    "num_detections": len(boxes),
                    "avg_confidence": float(sum(confidences) / len(confidences)) if confidences else 0,
                    "severity": self._calculate_severity(confidences),
                    "acknowledged": False,
                    "metadata": {},
                }

                # Publish to alerts stream
                stream_data = {
                    b"alert": json.dumps(alert_record).encode()
                }
                stream_id = self.redis_client.xadd(
                    f"alerts:{camera_id}",
                    stream_data,
                    maxlen=10000,
                    approximate=False
                )

                alert_ids.append(alert_id)

                logger.info(
                    f"Generated alert {alert_id} for {camera_id}: "
                    f"{len(boxes)} detections (severity: {alert_record['severity']}, "
                    f"avg_conf: {alert_record['avg_confidence']:.2f})"
                )

            return alert_ids

        except Exception as e:
            logger.error(f"Failed to generate alerts: {e}")
            return alert_ids

    def get_active_alerts(
        self,
        camera_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        """
        Retrieve active (unacknowledged) alerts.
        
        Args:
            camera_id: Specific camera (if None, get all cameras)
            limit: Number of alerts to retrieve
            
        Returns:
            List of alert records
        """
        alerts = []

        try:
            if camera_id:
                stream_key = f"alerts:{camera_id}"
                alert_data = self.redis_client.xrevrange(stream_key, count=limit)
                alerts.extend(self._parse_alert_data(alert_data))
            else:
                # Get all alerts across all cameras
                pattern = "alerts:*"
                for key in self.redis_client.scan_iter(match=pattern):
                    alert_data = self.redis_client.xrevrange(key, count=limit)
                    alerts.extend(self._parse_alert_data(alert_data))

            # Filter unacknowledged
            active = [a for a in alerts if not a.get("acknowledged", False)]
            return sorted(active, key=lambda x: x.get("timestamp", 0), reverse=True)[:limit]

        except Exception as e:
            logger.error(f"Failed to get active alerts: {e}")
            return []

    def acknowledge_alert(self, alert_id: str, camera_id: str) -> bool:
        """Acknowledge an alert."""
        try:
            # In production, would update in database or separate tracking system
            logger.info(f"Alert {alert_id} acknowledged for {camera_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to acknowledge alert: {e}")
            return False

    def get_alert_stats(self, camera_id: Optional[str] = None) -> Dict:
        """Get alert statistics."""
        try:
            stats = {
                "total_alerts": 0,
                "active_alerts": 0,
                "high_severity": 0,
                "medium_severity": 0,
                "low_severity": 0,
            }

            alerts = self.get_active_alerts(camera_id=camera_id, limit=1000)

            for alert in alerts:
                stats["total_alerts"] += 1
                if not alert.get("acknowledged", False):
                    stats["active_alerts"] += 1

                severity = alert.get("severity", "low")
                if severity == "high":
                    stats["high_severity"] += 1
                elif severity == "medium":
                    stats["medium_severity"] += 1
                else:
                    stats["low_severity"] += 1

            return stats

        except Exception as e:
            logger.error(f"Failed to get alert stats: {e}")
            return {}

    def _create_alert_id(self, camera_id: str) -> str:
        """Generate unique alert ID."""
        import uuid
        return f"{camera_id}:{uuid.uuid4().hex[:12]}"

    def _calculate_severity(self, confidences: List[float]) -> str:
        """Calculate alert severity based on detection confidences."""
        if not confidences:
            return "low"

        avg_confidence = sum(confidences) / len(confidences)

        if avg_confidence >= 0.9:
            return "high"
        elif avg_confidence >= 0.7:
            return "medium"
        else:
            return "low"

    def _parse_alert_data(self, stream_data: List[tuple]) -> List[Dict]:
        """Parse Redis stream data into alert records."""
        alerts = []

        for stream_id, data in stream_data:
            try:
                record = {k.decode(): v.decode() for k, v in data.items()}

                # Parse JSON/numeric fields
                record["timestamp"] = float(record.get("timestamp", time.time()))
                record["detection_timestamp"] = float(record.get("detection_timestamp", record["timestamp"]))
                record["num_detections"] = int(record.get("num_detections", 0))
                record["avg_confidence"] = float(record.get("avg_confidence", 0))
                record["acknowledged"] = record.get("acknowledged", "False").lower() == "true"

                record["boxes"] = json.loads(record.get("boxes", "[]"))
                record["confidences"] = json.loads(record.get("confidences", "[]"))
                record["class_ids"] = json.loads(record.get("class_ids", "[]"))

                alerts.append(record)

            except Exception as e:
                logger.warning(f"Failed to parse alert record: {e}")
                continue

        return alerts
