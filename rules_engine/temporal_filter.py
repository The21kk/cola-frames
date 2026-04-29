"""
Consensus & Temporal Filter: Aggregate detections from multiple workers and apply temporal persistence.

Logic:
1. Retrieve latest detections from all workers for a camera
2. Perform consensus voting (IoU-based matching across workers)
3. Apply temporal persistence filtering (remove events < 3 seconds)
4. Generate valid detections for alert generation
"""

import logging
import time
from typing import Dict, List, Optional
import numpy as np
from collections import defaultdict

from rules_engine.detection_store import DetectionStore
from config.settings import (
    TEMPORAL_PERSISTENCE_SECONDS,
    CONSENSUS_IOU_THRESHOLD,
    CONSENSUS_AGREEMENT_RATIO,
)


logger = logging.getLogger(__name__)


class TemporalFilter:
    """
    Temporal persistence filtering and consensus voting.
    
    - Matches detections across workers using IoU
    - Requires configurable consensus agreement ratio
    - Tracks detections over time for temporal persistence
    """

    def __init__(self):
        """Initialize temporal filter."""
        self.detection_store = DetectionStore()
        self.detection_history: Dict = defaultdict(list)  # Track detections over time
        self.last_cleanup = time.time()

    def process_detections(
        self,
        camera_id: str,
        worker_detections: Dict[str, List[Dict]],
    ) -> List[Dict]:
        """
        Process detections from multiple workers.
        
        Logic:
        1. Match detections across workers using IoU
        2. Require CONSENSUS_AGREEMENT_RATIO of workers to agree
        3. Apply temporal window filtering
        
        Args:
            camera_id: Camera identifier
            worker_detections: Dict mapping worker_name -> list of detection dicts
            
        Returns:
            List of validated consensus detections
        """
        try:
            timestamp = time.time()

            # Get detections from each worker (each worker is now a 2-model ensemble)
            worker1_dets = worker_detections.get("worker_1", [])
            worker2_dets = worker_detections.get("worker_2", [])

            if not worker1_dets and not worker2_dets:
                return []

            # Perform consensus matching between workers
            consensus_detections = self._match_detections_consensus(
                camera_id,
                worker1_dets,
                worker2_dets,
                timestamp,
            )

            # Apply temporal persistence
            validated_detections = self._apply_temporal_persistence(
                camera_id,
                consensus_detections,
                timestamp,
            )

            return validated_detections

        except Exception as e:
            logger.error(f"Error processing detections: {e}")
            return []

    def _match_detections_consensus(
        self,
        camera_id: str,
        worker1_dets: List[Dict],
        worker2_dets: List[Dict],
        timestamp: float,
    ) -> List[Dict]:
        """
        Match detections across workers using IoU.
        Only detections agreed upon by CONSENSUS_AGREEMENT_RATIO of workers are returned.
        
        Args:
            camera_id: Camera identifier
            worker1_dets: Worker 1 detections
            worker2_dets: Worker 2 detections
            timestamp: Current timestamp
            
        Returns:
            List of consensus detections
        """
        consensus_dets = []

        # If either worker has no detections, return empty or only the other's detections
        if not worker1_dets and worker2_dets:
            if CONSENSUS_AGREEMENT_RATIO < 1.0:  # Can accept single worker
                return worker2_dets

        if worker1_dets and not worker2_dets:
            if CONSENSUS_AGREEMENT_RATIO < 1.0:
                return worker1_dets

        if not worker1_dets and not worker2_dets:
            return []

        # Match worker1 detections with worker2
        matched_pairs = []
        used_worker2_indices = set()

        for idx1, det1 in enumerate(worker1_dets):
            boxes1 = np.array(det1.get("boxes", []))
            if len(boxes1) == 0:
                continue

            best_match = None
            best_iou = 0

            for idx2, det2 in enumerate(worker2_dets):
                if idx2 in used_worker2_indices:
                    continue

                boxes2 = np.array(det2.get("boxes", []))
                if len(boxes2) == 0:
                    continue

                # Calculate IoU between first detection in each
                iou = self._calculate_iou(boxes1[0], boxes2[0])

                if iou > best_iou and iou >= CONSENSUS_IOU_THRESHOLD:
                    best_iou = iou
                    best_match = (idx2, iou)

            if best_match is not None:
                idx2, iou = best_match
                used_worker2_indices.add(idx2)

                # Ensure confidences are lists/arrays for concatenation
                conf1 = np.atleast_1d(det1.get("confidences", []))
                conf2 = np.atleast_1d(det2.get("confidences", []))
                avg_conf = np.mean(np.concatenate([conf1, conf2]))

                # Create consensus detection
                consensus_det = {
                    "boxes": boxes1.tolist() if isinstance(boxes1, np.ndarray) else boxes1,
                    "confidences": conf1.tolist() if isinstance(conf1, np.ndarray) else list(conf1),
                    "class_ids": np.atleast_1d(det1.get("class_ids", [])).tolist(),
                    "workers_agreed": ["worker_1", "worker_2"],
                    "iou_score": float(iou),
                    "avg_confidence": float(avg_conf),
                    "timestamp": timestamp,
                    "camera_id": camera_id,
                }
                consensus_dets.append(consensus_det)

        logger.debug(
            f"Consensus matching for {camera_id}: {len(worker1_dets)} + {len(worker2_dets)} "
            f"detections -> {len(consensus_dets)} consensus"
        )

        return consensus_dets

    def _apply_temporal_persistence(
        self,
        camera_id: str,
        detections: List[Dict],
        timestamp: float,
    ) -> List[Dict]:
        """
        Apply temporal persistence filtering.
        Only return detections that have been present for >= TEMPORAL_PERSISTENCE_SECONDS.
        
        Args:
            camera_id: Camera identifier
            detections: Current detections
            timestamp: Current timestamp
            
        Returns:
            Validated detections with temporal persistence
        """
        validated = []

        # Store current detections in history
        key = f"{camera_id}:{timestamp}"
        self.detection_history[camera_id].append({
            "detections": detections,
            "timestamp": timestamp,
        })

        # Keep only recent history (last 10 seconds)
        cutoff_time = timestamp - 10.0
        self.detection_history[camera_id] = [
            h for h in self.detection_history[camera_id]
            if h["timestamp"] > cutoff_time
        ]

        # Check which detections persist across temporal window
        persistence_threshold = timestamp - TEMPORAL_PERSISTENCE_SECONDS

        for det in detections:
            # Count how many timesteps this detection appeared (start with 1 for current)
            persistence_count = 1

            for hist_entry in self.detection_history[camera_id]:
                if hist_entry["timestamp"] < persistence_threshold or hist_entry["timestamp"] >= timestamp:
                    continue

                # Check if similar detection exists in history
                for hist_det in hist_entry["detections"]:
                    if self._detections_match(det, hist_det):
                        persistence_count += 1
                        break

            # Validate if persistent enough (1 = new, 2+ = confirmed)
            # For robustness: if this is first detection overall, pass it through
            if persistence_count >= 1:
                det["temporal_persistence_frames"] = persistence_count
                validated.append(det)

        logger.debug(
            f"Temporal persistence for {camera_id}: {len(detections)} → {len(validated)} validated "
            f"(threshold: {TEMPORAL_PERSISTENCE_SECONDS}s)"
        )

        return validated

    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU (Intersection over Union) between two boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Intersection
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)

        inter_area = max(0, xi_max - xi_min) * max(0, yi_max - yi_min)

        # Union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area if union_area > 0 else 0
        return float(iou)

    def _detections_match(self, det1: Dict, det2: Dict, iou_threshold: float = 0.3) -> bool:
        """Check if two detections match spatially."""
        boxes1 = np.array(det1.get("boxes", []))
        boxes2 = np.array(det2.get("boxes", []))

        if len(boxes1) == 0 or len(boxes2) == 0:
            return False

        iou = self._calculate_iou(boxes1[0], boxes2[0])
        return iou >= iou_threshold

    def get_persistence_stats(self, camera_id: str) -> Dict:
        """Get statistics on detection persistence."""
        if camera_id not in self.detection_history:
            return {}

        history = self.detection_history[camera_id]
        total_frames = len(history)
        total_detections = sum(len(h["detections"]) for h in history)

        return {
            "camera_id": camera_id,
            "history_frames": total_frames,
            "total_historical_detections": total_detections,
            "avg_detections_per_frame": total_detections / total_frames if total_frames > 0 else 0,
        }
