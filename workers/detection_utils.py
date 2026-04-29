"""
Shared detection utilities for ensemble consensus and IoU calculations.

Provides reusable functions for:
- IoU (Intersection over Union) calculations between bounding boxes
- Detection matching across models
- Consensus voting between model outputs
"""

import logging
from typing import Dict, List, Tuple
import numpy as np

from config.settings import CONSENSUS_IOU_THRESHOLD

logger = logging.getLogger(__name__)


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate IoU (Intersection over Union) between two boxes.
    
    Boxes format: [x_min, y_min, x_max, y_max]
    
    Args:
        box1: First bounding box [x1_min, y1_min, x1_max, y1_max]
        box2: Second bounding box [x2_min, y2_min, x2_max, y2_max]
        
    Returns:
        IoU score (0.0 to 1.0)
    """
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


def detections_match(
    det1: Dict,
    det2: Dict,
    iou_threshold: float = CONSENSUS_IOU_THRESHOLD,
) -> bool:
    """
    Check if two detections match spatially using IoU.
    
    Args:
        det1: First detection dict with "boxes" key
        det2: Second detection dict with "boxes" key
        iou_threshold: Minimum IoU to consider as match
        
    Returns:
        True if detections match (IoU >= threshold)
    """
    boxes1 = np.array(det1.get("boxes", []))
    boxes2 = np.array(det2.get("boxes", []))

    if len(boxes1) == 0 or len(boxes2) == 0:
        return False

    # Match first box in each detection
    iou = calculate_iou(boxes1[0], boxes2[0])
    return iou >= iou_threshold


def consensus_two_detections(
    det1: Dict,
    det2: Dict,
    iou_threshold: float = CONSENSUS_IOU_THRESHOLD,
    model_names: Tuple[str, str] = ("model_1", "model_2"),
) -> Dict:
    """
    Create consensus detection from two model outputs via IoU matching.
    
    Only returns detections that both models agree on (IoU >= threshold).
    If no agreement, returns empty detection dict.
    
    Args:
        det1: First detection dict
        det2: Second detection dict
        iou_threshold: Minimum IoU for consensus
        model_names: Tuple of (model1_name, model2_name) for tracking
        
    Returns:
        Consensus detection dict with:
        - boxes, confidences, class_ids (from agreed detections)
        - num_detections
        - models_agreed: list of models that agreed
        - avg_confidence: average confidence
        - num_matches: count of IoU matches
    """
    consensus_dets = []

    boxes1 = np.array(det1.get("boxes", []))
    boxes2 = np.array(det2.get("boxes", []))

    if len(boxes1) == 0 or len(boxes2) == 0:
        # Return empty consensus if either model has no detections
        return {
            "boxes": np.array([]),
            "confidences": np.array([]),
            "class_ids": np.array([]),
            "num_detections": 0,
            "models_agreed": list(model_names),
            "avg_confidence": 0.0,
            "num_matches": 0,
        }

    # Match boxes from det1 to det2
    matched_pairs = []
    used_det2_indices = set()

    for idx1, box1 in enumerate(boxes1):
        best_match = None
        best_iou = 0

        for idx2, box2 in enumerate(boxes2):
            if idx2 in used_det2_indices:
                continue

            iou = calculate_iou(box1, box2)

            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_match = (idx2, iou)

        if best_match is not None:
            idx2, iou = best_match
            used_det2_indices.add(idx2)

            # Average confidences
            conf1 = np.atleast_1d(det1.get("confidences", []))
            conf2 = np.atleast_1d(det2.get("confidences", []))

            if len(conf1) > idx1 and len(conf2) > idx2:
                avg_conf = (conf1[idx1] + conf2[idx2]) / 2.0
            else:
                avg_conf = np.mean(np.concatenate([conf1, conf2])) if len(conf1) > 0 or len(conf2) > 0 else 0

            matched_pairs.append({
                "box": box1,
                "confidence": avg_conf,
                "class_id": det1.get("class_ids", [])[idx1] if len(det1.get("class_ids", [])) > idx1 else 0,
                "iou": iou,
            })

    # Build consensus output
    if len(matched_pairs) > 0:
        consensus_boxes = np.array([p["box"] for p in matched_pairs])
        consensus_confs = np.array([p["confidence"] for p in matched_pairs])
        consensus_class_ids = np.array([p["class_id"] for p in matched_pairs])
        
        return {
            "boxes": consensus_boxes.tolist() if isinstance(consensus_boxes, np.ndarray) else consensus_boxes,
            "confidences": consensus_confs.tolist() if isinstance(consensus_confs, np.ndarray) else list(consensus_confs),
            "class_ids": consensus_class_ids.tolist() if isinstance(consensus_class_ids, np.ndarray) else list(consensus_class_ids),
            "num_detections": len(matched_pairs),
            "models_agreed": list(model_names),
            "avg_confidence": float(np.mean(consensus_confs)),
            "num_matches": len(matched_pairs),
        }
    else:
        return {
            "boxes": np.array([]),
            "confidences": np.array([]),
            "class_ids": np.array([]),
            "num_detections": 0,
            "models_agreed": list(model_names),
            "avg_confidence": 0.0,
            "num_matches": 0,
        }
