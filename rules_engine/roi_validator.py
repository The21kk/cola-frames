"""
ROI Validator: Validate detections against Region of Interest (ROI) constraints.

Supports:
- Inclusion regions (only these areas are valid)
- Exclusion regions (these areas are invalid)
- Per-camera ROI configuration
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

from config.settings import REDIS_HOST, REDIS_PORT, REDIS_DB


logger = logging.getLogger(__name__)


class ROIValidator:
    """
    Validate detections against Region of Interest (ROI) constraints.
    
    ROI format per camera: List of regions
    {
        "inclusion_regions": [(x1, y1, x2, y2), ...],  # Only detect here
        "exclusion_regions": [(x1, y1, x2, y2), ...],  # Don't detect here
    }
    """

    def __init__(self, roi_config: Optional[Dict] = None):
        """
        Initialize ROI validator.
        
        Args:
            roi_config: Dictionary mapping camera_id -> ROI configuration
            
            Example:
            {
                "camera:1": {
                    "inclusion_regions": [(100, 100, 500, 500)],
                    "exclusion_regions": [(200, 200, 300, 300)],
                },
                "camera:2": {
                    "inclusion_regions": [],  # No restriction
                    "exclusion_regions": [],
                },
            }
        """
        self.roi_config = roi_config or {}
        logger.info(f"ROI Validator initialized with {len(self.roi_config)} camera configurations")

    def validate_detections(
        self,
        camera_id: str,
        detections: List[Dict],
    ) -> List[Dict]:
        """
        Validate detections against ROI constraints.
        
        Args:
            camera_id: Camera identifier
            detections: List of detection dictionaries with 'boxes' key
            
        Returns:
            Filtered detections that pass ROI validation
        """
        if camera_id not in self.roi_config:
            # No ROI constraints for this camera
            return detections

        roi = self.roi_config[camera_id]
        inclusion_regions = roi.get("inclusion_regions", [])
        exclusion_regions = roi.get("exclusion_regions", [])

        validated = []

        for det in detections:
            boxes = np.array(det.get("boxes", []))

            if len(boxes) == 0:
                validated.append(det)
                continue

            # Check each box
            valid_boxes = []
            valid_confidences = []
            valid_class_ids = []

            for idx, box in enumerate(boxes):
                if self._validate_box(box, inclusion_regions, exclusion_regions):
                    valid_boxes.append(box)

                    # Keep corresponding confidence and class_id
                    confidences = det.get("confidences", [])
                    if idx < len(confidences):
                        valid_confidences.append(confidences[idx])

                    class_ids = det.get("class_ids", [])
                    if idx < len(class_ids):
                        valid_class_ids.append(class_ids[idx])

            # Create new detection record with validated boxes
            if valid_boxes:
                validated_det = det.copy()
                validated_det["boxes"] = valid_boxes
                validated_det["confidences"] = valid_confidences
                validated_det["class_ids"] = valid_class_ids
                validated_det["num_detections"] = len(valid_boxes)
                validated_det["roi_filtered"] = len(boxes) - len(valid_boxes)
                validated.append(validated_det)

        logger.debug(
            f"ROI validation for {camera_id}: {len(detections)} → {len(validated)} detections "
            f"(inclusion: {len(inclusion_regions)}, exclusion: {len(exclusion_regions)})"
        )

        return validated

    def _validate_box(
        self,
        box: np.ndarray,
        inclusion_regions: List[Tuple],
        exclusion_regions: List[Tuple],
    ) -> bool:
        """
        Validate a single bounding box against ROI constraints.
        
        Returns:
            True if box passes validation, False otherwise
        """
        # Check inclusion regions (if specified)
        if inclusion_regions:
            in_any_inclusion = any(
                self._box_in_region(box, region)
                for region in inclusion_regions
            )
            if not in_any_inclusion:
                return False

        # Check exclusion regions
        for region in exclusion_regions:
            if self._box_in_region(box, region):
                return False

        return True

    def _box_in_region(self, box: np.ndarray, region: Tuple) -> bool:
        """Check if bounding box overlaps with region."""
        x1_box, y1_box, x2_box, y2_box = box
        x1_reg, y1_reg, x2_reg, y2_reg = region

        # Check if box center is in region (simple check)
        center_x = (x1_box + x2_box) / 2
        center_y = (y1_box + y2_box) / 2

        return (x1_reg <= center_x <= x2_reg and y1_reg <= center_y <= y2_reg)

    def set_camera_roi(self, camera_id: str, roi: Dict) -> None:
        """
        Set ROI configuration for a camera.
        
        Args:
            camera_id: Camera identifier
            roi: ROI configuration dictionary
        """
        self.roi_config[camera_id] = roi
        logger.info(f"Updated ROI configuration for {camera_id}")

    def get_camera_roi(self, camera_id: str) -> Optional[Dict]:
        """Get ROI configuration for a camera."""
        return self.roi_config.get(camera_id)

    def list_cameras_with_roi(self) -> List[str]:
        """List all cameras with ROI configuration."""
        return list(self.roi_config.keys())
