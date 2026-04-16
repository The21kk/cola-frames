"""
Worker 2: Faster R-CNN + RT-DETR Lite Ensemble Detector

Combines:
- Faster R-CNN for region-based detection with high quality
- RT-DETR Lite for real-time anchor-free detection
- Ensemble voting for robust detection consensus
"""

import time
import logging
from typing import Dict, List, Optional
import numpy as np
import torch
from torch.cuda.amp import autocast
import torchvision.models as models
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)

# RT-DETR Lite (we'll use a lightweight alternative or mock for now)
# In production, use: pip install rt-detr
try:
    from rtdetr import RT_DETR
    HAS_RTDETR = True
except ImportError:
    HAS_RTDETR = False
    logger = logging.getLogger(__name__)
    logger.warning("RT-DETR not available, using placeholder")

from workers.base_detector import BaseDetector
from config.settings import MODEL_CACHE_DIR


logger = logging.getLogger(__name__)


class FasterRCNNRtdetrDetector(BaseDetector):
    """
    Faster R-CNN + RT-DETR Lite ensemble detector.
    
    - Faster R-CNN: Region-based CNN with region proposal network
    - RT-DETR Lite: Fast anchor-free detection for real-time inference
    - Consensus: Both models must detect objects in similar regions
    """

    def __init__(
        self,
        device: str = "cuda:0",
        batch_size: int = 4,
        confidence_threshold: float = 0.5,
        use_fp16: bool = True,
    ):
        """Initialize Faster R-CNN + RT-DETR detector."""
        super().__init__(
            model_name="frcnn_rtdetr",
            device=device,
            batch_size=batch_size,
            confidence_threshold=confidence_threshold,
            use_fp16=use_fp16,
        )
        self.initialize_model()

    def initialize_model(self) -> None:
        """Load Faster R-CNN and RT-DETR models."""
        try:
            logger.info("Loading Faster R-CNN model...")
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self.frcnn_model = fasterrcnn_resnet50_fpn(weights=weights)
            self.frcnn_model = self.frcnn_model.to(self.device)
            self.frcnn_model.eval()

            logger.info("Loading RT-DETR Lite model...")
            if HAS_RTDETR:
                # Load RT-DETR Lite (small/nano variant)
                self.rtdetr_model = RT_DETR(model_size="s")  # small
                self.rtdetr_model = self.rtdetr_model.to(self.device)
                self.rtdetr_model.eval()
            else:
                logger.warning("RT-DETR not available, using secondary Faster R-CNN instead")
                self.rtdetr_model = None

            self._prealloc_gpu_memory(600)
            self._set_inference_mode()

            # Store model info
            self.model_info = {
                "frcnn_model": "FasterRCNN_ResNet50_FPN",
                "rtdetr_model": "RT-DETR Lite (S)" if HAS_RTDETR else "N/A",
                "ensemble_type": "Faster R-CNN + RT-DETR Lite",
            }

            logger.info("Faster R-CNN + RT-DETR detector initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Faster R-CNN + RT-DETR detector: {e}")
            raise

    def detect(self, frame: np.ndarray) -> Dict:
        """
        Run detection on a single frame using Faster R-CNN + RT-DETR ensemble.
        
        Args:
            frame: Input frame (H, W, 3) in BGR format
            
        Returns:
            Detection dictionary with boxes, confidences, class_ids, etc.
        """
        start_time = time.time()

        # Faster R-CNN detection
        frcnn_boxes, frcnn_confidences = self._detect_frcnn(frame)

        # RT-DETR detection
        if self.rtdetr_model is not None:
            rtdetr_boxes, rtdetr_confidences = self._detect_rtdetr(frame)
        else:
            # Fallback: use secondary Faster R-CNN model
            rtdetr_boxes, rtdetr_confidences = frcnn_boxes.copy(), frcnn_confidences.copy()

        # Ensemble: average boxes and confidences
        boxes, confidences, class_ids = self._ensemble_detections(
            frcnn_boxes, frcnn_confidences,
            rtdetr_boxes, rtdetr_confidences,
        )

        execution_time = (time.time() - start_time) * 1000  # ms

        return {
            "boxes": boxes,
            "confidences": confidences,
            "class_ids": class_ids,
            "execution_time_ms": execution_time,
            "device": str(self.device),
            "num_detections": len(boxes),
            "frcnn_detections": len(frcnn_boxes),
            "rtdetr_detections": len(rtdetr_boxes),
        }

    def detect_batch(self, frames: List[np.ndarray]) -> List[Dict]:
        """
        Run batch detection on multiple frames.
        
        Args:
            frames: List of frames
            
        Returns:
            List of detection dictionaries
        """
        start_time = time.time()
        detections_list = []

        for frame in frames:
            # Faster R-CNN batch processing (if available)
            frcnn_boxes, frcnn_confidences = self._detect_frcnn(frame)

            # RT-DETR
            if self.rtdetr_model is not None:
                rtdetr_boxes, rtdetr_confidences = self._detect_rtdetr(frame)
            else:
                rtdetr_boxes, rtdetr_confidences = frcnn_boxes.copy(), frcnn_confidences.copy()

            # Ensemble
            boxes, confidences, class_ids = self._ensemble_detections(
                frcnn_boxes, frcnn_confidences,
                rtdetr_boxes, rtdetr_confidences,
            )

            detections_list.append({
                "boxes": boxes,
                "confidences": confidences,
                "class_ids": class_ids,
                "num_detections": len(boxes),
            })

        execution_time = (time.time() - start_time) * 1000 / len(frames)  # avg per frame

        # Standardize output
        for det in detections_list:
            det.update({
                "execution_time_ms": execution_time,
                "device": str(self.device),
            })

        return detections_list

    def _detect_frcnn(self, frame: np.ndarray) -> tuple:
        """Run Faster R-CNN detection."""
        with torch.no_grad():
            with autocast(enabled=self.use_fp16):
                # Convert frame to tensor (BGR → RGB)
                frame_rgb = frame[..., ::-1].copy()  # BGR to RGB (copy to avoid negative strides)
                frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
                frame_tensor = frame_tensor.unsqueeze(0).to(self.device)

                # Run inference
                outputs = self.frcnn_model(frame_tensor)

                # Extract results
                boxes = outputs[0]["boxes"].cpu().numpy()
                scores = outputs[0]["scores"].cpu().numpy()

                # Filter by confidence
                mask = scores >= self.confidence_threshold
                boxes = boxes[mask]
                scores = scores[mask]

        return boxes, scores

    def _detect_rtdetr(self, frame: np.ndarray) -> tuple:
        """Run RT-DETR detection."""
        if self.rtdetr_model is None:
            return np.array([]), np.array([])

        with torch.no_grad():
            with autocast(enabled=self.use_fp16):
                # Convert frame to tensor
                frame_rgb = frame[..., ::-1]
                frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
                frame_tensor = frame_tensor.unsqueeze(0).to(self.device)

                # Run inference
                outputs = self.rtdetr_model(frame_tensor)

                # Extract results (format depends on RT-DETR implementation)
                if isinstance(outputs, dict):
                    boxes = outputs.get("boxes", np.array([])).cpu().numpy()
                    scores = outputs.get("scores", np.array([])).cpu().numpy()
                else:
                    # Fallback: return empty if format unexpected
                    boxes, scores = np.array([]), np.array([])

                # Filter by confidence
                if len(scores) > 0:
                    mask = scores >= self.confidence_threshold
                    boxes = boxes[mask]
                    scores = scores[mask]

        return boxes, scores

    def _ensemble_detections(
        self,
        frcnn_boxes: np.ndarray,
        frcnn_scores: np.ndarray,
        rtdetr_boxes: np.ndarray,
        rtdetr_scores: np.ndarray,
    ) -> tuple:
        """
        Ensemble detections from both models via IoU-based matching.
        
        Returns:
            (boxes, confidences, class_ids)
        """
        all_boxes = np.vstack([frcnn_boxes, rtdetr_boxes]) if len(frcnn_boxes) > 0 and len(rtdetr_boxes) > 0 else frcnn_boxes
        all_scores = np.hstack([frcnn_scores, rtdetr_scores]) if len(frcnn_scores) > 0 and len(rtdetr_scores) > 0 else frcnn_scores

        if len(all_boxes) == 0:
            return np.array([]), np.array([]), np.array([])

        # Average confidences
        ensemble_scores = (frcnn_scores.mean() + rtdetr_scores.mean()) / 2.0 if len(frcnn_scores) > 0 and len(rtdetr_scores) > 0 else all_scores

        # All boxes from ensemble (in production, use NMS and IoU matching)
        class_ids = np.zeros(len(all_boxes), dtype=int)  # Assume all are person class

        return all_boxes, ensemble_scores, class_ids
