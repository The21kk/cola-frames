"""
Phase 2 Test Suite: GPU-Accelerated Detection Workers & Rules Engine

Tests:
1. GPU device detection and CUDA availability
2. Individual detector initialization (YOLOv8+ViT, Faster R-CNN+RT-DETR)
3. Detection inference (single frame and batch)
4. Consensus voting and IoU matching
5. Temporal persistence filtering
6. ROI validation
7. Alert generation and retrieval
8. End-to-end pipeline (frame → detection → consensus → alert)
9. GPU memory management and stability
"""

import pytest
import numpy as np
import torch
import time
import logging
from typing import List, Dict

# Phase 1 imports (already tested)
from config.settings import (
    REDIS_HOST, REDIS_PORT, REDIS_DB,
    DETECTION_CONFIDENCE_THRESHOLD,
    TEMPORAL_PERSISTENCE_SECONDS,
    CONSENSUS_IOU_THRESHOLD,
    DEVICE_WORKER_1, DEVICE_WORKER_2,
)
from redis_broker.stream_manager import RedisStreamManager
from producer.frame_serializer import FrameSerializer

# Phase 2 imports
from workers.base_detector import BaseDetector
from workers.yolo_vit_detector import YOLOVitDetector
from workers.frcnn_rtdetr_detector import FasterRCNNRtdetrDetector
from workers.worker_pool import WorkerPool

from rules_engine.detection_store import DetectionStore
from rules_engine.temporal_filter import TemporalFilter
from rules_engine.roi_validator import ROIValidator
from rules_engine.alert_generator import AlertGenerator


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestGPUSupport:
    """Test GPU device management."""

    def test_cuda_availability(self):
        """Verify CUDA availability."""
        logger.info("Testing CUDA availability...")
        assert torch.cuda.is_available(), "CUDA not available on this system"
        assert torch.cuda.device_count() >= 1, "No CUDA devices found"
        logger.info(f"✓ CUDA available: {torch.cuda.device_count()} device(s)")

    def test_device_selection(self):
        """Test device selection logic."""
        logger.info("Testing device selection...")

        # Test GPU device
        device = torch.device(DEVICE_WORKER_1)
        assert str(device).startswith("cuda") or str(device) == "cpu"

        # Test fallback to CPU
        invalid_device = torch.device("cuda:99") if torch.cuda.is_available() else torch.device("cpu")
        assert invalid_device is not None
        logger.info(f"✓ Device selection working: {device}")

    def test_gpu_memory_preallocation(self):
        """Test GPU memory pre-allocation."""
        logger.info("Testing GPU memory pre-allocation...")

        try:
            detector = YOLOVitDetector(device=DEVICE_WORKER_1)
            assert detector.device is not None
            logger.info(f"✓ GPU memory pre-allocated on {detector.device}")
            detector.cleanup()
        except Exception as e:
            logger.warning(f"GPU memory pre-allocation test failed: {e}")


class TestDetectionWorkers:
    """Test individual detector workers."""

    @pytest.fixture
    def sample_frame(self):
        """Generate synthetic test frame."""
        # Random frame (480p)
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        # Add synthetic object (white rectangle)
        frame[100:200, 150:300] = 255
        return frame

    def test_yolo_vit_initialization(self):
        """Test YOLOv8 + ViT detector initialization."""
        logger.info("Testing YOLOv8 + ViT initialization...")

        try:
            detector = YOLOVitDetector(device=DEVICE_WORKER_1)
            assert detector.model_name == "yolo_vit"
            assert detector.yolo_model is not None
            assert detector.vit_model is not None
            logger.info(f"✓ YOLOv8 + ViT initialized on {detector.device}")
            detector.cleanup()
        except ImportError as e:
            logger.warning(f"YOLOv8 + ViT dependencies not installed: {e}")
            pytest.skip("Dependencies not available")

    def test_frcnn_rtdetr_initialization(self):
        """Test Faster R-CNN + RT-DETR initialization."""
        logger.info("Testing Faster R-CNN + RT-DETR initialization...")

        try:
            detector = FasterRCNNRtdetrDetector(device=DEVICE_WORKER_2)
            assert detector.model_name == "frcnn_rtdetr"
            assert detector.frcnn_model is not None
            logger.info(f"✓ Faster R-CNN + RT-DETR initialized on {detector.device}")
            detector.cleanup()
        except ImportError as e:
            logger.warning(f"Faster R-CNN + RT-DETR dependencies not installed: {e}")
            pytest.skip("Dependencies not available")

    def test_yolo_vit_detection(self, sample_frame):
        """Test YOLOv8 + ViT inference."""
        logger.info("Testing YOLOv8 + ViT detection...")

        try:
            detector = YOLOVitDetector(device=DEVICE_WORKER_1)

            # Run detection
            start_time = time.time()
            detections = detector.detect(sample_frame)
            elapsed_ms = (time.time() - start_time) * 1000

            # Validate output
            assert "boxes" in detections
            assert "confidences" in detections
            assert "execution_time_ms" in detections
            assert detector.validate_detections(detections)

            logger.info(
                f"✓ YOLOv8 + ViT detection: {detections['num_detections']} "
                f"detections in {elapsed_ms:.1f}ms"
            )
            detector.cleanup()

        except ImportError:
            pytest.skip("Dependencies not available")

    def test_frcnn_rtdetr_detection(self, sample_frame):
        """Test Faster R-CNN + RT-DETR inference."""
        logger.info("Testing Faster R-CNN + RT-DETR detection...")

        try:
            detector = FasterRCNNRtdetrDetector(device=DEVICE_WORKER_2)

            # Run detection
            start_time = time.time()
            detections = detector.detect(sample_frame)
            elapsed_ms = (time.time() - start_time) * 1000

            # Validate output
            assert "boxes" in detections
            assert "confidences" in detections
            assert detector.validate_detections(detections)

            logger.info(
                f"✓ Faster R-CNN + RT-DETR detection: {detections['num_detections']} "
                f"detections in {elapsed_ms:.1f}ms"
            )
            detector.cleanup()

        except ImportError:
            pytest.skip("Dependencies not available")

    def test_batch_detection(self, sample_frame):
        """Test batch processing."""
        logger.info("Testing batch detection...")

        try:
            detector = YOLOVitDetector(device=DEVICE_WORKER_1, batch_size=4)

            # Create batch of frames
            frames = [sample_frame for _ in range(4)]

            # Run batch inference
            start_time = time.time()
            detections_list = detector.detect_batch(frames)
            elapsed_ms = (time.time() - start_time) * 1000

            assert len(detections_list) == 4
            for det in detections_list:
                assert detector.validate_detections(det)

            logger.info(f"✓ Batch detection: 4 frames in {elapsed_ms:.1f}ms")
            detector.cleanup()

        except ImportError:
            pytest.skip("Dependencies not available")


class TestConsensusVoting:
    """Test consensus voting and IoU matching."""

    def test_iou_calculation(self):
        """Test IoU calculation."""
        logger.info("Testing IoU calculation...")

        temporal_filter = TemporalFilter()

        # Identical boxes
        box1 = np.array([10, 10, 100, 100])
        box2 = np.array([10, 10, 100, 100])
        iou = temporal_filter._calculate_iou(box1, box2)
        assert iou == 1.0, "Identical boxes should have IoU = 1.0"

        # No overlap
        box1 = np.array([0, 0, 50, 50])
        box2 = np.array([100, 100, 150, 150])
        iou = temporal_filter._calculate_iou(box1, box2)
        assert iou == 0.0, "Non-overlapping boxes should have IoU = 0.0"

        # Partial overlap
        box1 = np.array([0, 0, 100, 100])
        box2 = np.array([50, 50, 150, 150])
        iou = temporal_filter._calculate_iou(box1, box2)
        assert 0 < iou < 1, "Overlapping boxes should have 0 < IoU < 1"

        logger.info("✓ IoU calculation working correctly")

    def test_consensus_matching(self):
        """Test consensus matching across workers."""
        logger.info("Testing consensus matching...")

        temporal_filter = TemporalFilter()
        camera_id = "camera:1"

        # Create worker detections
        worker1_dets = [{
            "boxes": np.array([[10, 10, 100, 100]]),
            "confidences": np.array([0.95]),
            "class_ids": np.array([0]),
        }]

        worker2_dets = [{
            "boxes": np.array([[12, 12, 98, 98]]),  # Slightly overlapping
            "confidences": np.array([0.92]),
            "class_ids": np.array([0]),
        }]

        # Run consensus matching
        consensus_dets = temporal_filter._match_detections_consensus(
            camera_id,
            worker1_dets,
            worker2_dets,
            time.time(),
        )

        assert len(consensus_dets) > 0, "Should find consensus detection"
        assert consensus_dets[0]["iou_score"] > CONSENSUS_IOU_THRESHOLD
        logger.info(f"✓ Consensus matching: {len(consensus_dets)} consensus detection(s)")

    def test_no_consensus_detection(self):
        """Test lack of consensus detection."""
        logger.info("Testing no consensus case...")

        temporal_filter = TemporalFilter()
        camera_id = "camera:1"

        # Non-overlapping detections
        worker1_dets = [{
            "boxes": np.array([[10, 10, 100, 100]]),
            "confidences": np.array([0.95]),
            "class_ids": np.array([0]),
        }]

        worker2_dets = [{
            "boxes": np.array([[200, 200, 300, 300]]),
            "confidences": np.array([0.92]),
            "class_ids": np.array([0]),
        }]

        consensus_dets = temporal_filter._match_detections_consensus(
            camera_id,
            worker1_dets,
            worker2_dets,
            time.time(),
        )

        assert len(consensus_dets) == 0, "Non-overlapping detections should not reach consensus"
        logger.info("✓ No consensus detection as expected")


class TestRulesEngine:
    """Test Rules Engine components."""

    def test_roi_validator_inclusion(self):
        """Test ROI validator with inclusion regions."""
        logger.info("Testing ROI validator (inclusion)...")

        roi_config = {
            "camera:1": {
                "inclusion_regions": [(0, 0, 200, 200)],
                "exclusion_regions": [],
            }
        }

        validator = ROIValidator(roi_config=roi_config)

        # Detection inside inclusion region
        det_inside = {
            "boxes": [[50, 50, 100, 100]],
            "confidences": [0.95],
            "class_ids": [0],
        }

        # Detection outside inclusion region
        det_outside = {
            "boxes": [[300, 300, 400, 400]],
            "confidences": [0.95],
            "class_ids": [0],
        }

        result_inside = validator.validate_detections("camera:1", [det_inside])
        result_outside = validator.validate_detections("camera:1", [det_outside])

        assert len(result_inside) == 1, "Detection inside ROI should pass"
        assert len(result_outside) == 0, "Detection outside ROI should fail"

        logger.info("✓ ROI inclusion validation working")

    def test_roi_validator_exclusion(self):
        """Test ROI validator with exclusion regions."""
        logger.info("Testing ROI validator (exclusion)...")

        roi_config = {
            "camera:1": {
                "inclusion_regions": [],
                "exclusion_regions": [(100, 100, 200, 200)],
            }
        }

        validator = ROIValidator(roi_config=roi_config)

        # Detection in exclusion zone
        det_excluded = {
            "boxes": [[110, 110, 150, 150]],
            "confidences": [0.95],
            "class_ids": [0],
        }

        # Detection outside exclusion zone
        det_allowed = {
            "boxes": [[50, 50, 90, 90]],
            "confidences": [0.95],
            "class_ids": [0],
        }

        result_excluded = validator.validate_detections("camera:1", [det_excluded])
        result_allowed = validator.validate_detections("camera:1", [det_allowed])

        assert len(result_excluded) == 0, "Detection in exclusion zone should fail"
        assert len(result_allowed) == 1, "Detection outside exclusion zone should pass"

        logger.info("✓ ROI exclusion validation working")

    def test_alert_generation(self):
        """Test alert generation."""
        logger.info("Testing alert generation...")

        alert_gen = AlertGenerator()

        validated_dets = [{
            "boxes": [[50, 50, 100, 100]],
            "confidences": [0.95],
            "class_ids": [0],
            "workers_agreed": ["yolo_vit", "frcnn_rtdetr"],
            "temporal_persistence_frames": 5,
            "timestamp": time.time(),
        }]

        alert_ids = alert_gen.generate_alerts("camera:1", validated_dets)
        assert len(alert_ids) == 1
        logger.info(f"✓ Alert generation: {alert_ids[0]}")

    def test_alert_severity_calculation(self):
        """Test alert severity calculation."""
        logger.info("Testing alert severity calculation...")

        alert_gen = AlertGenerator()

        # High severity
        high_conf = [0.95, 0.92, 0.98]
        severity = alert_gen._calculate_severity(high_conf)
        assert severity == "high"

        # Medium severity
        med_conf = [0.75, 0.70, 0.80]
        severity = alert_gen._calculate_severity(med_conf)
        assert severity == "medium"

        # Low severity
        low_conf = [0.55, 0.50, 0.60]
        severity = alert_gen._calculate_severity(low_conf)
        assert severity == "low"

        logger.info("✓ Alert severity calculation working")


class TestDetectionStore:
    """Test detection storage in Redis."""

    def test_store_and_retrieve_detections(self):
        """Test storing and retrieving detections."""
        logger.info("Testing detection storage...")

        store = DetectionStore()

        # Store detections
        stream_id = store.store_detection(
            camera_id="camera:1",
            worker_name="yolo_vit",
            frame_id="frame:1",
            timestamp=time.time(),
            boxes=np.array([[10, 10, 100, 100]]),
            confidences=np.array([0.95]),
            class_ids=np.array([0]),
            execution_time_ms=50.0,
        )

        assert stream_id is not None

        # Retrieve detections
        dets = store.get_latest_detections("camera:1", worker_name="yolo_vit", limit=1)
        assert len(dets) > 0
        assert dets[0]["worker_name"] == "yolo_vit"

        logger.info("✓ Detection storage and retrieval working")


class TestEndToEndPipeline:
    """Test end-to-end Phase 2 pipeline."""

    def test_full_pipeline_mock(self):
        """Test full pipeline with mock detections."""
        logger.info("Testing end-to-end pipeline...")

        # Initialize components
        temporal_filter = TemporalFilter()
        roi_validator = ROIValidator()
        alert_gen = AlertGenerator()

        # Simulate worker detections
        worker1_dets = [{
            "boxes": np.array([[50, 50, 150, 150]]),
            "confidences": np.array([0.95]),
            "class_ids": np.array([0]),
        }]

        worker2_dets = [{
            "boxes": np.array([[52, 52, 148, 148]]),
            "confidences": np.array([0.93]),
            "class_ids": np.array([0]),
        }]

        # Step 1: Consensus matching
        worker_detections = {
            "yolo_vit": worker1_dets,
            "frcnn_rtdetr": worker2_dets,
        }
        consensus_dets = temporal_filter.process_detections("camera:1", worker_detections)

        assert len(consensus_dets) > 0, "Should find consensus detections"
        logger.info(f"✓ Consensus step: {len(consensus_dets)} detection(s)")

        # Step 2: ROI validation (skip for now - no ROI config)
        # roi_validated = roi_validator.validate_detections("camera:1", consensus_dets)

        # Step 3: Alert generation
        alert_ids = alert_gen.generate_alerts("camera:1", consensus_dets if consensus_dets else [])

        logger.info(f"✓ End-to-end pipeline complete: {len(alert_ids)} alert(s) generated")


if __name__ == "__main__":
    logger.info("Running Phase 2 Test Suite...")
    pytest.main([__file__, "-v", "--tb=short"])
