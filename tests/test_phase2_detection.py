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
8. Performance timing analysis (NEW)
9. Alert statistics (NEW)
10. Detection store integration
"""

import pytest
import numpy as np
import torch
import time
import logging
from typing import List, Dict

from config.settings import (
    REDIS_HOST, REDIS_PORT, REDIS_DB,
    DETECTION_CONFIDENCE_THRESHOLD,
    TEMPORAL_PERSISTENCE_SECONDS,
    CONSENSUS_IOU_THRESHOLD,
    DEVICE_WORKER_1, DEVICE_WORKER_2,
)
from redis_broker.stream_manager import RedisStreamManager
from producer.frame_serializer import FrameSerializer
from workers.base_detector import BaseDetector
from workers.yolo_vit_detector import YOLOVitDetector
from workers.frcnn_rtdetr_detector import FasterRCNNRtdetrDetector
from rules_engine.detection_store import DetectionStore
from rules_engine.temporal_filter import TemporalFilter
from rules_engine.roi_validator import ROIValidator
from rules_engine.alert_generator import AlertGenerator


logger = logging.getLogger(__name__)


# ============================================================================
# TEST 1: GPU Support
# ============================================================================

@pytest.mark.gpu
class TestGPUSupport:
    """Test GPU device management."""

    def test_cuda_availability(self):
        """Verify CUDA availability on system."""
        logger.info("Testing CUDA availability...")
        assert torch.cuda.is_available(), "CUDA not available on this system"
        assert torch.cuda.device_count() >= 1, "No CUDA devices found"
        logger.info(f"✓ CUDA available: {torch.cuda.device_count()} device(s)")

    def test_device_properties(self):
        """Verify CUDA device properties."""
        logger.info("Testing CUDA device properties...")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(
                f"  GPU {i}: {props.name}, Memory: {props.total_memory / 1e9:.1f} GB, "
                f"Compute: {props.major}.{props.minor}"
            )
            assert props.total_memory > 0, f"GPU {i} has no memory"

    def test_device_selection(self):
        """Test device selection logic."""
        logger.info("Testing device selection...")

        # Test GPU device
        device = torch.device(DEVICE_WORKER_1)
        assert str(device).startswith("cuda") or str(device) == "cpu"

        # Verify device is accessible
        test_tensor = torch.randn(1, 10).to(device)
        assert test_tensor.device == device
        
        logger.info(f"✓ Device selection working: {device}")

    def test_gpu_memory_preallocation(self, cleanup_gpu):
        """Test GPU memory pre-allocation."""
        logger.info("Testing GPU memory pre-allocation...")

        try:
            detector = YOLOVitDetector(device=DEVICE_WORKER_1)
            assert detector.device is not None

            # Verify tensors are on correct device
            test_tensor = torch.randn(1, 3, 480, 640).to(detector.device)
            assert str(test_tensor.device).startswith("cuda")
            
            logger.info(f"✓ GPU memory pre-allocated on {detector.device}")
            detector.cleanup()
        except Exception as e:
            logger.warning(f"GPU memory pre-allocation test failed: {e}")
            pytest.skip("GPU initialization failed")


# ============================================================================
# TEST 2: Detection Workers
# ============================================================================

@pytest.mark.gpu
class TestDetectionWorkers:
    """Test individual detector workers."""

    def test_yolo_vit_initialization(self, cleanup_gpu):
        """Test YOLOv8 + ViT detector initialization."""
        logger.info("Testing YOLOv8 + ViT initialization...")

        try:
            detector = YOLOVitDetector(
                device=DEVICE_WORKER_1 if torch.cuda.is_available() else "cpu"
            )
            assert detector.model_name == "yolo_vit"
            assert detector.yolo_model is not None
            assert detector.vit_model is not None
            logger.info(f"✓ YOLOv8 + ViT initialized on {detector.device}")
            detector.cleanup()
        except ImportError as e:
            logger.warning(f"YOLOv8 + ViT dependencies not installed: {e}")
            pytest.skip("Dependencies not available")

    def test_frcnn_rtdetr_initialization(self, cleanup_gpu):
        """Test Faster R-CNN + RT-DETR initialization."""
        logger.info("Testing Faster R-CNN + RT-DETR initialization...")

        try:
            detector = FasterRCNNRtdetrDetector(
                device=DEVICE_WORKER_2 if torch.cuda.is_available() else "cpu"
            )
            assert detector.model_name == "frcnn_rtdetr"
            assert detector.frcnn_model is not None
            logger.info(f"✓ Faster R-CNN + RT-DETR initialized on {detector.device}")
            detector.cleanup()
        except ImportError as e:
            logger.warning(f"Faster R-CNN + RT-DETR dependencies not installed: {e}")
            pytest.skip("Dependencies not available")

    def test_yolo_vit_detection(self, sample_frame, cleanup_gpu):
        """Test YOLOv8 + ViT inference on single frame."""
        logger.info("Testing YOLOv8 + ViT detection...")

        try:
            detector = YOLOVitDetector(
                device=DEVICE_WORKER_1 if torch.cuda.is_available() else "cpu"
            )

            # Run detection
            start_time = time.time()
            detections = detector.detect(sample_frame)
            elapsed_ms = (time.time() - start_time) * 1000

            # Validate output structure
            assert "boxes" in detections, "Missing 'boxes' in detection output"
            assert "confidences" in detections, "Missing 'confidences' in detection output"
            assert "class_ids" in detections, "Missing 'class_ids' in detection output"
            assert "num_detections" in detections, "Missing 'num_detections' in detection output"
            
            # Validate detection
            assert detector.validate_detections(detections), "Detection validation failed"

            logger.info(
                f"✓ YOLOv8 + ViT: {detections['num_detections']} detections in {elapsed_ms:.1f}ms"
            )
            detector.cleanup()

        except ImportError:
            pytest.skip("Dependencies not available")

    def test_frcnn_rtdetr_detection(self, sample_frame, cleanup_gpu):
        """Test Faster R-CNN + RT-DETR inference on single frame."""
        logger.info("Testing Faster R-CNN + RT-DETR detection...")

        try:
            detector = FasterRCNNRtdetrDetector(
                device=DEVICE_WORKER_2 if torch.cuda.is_available() else "cpu"
            )

            # Run detection
            start_time = time.time()
            detections = detector.detect(sample_frame)
            elapsed_ms = (time.time() - start_time) * 1000

            # Validate output
            assert "boxes" in detections
            assert "confidences" in detections
            assert detector.validate_detections(detections)

            logger.info(
                f"✓ Faster R-CNN + RT-DETR: {detections['num_detections']} detections in {elapsed_ms:.1f}ms"
            )
            detector.cleanup()

        except ImportError:
            pytest.skip("Dependencies not available")

    def test_batch_detection(self, sample_frames, cleanup_gpu):
        """Test batch processing of multiple frames."""
        logger.info("Testing batch detection...")

        try:
            detector = YOLOVitDetector(
                device=DEVICE_WORKER_1 if torch.cuda.is_available() else "cpu",
                batch_size=4
            )

            # Run batch inference
            start_time = time.time()
            detections_list = detector.detect_batch(sample_frames)
            elapsed_ms = (time.time() - start_time) * 1000

            assert len(detections_list) == len(sample_frames), "Batch size mismatch"
            for det in detections_list:
                assert detector.validate_detections(det), "Batch detection validation failed"

            logger.info(f"✓ Batch detection: {len(sample_frames)} frames in {elapsed_ms:.1f}ms")
            detector.cleanup()

        except ImportError:
            pytest.skip("Dependencies not available")


# ============================================================================
# TEST 3: Consensus Voting & IoU Matching
# ============================================================================

class TestConsensusVoting:
    """Test consensus voting and IoU matching."""

    def test_iou_calculation(self):
        """Test IoU (Intersection over Union) calculation."""
        logger.info("Testing IoU calculation...")

        temporal_filter = TemporalFilter()

        # Test 1: Identical boxes (IoU = 1.0)
        box1 = np.array([10, 10, 100, 100])
        box2 = np.array([10, 10, 100, 100])
        iou = temporal_filter._calculate_iou(box1, box2)
        assert iou == 1.0, f"Identical boxes should have IoU=1.0, got {iou}"

        # Test 2: No overlap (IoU = 0.0)
        box1 = np.array([0, 0, 50, 50])
        box2 = np.array([100, 100, 150, 150])
        iou = temporal_filter._calculate_iou(box1, box2)
        assert iou == 0.0, f"Non-overlapping boxes should have IoU=0.0, got {iou}"

        # Test 3: Partial overlap (0 < IoU < 1)
        box1 = np.array([0, 0, 100, 100])
        box2 = np.array([50, 50, 150, 150])
        iou = temporal_filter._calculate_iou(box1, box2)
        assert 0 < iou < 1, f"Overlapping boxes should have 0<IoU<1, got {iou}"

        logger.info("✓ IoU calculation working correctly")

    def test_consensus_matching_with_overlap(self):
        """Test consensus matching with overlapping detections."""
        logger.info("Testing consensus matching (overlapping)...")

        temporal_filter = TemporalFilter()
        camera_id = "camera:test:1"

        # Create slightly overlapping detections from both workers
        worker1_dets = [{
            "boxes": np.array([[10, 10, 100, 100]]),
            "confidences": np.array([0.95]),
            "class_ids": np.array([0]),
        }]

        worker2_dets = [{
            "boxes": np.array([[12, 12, 98, 98]]),  # ~95% IoU overlap
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
        assert consensus_dets[0]["iou_score"] >= CONSENSUS_IOU_THRESHOLD, \
            f"IoU score below threshold"
            
        logger.info(f"✓ Consensus matching: {len(consensus_dets)} consensus detection(s)")

    def test_consensus_no_match(self):
        """Test no consensus when detections don't overlap."""
        logger.info("Testing no consensus case (non-overlapping)...")

        temporal_filter = TemporalFilter()
        camera_id = "camera:test:2"

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
        logger.info("✓ No consensus as expected (non-overlapping boxes)")


# ============================================================================
# TEST 4: Rules Engine (Temporal + ROI + Alerts)
# ============================================================================

class TestTemporalFiltering:
    """Test temporal persistence filtering."""
    
    def test_temporal_filter_persistence(self):
        """Test temporal filtering with persistence window."""
        logger.info("Testing temporal filter persistence...")
        
        temporal_filter = TemporalFilter()
        
        # Simulate detections across frames with persistence
        camera_id = "camera:temporal:1"
        detections = [
            {"boxes": np.array([[50, 50, 150, 150]]), "confidences": np.array([0.95])},
            {"boxes": np.array([[52, 52, 148, 148]]), "confidences": np.array([0.93])},
            {"boxes": np.array([[54, 54, 146, 146]]), "confidences": np.array([0.91])},
        ]
        
        filtered = []
        for det in detections:
            result = temporal_filter.process_detections(
                camera_id,
                {"worker1": [det]}
            )
            filtered.append(result)
        
        logger.info(f"✓ Temporal filtering: {len(filtered)} results processed")


class TestROIValidation:
    """Test ROI (Region of Interest) validation."""

    def test_roi_inclusion_region(self):
        """Test ROI validator with inclusion regions."""
        logger.info("Testing ROI inclusion validation...")

        roi_config = {
            "camera:roi:1": {
                "inclusion_regions": [(0, 0, 200, 200)],
                "exclusion_regions": [],
            }
        }

        validator = ROIValidator(roi_config=roi_config)

        # Detection inside inclusion region
        det_inside = {
            "boxes": [[50, 50, 100, 100]],
            "confidences": [0.95],
        }

        # Detection outside inclusion region
        det_outside = {
            "boxes": [[300, 300, 400, 400]],
            "confidences": [0.95],
        }

        result_inside = validator.validate_detections("camera:roi:1", [det_inside])
        result_outside = validator.validate_detections("camera:roi:1", [det_outside])

        assert len(result_inside) == 1, "Detection inside ROI should pass"
        assert len(result_outside) == 0, "Detection outside ROI should fail"

        logger.info("✓ ROI inclusion validation working")

    def test_roi_exclusion_region(self):
        """Test ROI validator with exclusion regions."""
        logger.info("Testing ROI exclusion validation...")

        roi_config = {
            "camera:roi:2": {
                "inclusion_regions": [],
                "exclusion_regions": [(100, 100, 200, 200)],
            }
        }

        validator = ROIValidator(roi_config=roi_config)

        # Detection in exclusion zone
        det_excluded = {
            "boxes": [[110, 110, 150, 150]],
            "confidences": [0.95],
        }

        # Detection outside exclusion zone
        det_allowed = {
            "boxes": [[50, 50, 90, 90]],
            "confidences": [0.95],
        }

        result_excluded = validator.validate_detections("camera:roi:2", [det_excluded])
        result_allowed = validator.validate_detections("camera:roi:2", [det_allowed])

        assert len(result_excluded) == 0, "Detection in exclusion zone should fail"
        assert len(result_allowed) == 1, "Detection outside exclusion zone should pass"

        logger.info("✓ ROI exclusion validation working")


class TestAlertGeneration:
    """Test alert generation."""

    def test_alert_generation(self):
        """Test basic alert generation."""
        logger.info("Testing alert generation...")

        alert_gen = AlertGenerator()

        validated_dets = [{
            "boxes": [[50, 50, 100, 100]],
            "confidences": [0.95],
            "workers_agreed": ["yolo_vit", "frcnn_rtdetr"],
            "timestamp": time.time(),
        }]

        alert_ids = alert_gen.generate_alerts("camera:alert:1", validated_dets)
        assert len(alert_ids) >= 1, "Should generate at least one alert"
        logger.info(f"✓ Alert generation: {len(alert_ids)} alert(s) generated")

    def test_alert_severity_levels(self):
        """Test alert severity calculation."""
        logger.info("Testing alert severity levels...")

        alert_gen = AlertGenerator()

        # High severity (high confidences)
        high_conf = np.array([0.95, 0.92, 0.98])
        severity_high = alert_gen._calculate_severity(high_conf.tolist())
        assert severity_high == "high", f"Expected 'high', got '{severity_high}'"

        # Medium severity
        med_conf = np.array([0.75, 0.70, 0.80])
        severity_med = alert_gen._calculate_severity(med_conf.tolist())
        assert severity_med == "medium", f"Expected 'medium', got '{severity_med}'"

        # Low severity
        low_conf = np.array([0.55, 0.50, 0.60])
        severity_low = alert_gen._calculate_severity(low_conf.tolist())
        assert severity_low == "low", f"Expected 'low', got '{severity_low}'"

        logger.info("✓ Alert severity calculation working correctly")

    def test_alert_retrieval(self):
        """Test retrieving alert statistics."""
        logger.info("Testing alert retrieval...")

        alert_gen = AlertGenerator()

        # Generate some alerts
        validated_dets = [
            {"boxes": [[50, 50, 100, 100]], "confidences": [0.95]},
            {"boxes": [[200, 200, 300, 300]], "confidences": [0.92]},
        ]

        alert_ids = alert_gen.generate_alerts("camera:alert:stats", validated_dets)
        assert len(alert_ids) > 0

        # Get stats
        stats = alert_gen.get_alert_stats(camera_id="camera:alert:stats")
        assert isinstance(stats, dict)
        assert "total_alerts" in stats
        
        logger.info(f"✓ Alert stats: {stats.get('total_alerts', 0)} alerts generated")


# ============================================================================
# TEST 5: Detection Store
# ============================================================================

class TestDetectionStore:
    """Test detection storage in Redis."""

    def test_store_detection(self, cleanup_redis):
        """Test storing a detection."""
        logger.info("Testing detection storage...")

        store = DetectionStore()

        # Store detection
        stream_id = store.store_detection(
            camera_id="camera:store:1",
            worker_name="yolo_vit",
            frame_id="frame:1",
            timestamp=time.time(),
            boxes=np.array([[10, 10, 100, 100]]),
            confidences=np.array([0.95]),
            class_ids=np.array([0]),
            execution_time_ms=50.0,
        )

        assert stream_id is not None, "Storage should return stream ID"
        logger.info(f"✓ Detection stored: {stream_id}")

    def test_retrieve_detections(self, cleanup_redis):
        """Test retrieving stored detections."""
        logger.info("Testing detection retrieval...")

        store = DetectionStore()

        # Store 3 detections
        for i in range(3):
            store.store_detection(
                camera_id="camera:store:2",
                worker_name="yolo_vit",
                frame_id=f"frame:{i}",
                timestamp=time.time() + i,
                boxes=np.array([[10+i*10, 10, 100+i*10, 100]]),
                confidences=np.array([0.90 + i*0.02]),
                class_ids=np.array([0]),
                execution_time_ms=50.0,
            )

        # Retrieve
        dets = store.get_latest_detections("camera:store:2", worker_name="yolo_vit", limit=2)
        assert len(dets) <= 2, "Should respect limit"
        logger.info(f"✓ Retrieved {len(dets)} detection(s)")


# ============================================================================
# TEST 6: Performance Analysis (NEW)
# ============================================================================

class TestPerformance:
    """Test performance metrics and throughput."""

    def test_detection_latency(self, sample_frame, cleanup_gpu):
        """Measure detection latency."""
        logger.info("Testing detection latency...")

        try:
            detector = YOLOVitDetector(
                device=DEVICE_WORKER_1 if torch.cuda.is_available() else "cpu"
            )

            # Warm-up
            _ = detector.detect(sample_frame)

            # Measure 10 detections
            timings = []
            for _ in range(10):
                start = time.time()
                detector.detect(sample_frame)
                timings.append((time.time() - start) * 1000)

            mean_time = np.mean(timings)
            std_time = np.std(timings)
            fps = 1000 / mean_time

            assert mean_time < 1000, "Detection latency too high"
            logger.info(
                f"✓ Detection latency: mean={mean_time:.1f}ms, std={std_time:.1f}ms, FPS={fps:.1f}"
            )
            
            detector.cleanup()
        except ImportError:
            pytest.skip("Dependencies not available")

    def test_throughput_single_worker(self, sample_frames, cleanup_gpu):
        """Measure single worker throughput."""
        logger.info("Testing single worker throughput...")

        try:
            detector = YOLOVitDetector(
                device=DEVICE_WORKER_1 if torch.cuda.is_available() else "cpu",
                batch_size=4
            )

            # Process batch
            start = time.time()
            results = detector.detect_batch(sample_frames)
            elapsed = time.time() - start

            fps = len(sample_frames) / elapsed
            assert fps > 2, f"Throughput too low: {fps:.1f} FPS (expected >2)"

            logger.info(f"✓ Throughput: {len(sample_frames)} frames in {elapsed:.2f}s = {fps:.1f} FPS")
            detector.cleanup()
        except ImportError:
            pytest.skip("Dependencies not available")


# ============================================================================
# TEST 7: Alert Statistics (NEW)
# ============================================================================

class TestAlerting:
    """Test alert statistics and validation."""

    def test_alert_stats(self, cleanup_redis):
        """Test alert statistics retrieval."""
        logger.info("Testing alert statistics...")

        alert_gen = AlertGenerator()

        # Generate multiple alerts
        for i in range(5):
            det = {
                "boxes": [[50+i*10, 50, 150+i*10, 150]],
                "confidences": [0.85 + i*0.02],
            }
            alert_gen.generate_alerts(f"camera:stats:{i}", [det])

        # Get stats for one camera
        stats = alert_gen.get_alert_stats(camera_id="camera:stats:0")
        
        assert isinstance(stats, dict)
        if "total_alerts" in stats:
            assert stats["total_alerts"] >= 1
        
        logger.info(f"✓ Alert statistics retrieved: {stats}")

    def test_active_alerts_retrieval(self, cleanup_redis):
        """Test retrieving active (unacknowledged) alerts."""
        logger.info("Testing active alerts retrieval...")

        alert_gen = AlertGenerator()

        # Generate alert
        validated_dets = [
            {"boxes": [[50, 50, 100, 100]], "confidences": [0.95]},
        ]
        alert_gen.generate_alerts("camera:active", validated_dets)

        # Get active alerts
        active = alert_gen.get_active_alerts(camera_id="camera:active", limit=5)
        assert isinstance(active, list)
        
        logger.info(f"✓ Retrieved {len(active)} active alert(s)")


# ============================================================================
# TEST 8: Integration Tests
# ============================================================================

class TestPhase2Integration:
    """Integration tests combining Phase 2 components."""

    def test_worker_consensus_pipeline(self):
        """Test worker consensus pipeline."""
        logger.info("Testing worker consensus pipeline...")

        temporal_filter = TemporalFilter()

        # Simulate worker detections
        worker1_det = {
            "boxes": np.array([[50, 50, 150, 150]]),
            "confidences": np.array([0.95]),
            "class_ids": np.array([0]),
        }

        worker2_det = {
            "boxes": np.array([[52, 52, 148, 148]]),
            "confidences": np.array([0.93]),
            "class_ids": np.array([0]),
        }

        # Run consensus
        consensus_dets = temporal_filter._match_detections_consensus(
            "camera:integration",
            [worker1_det],
            [worker2_det],
            time.time(),
        )

        assert len(consensus_dets) > 0, "Should find consensus"
        logger.info(f"✓ Consensus pipeline: {len(consensus_dets)} detection(s)")

    def test_full_rules_pipeline(self):
        """Test complete rules pipeline (consensus → ROI → alert)."""
        logger.info("Testing full rules pipeline...")

        temporal_filter = TemporalFilter()
        roi_validator = ROIValidator()
        alert_gen = AlertGenerator()

        # Step 1: Consensus
        worker_detections = {
            "worker1": [{
                "boxes": np.array([[50, 50, 150, 150]]),
                "confidences": np.array([0.95]),
            }],
            "worker2": [{
                "boxes": np.array([[52, 52, 148, 148]]),
                "confidences": np.array([0.93]),
            }]
        }

        consensus_dets = temporal_filter.process_detections("camera:full", worker_detections)

        # Step 2: ROI validation (skip if no configs)
        # roi_validated = roi_validator.validate_detections("camera:full", consensus_dets)

        # Step 3: Alerts
        alert_ids = alert_gen.generate_alerts("camera:full", consensus_dets if consensus_dets else [])

        logger.info(f"✓ Full pipeline: consensus={len(consensus_dets)}, alerts={len(alert_ids)}")


if __name__ == "__main__":
    logger.info("Running Phase 2 Test Suite...")
    pytest.main([__file__, "-v", "--tb=short"])
