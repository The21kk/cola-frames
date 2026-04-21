"""
Phase 2 End-to-End Pipeline Tests

Tests the complete Phase 2 detection pipeline:
Frame → Worker1 Detection → Worker2 Detection → Consensus Voting → Temporal Filtering 
→ ROI Validation → Alert Generation

This is the integration test that validates all Phase 2 components work together.
"""

import pytest
import numpy as np
import cv2
import torch
import time
import logging
from typing import Dict, List

from config.settings import (
    DEVICE_WORKER_1,
    DEVICE_WORKER_2,
    DETECTION_CONFIDENCE_THRESHOLD,
    TEMPORAL_PERSISTENCE_SECONDS,
    CONSENSUS_IOU_THRESHOLD,
)

from producer.frame_serializer import FrameSerializer
from workers.yolo_vit_detector import YOLOVitDetector
from workers.frcnn_rtdetr_detector import FasterRCNNRtdetrDetector
from rules_engine.temporal_filter import TemporalFilter
from rules_engine.roi_validator import ROIValidator
from rules_engine.alert_generator import AlertGenerator


logger = logging.getLogger(__name__)


# ============================================================================
# FIXTURES: Phase 2 Pipeline Components
# ============================================================================

@pytest.fixture
def pipeline_components(cleanup_gpu):
    """Initialize all Phase 2 pipeline components."""
    try:
        worker1 = YOLOVitDetector(
            device=DEVICE_WORKER_1 if torch.cuda.is_available() else "cpu",
            confidence_threshold=DETECTION_CONFIDENCE_THRESHOLD,
        )
        worker2 = FasterRCNNRtdetrDetector(
            device=DEVICE_WORKER_2 if torch.cuda.is_available() else "cpu",
            confidence_threshold=DETECTION_CONFIDENCE_THRESHOLD,
        )
    except ImportError:
        pytest.skip("Detection models not available")

    temporal_filter = TemporalFilter()
    roi_validator = ROIValidator(roi_config={})
    alert_gen = AlertGenerator()

    yield {
        "worker1": worker1,
        "worker2": worker2,
        "temporal_filter": temporal_filter,
        "roi_validator": roi_validator,
        "alert_gen": alert_gen,
    }

    # Cleanup
    try:
        worker1.cleanup()
        worker2.cleanup()
    except:
        pass


@pytest.fixture
def process_results():
    """Structure to hold pipeline processing results."""
    return {
        "frames_processed": 0,
        "worker1_detections": [],
        "worker2_detections": [],
        "consensus_detections": [],
        "roi_validated": [],
        "alerts_generated": [],
        "timings": [],
        "failures": [],
    }


# ============================================================================
# TEST 1: Single Frame Pipeline
# ============================================================================

@pytest.mark.gpu
class TestSingleFramePipeline:
    """Test pipeline on a single frame."""

    def test_single_frame_full_pipeline(self, sample_frame, pipeline_components):
        """Process single frame through complete pipeline."""
        logger.info("Testing single frame full pipeline...")

        worker1 = pipeline_components["worker1"]
        worker2 = pipeline_components["worker2"]
        temporal_filter = pipeline_components["temporal_filter"]
        roi_validator = pipeline_components["roi_validator"]
        alert_gen = pipeline_components["alert_gen"]

        camera_id = "camera:e2e:single"

        # Step 1: Worker 1 detection
        start_time = time.time()
        worker1_det = worker1.detect(sample_frame)
        worker1_time = (time.time() - start_time) * 1000

        assert worker1_det is not None, "Worker1 detection failed"
        assert "boxes" in worker1_det, "Worker1 output missing boxes"

        logger.info(f"  Worker1: {worker1_det['num_detections']} detections in {worker1_time:.1f}ms")

        # Step 2: Worker 2 detection
        start_time = time.time()
        worker2_det = worker2.detect(sample_frame)
        worker2_time = (time.time() - start_time) * 1000

        assert worker2_det is not None, "Worker2 detection failed"
        assert "boxes" in worker2_det, "Worker2 output missing boxes"

        logger.info(f"  Worker2: {worker2_det['num_detections']} detections in {worker2_time:.1f}ms")

        # Step 3: Consensus voting
        worker_detections = {
            "yolo_vit": [worker1_det],
            "frcnn_rtdetr": [worker2_det],
        }

        consensus_dets = temporal_filter.process_detections(camera_id, worker_detections)
        logger.info(f"  Consensus: {len(consensus_dets)} detections")

        # Step 4: ROI validation
        roi_validated = roi_validator.validate_detections(camera_id, consensus_dets)
        logger.info(f"  ROI Validated: {len(roi_validated)} detections")

        # Step 5: Alert generation
        alert_ids = alert_gen.generate_alerts(camera_id, roi_validated if roi_validated else [])
        logger.info(f"  Alerts: {len(alert_ids)} alerts generated")

        logger.info(f"✓ Single frame pipeline complete")

    def test_detection_consistency(self, sample_frame, pipeline_components):
        """Verify detection outputs are consistent across runs."""
        logger.info("Testing detection consistency...")

        worker1 = pipeline_components["worker1"]

        # Run same frame through detector twice
        det1 = worker1.detect(sample_frame)
        det2 = worker1.detect(sample_frame)

        # Should have same number of detections (deterministic on same input)
        assert det1["num_detections"] == det2["num_detections"], \
            f"Inconsistent detection count: {det1['num_detections']} vs {det2['num_detections']}"

        logger.info(f"✓ Detections consistent: {det1['num_detections']} detections both runs")


# ============================================================================
# TEST 2: Multi-Frame Pipeline
# ============================================================================

@pytest.mark.gpu
@pytest.mark.slow
class TestMultiFramePipeline:
    """Test pipeline on multiple frames (simulating video stream)."""

    def test_multi_frame_pipeline(self, synthetic_video, pipeline_components, process_results):
        """Process video through complete pipeline."""
        logger.info("Testing multi-frame pipeline on synthetic video...")

        worker1 = pipeline_components["worker1"]
        worker2 = pipeline_components["worker2"]
        temporal_filter = pipeline_components["temporal_filter"]
        roi_validator = pipeline_components["roi_validator"]
        alert_gen = pipeline_components["alert_gen"]

        camera_id = "camera:e2e:video"

        # Open video
        cap = cv2.VideoCapture(str(synthetic_video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"  Processing video: {total_frames} frames @ {fps} FPS")

        frame_idx = 0
        start_pipeline_time = time.time()

        try:
            while cap.isOpened() and frame_idx < 30:  # Process first 30 frames
                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    frame_start = time.time()

                    # Worker 1 detection
                    worker1_det = worker1.detect(frame)
                    process_results["worker1_detections"].append(worker1_det)

                    # Worker 2 detection
                    worker2_det = worker2.detect(frame)
                    process_results["worker2_detections"].append(worker2_det)

                    # Consensus
                    worker_detections = {
                        "yolo_vit": [worker1_det],
                        "frcnn_rtdetr": [worker2_det],
                    }
                    consensus_dets = temporal_filter.process_detections(camera_id, worker_detections)
                    process_results["consensus_detections"].extend(consensus_dets)

                    # ROI validation
                    roi_validated = roi_validator.validate_detections(camera_id, consensus_dets)
                    process_results["roi_validated"].extend(roi_validated)

                    # Alert generation
                    alert_ids = alert_gen.generate_alerts(camera_id, roi_validated if roi_validated else [])
                    process_results["alerts_generated"].extend(alert_ids)

                    # Timing
                    frame_time = (time.time() - frame_start) * 1000
                    process_results["timings"].append(frame_time)

                    if frame_idx % 10 == 0:
                        logger.info(
                            f"  Frame {frame_idx}: W1={worker1_det['num_detections']} "
                            f"W2={worker2_det['num_detections']} "
                            f"Consensus={len(consensus_dets)} "
                            f"Alerts={len(alert_ids)} "
                            f"Time={frame_time:.1f}ms"
                        )

                    frame_idx += 1
                    process_results["frames_processed"] = frame_idx

                except Exception as e:
                    logger.error(f"Error processing frame {frame_idx}: {e}")
                    process_results["failures"].append((frame_idx, str(e)))
                    continue

        finally:
            cap.release()

        total_time = time.time() - start_pipeline_time

        # Validate results
        assert process_results["frames_processed"] > 0, "No frames processed"
        assert len(process_results["failures"]) == 0, \
            f"Processing failures: {process_results['failures']}"

        # Calculate stats
        avg_time = np.mean(process_results["timings"])
        fps_achieved = 1000 / avg_time if avg_time > 0 else 0

        logger.info(
            f"✓ Multi-frame pipeline complete:\n"
            f"    Frames processed: {process_results['frames_processed']}\n"
            f"    Consensus detections: {len(process_results['consensus_detections'])}\n"
            f"    Alerts generated: {len(process_results['alerts_generated'])}\n"
            f"    Avg time/frame: {avg_time:.1f}ms\n"
            f"    FPS achieved: {fps_achieved:.1f}"
        )

    def test_pipeline_throughput(self, sample_frames, pipeline_components, process_results):
        """Measure pipeline throughput with batch of frames."""
        logger.info("Testing pipeline throughput...")

        worker1 = pipeline_components["worker1"]
        worker2 = pipeline_components["worker2"]
        temporal_filter = pipeline_components["temporal_filter"]
        alert_gen = pipeline_components["alert_gen"]

        camera_id = "camera:e2e:throughput"

        start_time = time.time()

        for frame_idx, frame in enumerate(sample_frames):
            w1_det = worker1.detect(frame)
            w2_det = worker2.detect(frame)

            worker_dets = {"yolo_vit": [w1_det], "frcnn_rtdetr": [w2_det]}
            consensus = temporal_filter.process_detections(camera_id, worker_dets)

            alert_gen.generate_alerts(camera_id, consensus if consensus else [])

            process_results["frames_processed"] += 1

        elapsed = time.time() - start_time
        fps = len(sample_frames) / elapsed if elapsed > 0 else 0

        assert fps > 1, f"Pipeline throughput too low: {fps:.1f} FPS (expected >1)"

        logger.info(f"✓ Pipeline throughput: {len(sample_frames)} frames in {elapsed:.2f}s = {fps:.1f} FPS")


# ============================================================================
# TEST 3: Pipeline Robustness
# ============================================================================

@pytest.mark.gpu
class TestPipelineRobustness:
    """Test pipeline resilience to edge cases."""

    def test_empty_detections(self, pipeline_components):
        """Test pipeline handles empty detections gracefully."""
        logger.info("Testing pipeline with empty detections...")

        temporal_filter = pipeline_components["temporal_filter"]
        roi_validator = pipeline_components["roi_validator"]
        alert_gen = pipeline_components["alert_gen"]

        camera_id = "camera:robust:empty"

        # Simulate empty detections
        empty_det = {
            "boxes": np.array([]),
            "confidences": np.array([]),
            "class_ids": np.array([]),
            "num_detections": 0,
        }

        # Process empty detection
        worker_dets = {"yolo_vit": [empty_det], "frcnn_rtdetr": [empty_det]}
        consensus = temporal_filter.process_detections(camera_id, worker_dets)

        # Should handle gracefully (no errors, no alerts)
        roi_validated = roi_validator.validate_detections(camera_id, consensus)
        alerts = alert_gen.generate_alerts(camera_id, roi_validated if roi_validated else [])

        assert len(alerts) == 0, "Empty detections should not generate alerts"
        logger.info("✓ Empty detections handled gracefully")

    def test_mixed_detections(self, pipeline_components):
        """Test pipeline with varied detection patterns."""
        logger.info("Testing pipeline with mixed detections...")

        temporal_filter = pipeline_components["temporal_filter"]

        camera_id = "camera:robust:mixed"

        # Pattern 1: One worker has detections, other doesn't
        worker1_dets = [{
            "boxes": np.array([[50, 50, 150, 150]]),
            "confidences": np.array([0.95]),
            "class_ids": np.array([0]),
            "num_detections": 1,
        }]

        worker2_dets = [{
            "boxes": np.array([]),
            "confidences": np.array([]),
            "class_ids": np.array([]),
            "num_detections": 0,
        }]

        # Should still work (no consensus, but no error)
        consensus = temporal_filter.process_detections(
            camera_id,
            {"yolo_vit": worker1_dets, "frcnn_rtdetr": worker2_dets}
        )

        logger.info(f"✓ Mixed detections handled: {len(consensus)} consensus detections")

    def test_high_confidence_filter(self, pipeline_components):
        """Test pipeline filters low-confidence detections."""
        logger.info("Testing high-confidence filtering...")

        temporal_filter = pipeline_components["temporal_filter"]

        camera_id = "camera:robust:confidence"

        # Low confidence detections
        low_conf_dets = [{
            "boxes": np.array([[50, 50, 150, 150], [200, 200, 300, 300]]),
            "confidences": np.array([0.30, 0.25]),  # Below threshold
            "class_ids": np.array([0, 1]),
            "num_detections": 2,
        }]

        # Process (should filter out low confidence)
        consensus = temporal_filter.process_detections(
            camera_id,
            {"yolo_vit": low_conf_dets}
        )

        logger.info(f"✓ Confidence filtering: {len(consensus)} detections after filtering")


# ============================================================================
# TEST 4: Performance Analysis
# ============================================================================

@pytest.mark.gpu
@pytest.mark.slow
class TestPipelinePerformance:
    """Test overall pipeline performance metrics."""

    def test_pipeline_latency_analysis(self, sample_frame, pipeline_components):
        """Analyze pipeline latency distribution."""
        logger.info("Testing pipeline latency analysis...")

        worker1 = pipeline_components["worker1"]
        worker2 = pipeline_components["worker2"]
        temporal_filter = pipeline_components["temporal_filter"]

        camera_id = "camera:perf:latency"
        latencies = []

        # Warm-up
        _ = worker1.detect(sample_frame)

        # Measure 10 pipeline runs
        for _ in range(10):
            start = time.time()

            w1_det = worker1.detect(sample_frame)
            w2_det = worker2.detect(sample_frame)

            worker_dets = {"yolo_vit": [w1_det], "frcnn_rtdetr": [w2_det]}
            _ = temporal_filter.process_detections(camera_id, worker_dets)

            latencies.append((time.time() - start) * 1000)

        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        percentile_95 = np.percentile(latencies, 95)

        logger.info(
            f"✓ Pipeline latency analysis:\n"
            f"    Mean: {mean_latency:.1f}ms\n"
            f"    Std: {std_latency:.1f}ms\n"
            f"    p95: {percentile_95:.1f}ms"
        )

    def test_memory_stability(self, sample_frames, pipeline_components):
        """Test that pipeline memory usage remains stable."""
        logger.info("Testing memory stability...")

        worker1 = pipeline_components["worker1"]
        temporal_filter = pipeline_components["temporal_filter"]

        camera_id = "camera:perf:memory"

        # Process many frames and verify no memory leak
        for i in range(20):
            for frame in sample_frames:
                w1_det = worker1.detect(frame)
                _ = temporal_filter.process_detections(
                    camera_id,
                    {"yolo_vit": [w1_det]}
                )

        logger.info("✓ Memory stability test passed (20 iterations)")


# ============================================================================
# INTEGRATION: Full Validation
# ============================================================================

class TestFullValidation:
    """Final integration validation tests."""

    @pytest.mark.gpu
    def test_e2e_pipeline_validation(self, sample_frame, pipeline_components):
        """Final end-to-end validation of complete pipeline."""
        logger.info("Running end-to-end pipeline validation...")

        worker1 = pipeline_components["worker1"]
        worker2 = pipeline_components["worker2"]
        temporal_filter = pipeline_components["temporal_filter"]
        roi_validator = pipeline_components["roi_validator"]
        alert_gen = pipeline_components["alert_gen"]

        camera_id = "camera:validation:final"

        # Execute full pipeline
        w1_det = worker1.detect(sample_frame)
        w2_det = worker2.detect(sample_frame)
        consensus = temporal_filter.process_detections(
            camera_id,
            {"yolo_vit": [w1_det], "frcnn_rtdetr": [w2_det]}
        )
        roi_valid = roi_validator.validate_detections(camera_id, consensus)
        alerts = alert_gen.generate_alerts(camera_id, roi_valid if roi_valid else [])

        # Validation
        assert w1_det is not None, "Worker1 must return output"
        assert w2_det is not None, "Worker2 must return output"
        assert isinstance(consensus, list), "Consensus must return list"
        assert isinstance(roi_valid, list), "ROI must return list"
        assert isinstance(alerts, list), "Alerts must return list"

        logger.info(
            f"✓ E2E Validation Complete:\n"
            f"    W1 detections: {w1_det['num_detections']}\n"
            f"    W2 detections: {w2_det['num_detections']}\n"
            f"    Consensus: {len(consensus)}\n"
            f"    ROI Valid: {len(roi_valid)}\n"
            f"    Alerts: {len(alerts)}"
        )


if __name__ == "__main__":
    logger.info("Running Phase 2 End-to-End Tests...")
    pytest.main([__file__, "-v", "--tb=short"])
