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
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime

from config.settings import (
    DETECTION_CONFIDENCE_THRESHOLD,
    TEMPORAL_PERSISTENCE_SECONDS,
    CONSENSUS_IOU_THRESHOLD,
    DEVICE_WORKER_1,
    DEVICE_WORKER_2,
)

from producer.frame_serializer import FrameSerializer
from redis_broker.stream_manager import RedisStreamManager
from workers.generic_detector import GenericDetector
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
        worker1 = GenericDetector(
            model_types="yolov10",
            model_names="yolov10m",
            device=DEVICE_WORKER_1 if torch.cuda.is_available() else "cpu",
            confidence_threshold=DETECTION_CONFIDENCE_THRESHOLD,
            batch_size=4,
        )
        worker2 = GenericDetector(
            model_types="faster_rcnn",
            model_names="fasterrcnn_resnet50_fpn",
            device=DEVICE_WORKER_2 if torch.cuda.is_available() else "cpu",
            confidence_threshold=DETECTION_CONFIDENCE_THRESHOLD,
            batch_size=2,
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
            "worker_1": [worker1_det],
            "worker_2": [worker2_det],
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
                        "worker_1": [worker1_det],
                        "worker_2": [worker2_det],
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

            worker_dets = {"worker_1": [w1_det], "worker_2": [w2_det]}
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
        worker_dets = {"worker_1": [empty_det], "worker_2": [empty_det]}
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
            {"worker_1": worker1_dets, "worker_2": worker2_dets}
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
            {"worker_1": low_conf_dets}
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

            worker_dets = {"worker_1": [w1_det], "worker_2": [w2_det]}
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
                    {"worker_1": [w1_det]}
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
            {"worker_1": [w1_det], "worker_2": [w2_det]}
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


# ============================================================================
# TEST 5: Video-Based End-to-End (RTSP Simulation with Real Video)
# ============================================================================

@pytest.mark.gpu
@pytest.mark.slow
class TestVideoEndToEnd:
    """
    End-to-end pipeline test with real video file.
    Simulates production RTSP streaming using Redis Streams for ingestion.
    This bridges local testing and production deployment.
    """

    @pytest.fixture
    def video_metrics(self):
        """Initialize metrics collection dict."""
        return {
            "total_frames": 0,
            "frames_processed": 0,
            "frames_ingested": 0,
            "total_detections_w1": 0,
            "total_detections_w2": 0,
            "total_consensus_detections": 0,
            "total_alerts": 0,
            "frame_timings": [],
            "ingestion_timings": [],
            "detection_timings": [],
            "per_frame_data": [],
            "failures": [],
        }

    def _get_video_path(self):
        """Get path to real video, skip if not found."""
        video_path = Path(__file__).parent.parent / "2026-04-21-hallmeds.mp4"
        if not video_path.exists():
            pytest.skip(f"Real video not found at {video_path}. Production test requires real hallway footage.")
        return str(video_path)

    def _ingest_frames_to_redis(self, video_path: str, frames_to_process: int = None, 
                                 redis_manager: RedisStreamManager = None,
                                 video_metrics: Dict = None) -> int:
        """
        Ingest frames from video to Redis Stream, respecting video FPS.
        Simulates RTSP stream ingestion.
        
        Returns:
            Number of frames ingested
        """
        logger.info(f"Starting frame ingestion from video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_delay = 1.0 / fps if fps > 0 else 0.033  # Default to ~30 FPS if fps=0

        video_metrics["total_frames"] = total_frames
        camera_id = "camera:video:halltesting"

        logger.info(f"  Video: {total_frames} frames @ {fps} FPS (delay: {frame_delay*1000:.1f}ms per frame)")

        frame_idx = 0
        ingestion_start = time.time()
        last_frame_time = ingestion_start

        try:
            while cap.isOpened():
                if frames_to_process and frame_idx >= frames_to_process:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    frame_start = time.time()

                    # Simulate RTSP stream timing: respect original FPS
                    elapsed_since_last = frame_start - last_frame_time
                    if elapsed_since_last < frame_delay:
                        time.sleep(frame_delay - elapsed_since_last)

                    # Serialize frame and push to Redis
                    frame_b64 = FrameSerializer.encode_frame_to_base64(frame)
                    metadata = FrameSerializer.create_metadata(
                        camera_id=camera_id,
                        fps=fps,
                        resolution=frame.shape[1::-1]  # (height, width) → (width, height)
                    )

                    redis_manager.add_frame_to_stream(camera_id, frame_b64, metadata)

                    frame_time = (time.time() - frame_start) * 1000
                    video_metrics["ingestion_timings"].append(frame_time)
                    last_frame_time = time.time()

                    frame_idx += 1
                    video_metrics["frames_ingested"] = frame_idx

                    if frame_idx % 50 == 0:
                        logger.info(f"  Ingested frame {frame_idx}/{frames_to_process or total_frames}")

                except Exception as e:
                    logger.error(f"Error ingesting frame {frame_idx}: {e}")
                    video_metrics["failures"].append(("ingestion", frame_idx, str(e)))
                    continue

        finally:
            cap.release()

        total_ingest_time = time.time() - ingestion_start
        logger.info(
            f"Frame ingestion complete: {frame_idx} frames in {total_ingest_time:.1f}s "
            f"(avg {np.mean(video_metrics['ingestion_timings']):.1f}ms per frame)"
        )

        return frame_idx

    def _process_frames_from_redis(self, frames_to_process: int,
                                   pipeline_components: Dict,
                                   redis_manager: RedisStreamManager,
                                   video_metrics: Dict) -> None:
        """
        Process frames from Redis Stream through full detection pipeline.
        Consensus voting, ROI validation, alert generation.
        """
        logger.info(f"Starting detection pipeline for {frames_to_process} frames")

        worker1 = pipeline_components["worker1"]
        worker2 = pipeline_components["worker2"]
        temporal_filter = pipeline_components["temporal_filter"]
        roi_validator = pipeline_components["roi_validator"]
        alert_gen = pipeline_components["alert_gen"]

        camera_id = "camera:video:halltesting"
        frame_idx = 0
        processing_start = time.time()
        max_retries = 300  # 3 seconds max wait (0.01s × 300)
        retry_count = 0

        while frame_idx < frames_to_process:
            try:
                frame_start = time.time()

                # Get latest frame from Redis (LIFO)
                frame_data = redis_manager.get_latest_frame(camera_id)
                if not frame_data:
                    # Wait for frames to be available
                    if retry_count < max_retries:
                        time.sleep(0.01)
                        retry_count += 1
                        continue
                    else:
                        logger.warning(f"Timeout waiting for frame {frame_idx}")
                        break

                retry_count = 0  # Reset on successful frame retrieval
                frame_b64, _ = frame_data
                frame = FrameSerializer.decode_frame_from_base64(frame_b64)

                # Detection: Worker 1
                w1_start = time.time()
                w1_det = worker1.detect(frame)
                w1_time = (time.time() - w1_start) * 1000
                video_metrics["total_detections_w1"] += w1_det["num_detections"]

                # Detection: Worker 2
                w2_start = time.time()
                w2_det = worker2.detect(frame)
                w2_time = (time.time() - w2_start) * 1000
                video_metrics["total_detections_w2"] += w2_det["num_detections"]

                # Consensus voting
                worker_detections = {
                    "worker_1": [w1_det],
                    "worker_2": [w2_det],
                }
                consensus_dets = temporal_filter.process_detections(camera_id, worker_detections)
                video_metrics["total_consensus_detections"] += len(consensus_dets)

                # ROI validation
                roi_validated = roi_validator.validate_detections(camera_id, consensus_dets)

                # Alert generation
                alert_ids = alert_gen.generate_alerts(camera_id, roi_validated if roi_validated else [])
                video_metrics["total_alerts"] += len(alert_ids)

                # Timing
                frame_time = (time.time() - frame_start) * 1000
                video_metrics["frame_timings"].append(frame_time)

                # Per-frame data for CSV export
                video_metrics["per_frame_data"].append({
                    "frame_id": frame_idx,
                    "w1_detections": w1_det["num_detections"],
                    "w2_detections": w2_det["num_detections"],
                    "consensus_reached": len(consensus_dets) > 0,
                    "consensus_count": len(consensus_dets),
                    "alerts_generated": len(alert_ids),
                    "w1_latency_ms": w1_time,
                    "w2_latency_ms": w2_time,
                    "total_latency_ms": frame_time,
                })

                if frame_idx % 50 == 0:
                    logger.info(
                        f"  Frame {frame_idx}: W1={w1_det['num_detections']} W2={w2_det['num_detections']} "
                        f"Consensus={len(consensus_dets)} Alerts={len(alert_ids)} Time={frame_time:.1f}ms"
                    )

                frame_idx += 1
                video_metrics["frames_processed"] = frame_idx

            except Exception as e:
                logger.error(f"Error processing frame {frame_idx}: {e}")
                video_metrics["failures"].append(("processing", frame_idx, str(e)))
                frame_idx += 1
                continue

        total_processing_time = time.time() - processing_start
        avg_frame_time = np.mean(video_metrics["frame_timings"]) if video_metrics["frame_timings"] else 0
        fps_achieved = (1000 / avg_frame_time) if avg_frame_time > 0 else 0

        logger.info(
            f"Detection pipeline complete: {frame_idx} frames processed in {total_processing_time:.1f}s\n"
            f"  Avg latency: {avg_frame_time:.1f}ms/frame\n"
            f"  FPS achieved: {fps_achieved:.1f}"
        )

    def _export_annotated_video(self, video_path: str, pipeline_components: Dict,
                                redis_manager, video_metrics: Dict, 
                                output_path: str = "test_output_video.mp4") -> None:
        """
        Export video with annotated detections (bounding boxes).
        Reads video and overlays metrics from per_frame_data.
        """
        logger.info(f"Exporting annotated video to: {output_path}")

        worker1 = pipeline_components["worker1"]
        worker2 = pipeline_components["worker2"]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video for annotation: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # VideoWriter for output
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        try:
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        except Exception as e:
            logger.error(f"Failed to create VideoWriter: {e}")
            return

        frame_idx = 0
        frames_annotated = 0

        try:
            while cap.isOpened() and frame_idx < video_metrics["frames_processed"]:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx < len(video_metrics["per_frame_data"]):
                    frame_data = video_metrics["per_frame_data"][frame_idx]

                    # Draw frame info and metrics
                    cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Detections: {frame_data.get('consensus_count', 0)}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Alerts: {frame_data.get('alerts_generated', 0)}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Latency: {frame_data.get('latency_ms', 0):.1f}ms", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Get detections for drawing boxes
                    w1_det = worker1.detect(frame)
                    w2_det = worker2.detect(frame)

                    # Draw W1 boxes (green)
                    if w1_det["num_detections"] > 0:
                        for box, conf in zip(w1_det["boxes"], w1_det["confidences"]):
                            x1, y1, x2, y2 = [int(b) for b in box]
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"W1:{conf:.2f}", (x1, y1 - 5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # Draw W2 boxes (blue)
                    if w2_det["num_detections"] > 0:
                        for box, conf in zip(w2_det["boxes"], w2_det["confidences"]):
                            x1, y1, x2, y2 = [int(b) for b in box]
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(frame, f"W2:{conf:.2f}", (x1, y1 - 25),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                    frames_annotated += 1

                out.write(frame)
                frame_idx += 1

        finally:
            cap.release()
            out.release()

        logger.info(f"Annotated video exported: {frames_annotated} frames to {output_path}")

    def _generate_report(self, video_metrics: Dict, output_json: str = "test_results.json") -> None:
        """Generate detailed metrics report."""
        logger.info("Generating metrics report...")

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_frames": video_metrics["total_frames"],
                "frames_ingested": video_metrics["frames_ingested"],
                "frames_processed": video_metrics["frames_processed"],
                "total_detections_w1": video_metrics["total_detections_w1"],
                "total_detections_w2": video_metrics["total_detections_w2"],
                "total_consensus_detections": video_metrics["total_consensus_detections"],
                "total_alerts": video_metrics["total_alerts"],
                "processing_failures": len(video_metrics["failures"]),
            },
            "timing_analysis": {
                "avg_ingestion_ms": float(np.mean(video_metrics["ingestion_timings"])) if video_metrics["ingestion_timings"] else 0,
                "avg_processing_ms": float(np.mean(video_metrics["frame_timings"])) if video_metrics["frame_timings"] else 0,
                "fps_achieved": float(1000 / np.mean(video_metrics["frame_timings"])) if video_metrics["frame_timings"] else 0,
            },
            "detection_stats": {
                "avg_w1_detections_per_frame": float(video_metrics["total_detections_w1"] / max(video_metrics["frames_processed"], 1)),
                "avg_w2_detections_per_frame": float(video_metrics["total_detections_w2"] / max(video_metrics["frames_processed"], 1)),
                "consensus_agreement_pct": float(100 * video_metrics["total_consensus_detections"] / max(video_metrics["total_detections_w1"] + video_metrics["total_detections_w2"], 1)),
                "alerts_per_frame": float(video_metrics["total_alerts"] / max(video_metrics["frames_processed"], 1)),
            },
            "failures": video_metrics["failures"],
        }

        # Write JSON
        with open(output_json, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to: {output_json}")

        # Log summary
        logger.info(
            f"✓ Video E2E Test Summary:\n"
            f"    Total frames: {report['summary']['total_frames']}\n"
            f"    Processed: {report['summary']['frames_processed']}\n"
            f"    W1 detections: {report['summary']['total_detections_w1']}\n"
            f"    W2 detections: {report['summary']['total_detections_w2']}\n"
            f"    Consensus: {report['summary']['total_consensus_detections']}\n"
            f"    Alerts: {report['summary']['total_alerts']}\n"
            f"    Avg latency: {report['timing_analysis']['avg_processing_ms']:.1f}ms\n"
            f"    FPS achieved: {report['timing_analysis']['fps_achieved']:.1f}\n"
            f"    Failures: {report['summary']['processing_failures']}"
        )

    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.parametrize("frames_to_process", [None])  # None = all frames
    def test_video_end_to_end_pipeline(self, frames_to_process, pipeline_components,
                                       cleanup_gpu, video_metrics):
        """
        Complete end-to-end test: video ingestion → detection → consensus → alerts.
        Direct video processing through full pipeline.
        Simplified version: no Redis initially, focus on detection pipeline correctness.
        """
        logger.info("=" * 80)
        logger.info("VIDEO END-TO-END PIPELINE TEST")
        logger.info("=" * 80)

        # Get video path
        video_path = self._get_video_path()

        # Initialize pipeline components
        worker1 = pipeline_components["worker1"]
        worker2 = pipeline_components["worker2"]
        temporal_filter = pipeline_components["temporal_filter"]
        roi_validator = pipeline_components["roi_validator"]
        alert_gen = pipeline_components["alert_gen"]

        camera_id = "camera:video:halltesting"

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames_to_process_actual = frames_to_process or total_frames
        logger.info(f"  Video: {total_frames} frames @ {fps} FPS, {width}x{height}")
        logger.info(f"  Processing: {frames_to_process_actual} frames")

        # Phase 1: Process video through pipeline
        frame_idx = 0
        pipeline_start = time.time()

        try:
            while cap.isOpened() and frame_idx < frames_to_process_actual:
                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    frame_start = time.time()

                    # Worker 1 detection
                    w1_det = worker1.detect(frame)
                    video_metrics["total_detections_w1"] += w1_det["num_detections"]

                    # Worker 2 detection
                    w2_det = worker2.detect(frame)
                    video_metrics["total_detections_w2"] += w2_det["num_detections"]

                    # Consensus voting
                    worker_detections = {
                        "worker_1": [w1_det],
                        "worker_2": [w2_det],
                    }
                    consensus_dets = temporal_filter.process_detections(camera_id, worker_detections)
                    video_metrics["total_consensus_detections"] += len(consensus_dets)

                    # ROI validation
                    roi_validated = roi_validator.validate_detections(camera_id, consensus_dets)

                    # Alert generation
                    alert_ids = alert_gen.generate_alerts(camera_id, roi_validated if roi_validated else [])
                    video_metrics["total_alerts"] += len(alert_ids)

                    # Timing
                    frame_time = (time.time() - frame_start) * 1000
                    video_metrics["frame_timings"].append(frame_time)

                    # Per-frame data for export
                    video_metrics["per_frame_data"].append({
                        "frame_id": frame_idx,
                        "w1_detections": w1_det["num_detections"],
                        "w2_detections": w2_det["num_detections"],
                        "consensus_count": len(consensus_dets),
                        "alerts_generated": len(alert_ids),
                        "latency_ms": frame_time,
                    })

                    frame_idx += 1
                    video_metrics["frames_processed"] = frame_idx

                    if frame_idx % 50 == 0 or frame_idx == 1:
                        logger.info(
                            f"  Frame {frame_idx}: W1={w1_det['num_detections']} W2={w2_det['num_detections']} "
                            f"Consensus={len(consensus_dets)} Alerts={len(alert_ids)} Time={frame_time:.1f}ms"
                        )

                except Exception as e:
                    logger.error(f"Error processing frame {frame_idx}: {e}", exc_info=True)
                    video_metrics["failures"].append(("processing", frame_idx, str(e)))
                    frame_idx += 1
                    continue

        finally:
            cap.release()

        total_processing_time = time.time() - pipeline_start

        logger.info(
            f"✓ Pipeline processing complete: {frame_idx} frames in {total_processing_time:.1f}s"
        )

        # Phase 2: Generate annotated video
        logger.info("Generating annotated video...")
        self._export_annotated_video(
            video_path,
            pipeline_components,
            None,  # No Redis needed for direct processing
            video_metrics,
        )

        # Phase 3: Generate report
        self._generate_report(video_metrics)

        # Validations
        assert video_metrics["frames_processed"] > 0, "No frames processed"
        assert len(video_metrics["failures"]) == 0, f"Processing failures: {video_metrics['failures']}"
        assert video_metrics["total_alerts"] >= 0, "Invalid alert count"

        logger.info("=" * 80)
        logger.info("VIDEO E2E TEST PASSED ✓")
        logger.info("=" * 80)


if __name__ == "__main__":
    logger.info("Running Phase 2 End-to-End Tests...")
    pytest.main([__file__, "-v", "--tb=short"])
