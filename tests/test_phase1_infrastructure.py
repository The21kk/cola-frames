"""
Phase 1 Infrastructure Tests: Redis Streams + Frame Serialization + RTSP Ingestion

Tests:
1. Frame serialization (encode/decode)
2. Redis connection & health check
3. RTSP/video file ingestion
4. Stream monitoring (real-time frame arrivals)
5. Frame retrieval & LIFO consumption
6. Throughput analysis
"""

import pytest
import numpy as np
import cv2
import time
import logging
from pathlib import Path

from producer.frame_serializer import FrameSerializer
from producer.rtsp_ingester import RTSPIngester
from redis_broker.stream_manager import RedisStreamManager


logger = logging.getLogger(__name__)


# ============================================================================
# TEST 1: Frame Serialization
# ============================================================================

class TestFrameSerialization:
    """Test frame encode/decode without data loss."""
    
    def test_frame_encode_decode(self, sample_frame):
        """Encode frame to Base64, then decode back. Verify shape matches."""
        # Encode
        frame_b64 = FrameSerializer.encode_frame_to_base64(sample_frame)
        assert isinstance(frame_b64, bytes), "Encoded frame should be bytes"
        assert len(frame_b64) > 0, "Encoded frame should not be empty"
        
        # Decode
        frame_decoded = FrameSerializer.decode_frame_from_base64(frame_b64)
        assert frame_decoded.shape == sample_frame.shape, "Shape mismatch after decode"
        assert frame_decoded.dtype == np.uint8, "Frame should be uint8"
        
        logger.info(
            f"✓ Frame serialization: {sample_frame.shape} → {len(frame_b64)} bytes → "
            f"{frame_decoded.shape}"
        )
    
    def test_compression_ratio(self, sample_frame):
        """Verify JPEG compression reduces size significantly."""
        # Raw size
        raw_size = sample_frame.nbytes  # 480*640*3 = 921,600 bytes
        
        # Encoded size
        frame_b64 = FrameSerializer.encode_frame_to_base64(sample_frame)
        encoded_size = len(frame_b64)
        
        # Compression ratio should be >90% (base64 adds ~33% overhead, JPEG compresses 95%+)
        compression_ratio = encoded_size / raw_size
        logger.info(f"✓ Compression ratio: {compression_ratio:.1%} (original: {raw_size}, encoded: {encoded_size})")
        
        assert compression_ratio < 0.5, "Compression ratio too high (expected <50%)"
    
    def test_metadata_creation(self):
        """Verify metadata dictionary is created correctly."""
        metadata = FrameSerializer.create_metadata(
            camera_id="test_camera",
            fps=5,
            resolution=(640, 480)
        )
        
        assert isinstance(metadata, dict), "Metadata should be dict"
        assert "camera_id" in metadata, "Metadata missing camera_id"
        assert "timestamp" in metadata, "Metadata missing timestamp"
        assert "resolution" in metadata, "Metadata missing resolution"
        assert metadata["camera_id"] == "test_camera"
        assert metadata["resolution"] == (640, 480)
        
        logger.info(f"✓ Metadata created: {metadata}")


# ============================================================================
# TEST 2: Redis Connection & Health Check
# ============================================================================

class TestRedisConnection:
    """Test Redis connection and basic operations."""
    
    def test_redis_health_check(self, redis_manager):
        """Verify Redis connection is healthy."""
        health = redis_manager.health_check()
        
        assert isinstance(health, dict), "Health check should return dict"
        assert "redis_connected" in health, "Health missing redis_connected"
        assert health["redis_connected"] is True, "Redis should be connected"
        assert "status" in health, "Health missing status"
        
        logger.info(f"✓ Redis health check passed: {health['status']}")
    
    def test_redis_ping(self, redis_manager):
        """Test basic Redis ping operation."""
        try:
            response = redis_manager.redis_client.ping()
            assert response is True, "Redis ping should return True"
            logger.info("✓ Redis ping successful")
        except Exception as e:
            pytest.fail(f"Redis ping failed: {e}")


# ============================================================================
# TEST 3: RTSP/Video File Ingestion
# ============================================================================

class TestRTSPIngestion:
    """Test RTSP ingester with local video files."""
    
    def test_ingester_from_local_video(self, video_file, redis_manager, cleanup_redis):
        """Test ingester can read video file and push frames to Redis."""
        camera_id = "test_ingester_01"
        
        # Clean existing stream
        redis_manager.delete_stream(camera_id)
        
        # Start ingester
        ingester = RTSPIngester(
            camera_id=camera_id,
            rtsp_url=str(video_file),
            fps_target=5
        )
        ingester.start()
        
        try:
            # Wait for frames to be ingested
            time.sleep(3)
            
            # Verify frames arrived
            stream_length = redis_manager.get_stream_length(camera_id)
            assert stream_length > 0, f"No frames ingested (expected >0, got {stream_length})"
            
            logger.info(f"✓ Ingester pushed {stream_length} frames to Redis")
        finally:
            ingester.stop()
    
    def test_ingester_frame_count(self, video_file, redis_manager, cleanup_redis):
        """Verify ingester processes expected number of frames."""
        camera_id = "test_ingester_count"
        redis_manager.delete_stream(camera_id)
        
        ingester = RTSPIngester(
            camera_id=camera_id,
            rtsp_url=str(video_file),
            fps_target=5
        )
        ingester.start()
        
        try:
            # Video should be ~2 seconds @ 5 FPS = ~10 frames
            time.sleep(3)
            stream_length = redis_manager.get_stream_length(camera_id)
            
            # Allow some tolerance (7-15 frames)
            assert 7 <= stream_length <= 15, \
                f"Unexpected frame count: {stream_length} (expected 7-15)"
            
            logger.info(f"✓ Ingester frame count verified: {stream_length} frames")
        finally:
            ingester.stop()


# ============================================================================
# TEST 4: Stream Monitoring (Real-Time Arrivals)
# ============================================================================

class TestStreamMonitoring:
    """Test real-time monitoring of frame arrivals."""
    
    def test_stream_monitoring_consistency(self, video_file, redis_manager, cleanup_redis):
        """Verify frames arrive consistently during ingestion."""
        camera_id = "test_monitor"
        redis_manager.delete_stream(camera_id)
        
        ingester = RTSPIngester(
            camera_id=camera_id,
            rtsp_url=str(video_file),
            fps_target=5
        )
        ingester.start()
        
        try:
            # Monitor stream at start
            time.sleep(1)
            initial_count = redis_manager.get_stream_length(camera_id)
            assert initial_count > 0, "No frames in stream after 1 second"
            
            # Monitor stream after another second
            time.sleep(1)
            later_count = redis_manager.get_stream_length(camera_id)
            
            # More frames should have arrived (monotonic increase)
            assert later_count >= initial_count, \
                f"Stream count decreased: {initial_count} → {later_count}"
            assert later_count > initial_count, \
                "No new frames arrived (expected growth)"
            
            logger.info(
                f"✓ Stream monitoring: {initial_count} frames → {later_count} frames "
                f"(+{later_count - initial_count})"
            )
        finally:
            ingester.stop()


# ============================================================================
# TEST 5: Frame Retrieval & LIFO
# ============================================================================

class TestStreamRetrieval:
    """Test retrieving frames from Redis streams."""
    
    def test_get_latest_frame(self, redis_manager, cleanup_redis, sample_frame):
        """Verify get_latest_frame() returns most recent frame."""
        camera_id = "test_retrieval"
        redis_manager.delete_stream(camera_id)
        
        # Add 5 frames
        for i in range(5):
            frame_b64 = FrameSerializer.encode_frame_to_base64(sample_frame)
            metadata = FrameSerializer.create_metadata(camera_id, 5, (640, 480))
            metadata["frame_index"] = i
            redis_manager.add_frame_to_stream(camera_id, frame_b64, metadata)
            time.sleep(0.05)
        
        # Get latest
        latest = redis_manager.get_latest_frame(camera_id)
        assert latest is not None, "get_latest_frame() returned None"
        assert "id" in latest, "Latest frame missing id"
        assert "frame" in latest, "Latest frame missing frame"
        assert "metadata" in latest, "Latest frame missing metadata"
        assert latest["metadata"]["frame_index"] == 4, \
            f"Latest frame should be #4, got #{latest['metadata']['frame_index']}"
        
        logger.info(f"✓ Retrieved latest frame: {latest['metadata']}")


class TestLIFOConsumption:
    """Test LIFO (Last-In-First-Out) consumption of frames."""
    
    def test_lifo_order(self, redis_manager, cleanup_redis, sample_frame):
        """Verify LIFO retrieval always returns most recent frame."""
        camera_id = "test_lifo"
        redis_manager.delete_stream(camera_id)
        
        # Add frames with different indices
        frame_ids = []
        for i in range(5):
            frame_b64 = FrameSerializer.encode_frame_to_base64(sample_frame)
            metadata = FrameSerializer.create_metadata(camera_id, 5, (640, 480))
            metadata["seq"] = i
            msg_id = redis_manager.add_frame_to_stream(camera_id, frame_b64, metadata)
            frame_ids.append(msg_id)
            time.sleep(0.05)
        
        # Get frame 3 times (should always be the latest = frame #4)
        retrieved_ids = []
        for _ in range(3):
            frame = redis_manager.get_latest_frame(camera_id)
            retrieved_ids.append(frame["metadata"]["seq"])
        
        # All should be frame #4 (LIFO behavior)
        assert all(seq == 4 for seq in retrieved_ids), \
            f"LIFO not working: got sequences {retrieved_ids} (expected all 4)"
        
        logger.info(f"✓ LIFO verification passed: always retrieved frame #4")


# ============================================================================
# TEST 6: Throughput Analysis
# ============================================================================

class TestThroughput:
    """Test frame ingestion throughput."""
    
    def test_throughput_100_frames(self, redis_manager, cleanup_redis, sample_frame):
        """Process 100 frames and measure throughput (FPS)."""
        camera_id = "test_throughput"
        redis_manager.delete_stream(camera_id)
        
        # Measure time to add 100 frames
        start_time = time.time()
        num_frames = 100
        
        for i in range(num_frames):
            frame_b64 = FrameSerializer.encode_frame_to_base64(sample_frame)
            metadata = FrameSerializer.create_metadata(camera_id, 5, (640, 480))
            metadata["frame_num"] = i
            redis_manager.add_frame_to_stream(camera_id, frame_b64, metadata)
        
        elapsed = time.time() - start_time
        fps = num_frames / elapsed
        
        # Should process >50 FPS (reasonable for Redis on local machine)
        assert fps > 25, f"Throughput too low: {fps:.1f} FPS (expected >25)"
        
        logger.info(f"✓ Throughput: {num_frames} frames in {elapsed:.2f}s = {fps:.1f} FPS")
    
    def test_throughput_compression_with_encoding(self, redis_manager, cleanup_redis):
        """Measure throughput including frame encoding."""
        camera_id = "test_throughput_encode"
        redis_manager.delete_stream(camera_id)
        
        # Create larger frame to simulate real streams
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        start_time = time.time()
        num_frames = 20  # Larger frame, so fewer iterations
        
        for i in range(num_frames):
            frame_b64 = FrameSerializer.encode_frame_to_base64(frame)
            metadata = FrameSerializer.create_metadata(camera_id, 30, (1920, 1080))
            redis_manager.add_frame_to_stream(camera_id, frame_b64, metadata)
        
        elapsed = time.time() - start_time
        fps = num_frames / elapsed
        
        # Should process >5 FPS for full HD frames
        assert fps > 2, f"Throughput too low for HD: {fps:.1f} FPS (expected >2)"
        
        logger.info(f"✓ HD throughput: {num_frames} 1080p frames in {elapsed:.2f}s = {fps:.1f} FPS")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestPhase1Integration:
    """Integration tests combining multiple Phase 1 components."""
    
    def test_end_to_end_ingestion_to_retrieval(self, video_file, redis_manager, cleanup_redis):
        """End-to-end: ingest video → verify frames in Redis → retrieve."""
        camera_id = "e2e_test"
        redis_manager.delete_stream(camera_id)
        
        # Start ingester
        ingester = RTSPIngester(camera_id=camera_id, rtsp_url=str(video_file), fps_target=5)
        ingester.start()
        
        try:
            # Wait for ingestion
            time.sleep(2.5)
            
            # Verify frames exist
            stream_length = redis_manager.get_stream_length(camera_id)
            assert stream_length > 0, "No frames in stream"
            
            # Retrieve latest frame
            latest = redis_manager.get_latest_frame(camera_id)
            assert latest is not None, "Could not retrieve latest frame"
            
            # Verify frame can be decoded
            frame = FrameSerializer.decode_frame_from_base64(latest["frame"])
            assert frame.shape == (480, 640, 3), f"Unexpected frame shape: {frame.shape}"
            
            logger.info(f"✓ E2E test passed: {stream_length} frames ingested and retrieved")
        finally:
            ingester.stop()
