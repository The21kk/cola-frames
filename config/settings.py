import os
from dotenv import load_dotenv

load_dotenv()

# =====================================
# Redis Configuration
# =====================================
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379)) # Default Redis port
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# =====================================
# Stream Configuration
# =====================================
STREAM_PREFIX = "camera:"
FRAME_BATCH_SIZE = 10
MAX_STREAM_LENGTH = 1000  # Keeps ~1 second of frames at 1 FPS

# =====================================
# Detection Configuration
# =====================================
DETECTION_CONFIDENCE_THRESHOLD = 0.5
ENSEMBLE_AGREEMENT_THRESHOLD = 0.5  # % of models that must agree

# =====================================
# Temporal Logic
# =====================================
TEMPORAL_PERSISTENCE_SECONDS = 2

# =====================================
# Frame Serialization
# =====================================
FRAME_JPEG_QUALITY = 80  # 0-100, higher = better quality but larger size
FRAME_FORMAT = "base64_jpeg"

# =====================================
# Worker Configuration (Scalable)
# =====================================
# Path to worker configuration file (YAML)
WORKERS_CONFIG_PATH = os.getenv(
    "WORKERS_CONFIG_PATH",
    os.path.join(os.path.dirname(__file__), "workers_config.yaml"),
)

# GPU Device Assignment (Legacy - kept for backward compatibility)
USE_GPU = os.getenv("USE_GPU", "True").lower() == "true"
DEVICE_WORKER_1 = os.getenv("DEVICE_WORKER_1", "cuda:0")  # YOLOv8 + Vision Transformer
DEVICE_WORKER_2 = os.getenv("DEVICE_WORKER_2", "cuda:0")  # Faster R-CNN + RT-DETR Lite
WORKERS_PER_CAMERA = 2  # Legacy setting
WORKER_QUEUE_TIMEOUT = 5  # seconds

# Model Precision & Batch Processing
MIXED_PRECISION_FP16 = os.getenv("MIXED_PRECISION_FP16", "True").lower() == "true"
DETECTION_BATCH_SIZE = int(os.getenv("DETECTION_BATCH_SIZE", 4))

# Consensus & Voting
CONSENSUS_IOU_THRESHOLD = float(os.getenv("CONSENSUS_IOU_THRESHOLD", 0.3))  # IoU for bbox match
CONSENSUS_AGREEMENT_RATIO = float(os.getenv("CONSENSUS_AGREEMENT_RATIO", 1.0))  # 1.0 = all workers must agree

# Detection Stream Storage
MAX_DETECTIONS_STREAM_LENGTH = 10000  # Trim detection streams to prevent memory bloat
DETECTIONS_STREAM_PREFIX = "detections:"

# Model Caching
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./model_cache")

# =====================================
# Logging
# =====================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
