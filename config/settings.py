import os
from dotenv import load_dotenv

load_dotenv()

# =====================================
# Redis Configuration
# =====================================
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
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
ENSEMBLE_AGREEMENT_THRESHOLD = 0.7  # % of models that must agree

# =====================================
# Temporal Logic
# =====================================
TEMPORAL_PERSISTENCE_SECONDS = 3

# =====================================
# Frame Serialization
# =====================================
FRAME_JPEG_QUALITY = 80  # 0-100, higher = better quality but larger size
FRAME_FORMAT = "base64_jpeg"

# =====================================
# Worker Configuration
# =====================================
WORKERS_PER_CAMERA = 2
WORKER_QUEUE_TIMEOUT = 5  # seconds

# =====================================
# Logging
# =====================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
