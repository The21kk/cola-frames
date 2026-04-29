#!/usr/bin/env python
"""Refactor Phase 2 tests to use GenericDetector."""

import re
import sys

# Lee el archivo
test_file = "tests/test_phase2_detection.py"
with open(test_file, "r", encoding="utf-8") as f:
    content = f.read()

# Reemplazos
replacements = [
    # YOLOVitDetector initialization -> GenericDetector
    (
        r'detector = YOLOVitDetector\(\s*device=DEVICE_WORKER_1\s*\)',
        'detector = GenericDetector(\n            model_type="yolov10",\n            model_name="yolov10m",\n            device=DEVICE_WORKER_1,\n            batch_size=4,\n        )'
    ),
    (
        r'detector = YOLOVitDetector\(\s*device=DEVICE_WORKER_1 if torch\.cuda\.is_available\(\) else "cpu"\s*\)',
        'detector = GenericDetector(\n                model_type="yolov10",\n                model_name="yolov10m",\n                device=DEVICE_WORKER_1 if torch.cuda.is_available() else "cpu",\n                batch_size=4,\n            )'
    ),
    (
        r'detector = YOLOVitDetector\(\s*device=DEVICE_WORKER_1 if torch\.cuda\.is_available\(\) else "cpu",\s*batch_size=4\s*\)',
        'detector = GenericDetector(\n                model_type="yolov10",\n                model_name="yolov10m",\n                device=DEVICE_WORKER_1 if torch.cuda.is_available() else "cpu",\n                batch_size=4,\n            )'
    ),
    # FasterRCNNRtdetrDetector -> GenericDetector
    (
        r'detector = FasterRCNNRtdetrDetector\(\s*device=DEVICE_WORKER_2 if torch\.cuda\.is_available\(\) else "cpu"\s*\)',
        'detector = GenericDetector(\n                model_type="faster_rcnn",\n                model_name="fasterrcnn_resnet50_fpn",\n                device=DEVICE_WORKER_2 if torch.cuda.is_available() else "cpu",\n                batch_size=2,\n            )'
    ),
    (
        r'detector = FasterRCNNRtdetrDetector\(\s*device=DEVICE_WORKER_2 if torch\.cuda\.is_available\(\) else "cpu",\s*batch_size=2\s*\)',
        'detector = GenericDetector(\n                model_type="faster_rcnn",\n                model_name="fasterrcnn_resnet50_fpn",\n                device=DEVICE_WORKER_2 if torch.cuda.is_available() else "cpu",\n                batch_size=2,\n            )'
    ),
]

# Apply replacements
for old, new in replacements:
    content = re.sub(old, new, content, flags=re.MULTILINE)

# Reemplaza aserciones que verifican atributos específicos
content = re.sub(
    r'assert detector\.model_name == "yolo_vit"\s*\n\s*assert detector\.yolo_model is not None\s*\n\s*assert detector\.vit_model is not None',
    'assert detector.model_name is not None\n            assert detector.model is not None',
    content
)

content = re.sub(
    r'assert detector\.model_name == "frcnn_rtdetr"\s*\n\s*assert detector\.frcnn_model is not None',
    'assert detector.model_name is not None\n            assert detector.model is not None',
    content
)

# Write back
with open(test_file, "w", encoding="utf-8") as f:
    f.write(content)

print(f"Refactored {test_file}")

# Do the same for E2E tests
e2e_file = "tests/test_phase2_end_to_end.py"
with open(e2e_file, "r", encoding="utf-8") as f:
    content = f.read()

# Similar replacements for E2E
content = re.sub(
    r'worker1 = YOLOVitDetector\(\s*device=DEVICE_WORKER_1 if torch\.cuda\.is_available\(\) else "cpu",\s*confidence_threshold=DETECTION_CONFIDENCE_THRESHOLD,\s*\)',
    'worker1 = GenericDetector(\n            model_type="yolov10",\n            model_name="yolov10m",\n            device=DEVICE_WORKER_1 if torch.cuda.is_available() else "cpu",\n            confidence_threshold=DETECTION_CONFIDENCE_THRESHOLD,\n            batch_size=4,\n        )',
    content
)

content = re.sub(
    r'worker2 = FasterRCNNRtdetrDetector\(\s*device=DEVICE_WORKER_2 if torch\.cuda\.is_available\(\) else "cpu",\s*confidence_threshold=DETECTION_CONFIDENCE_THRESHOLD,\s*\)',
    'worker2 = GenericDetector(\n            model_type="faster_rcnn",\n            model_name="fasterrcnn_resnet50_fpn",\n            device=DEVICE_WORKER_2 if torch.cuda.is_available() else "cpu",\n            confidence_threshold=DETECTION_CONFIDENCE_THRESHOLD,\n            batch_size=2,\n        )',
    content
)

with open(e2e_file, "w", encoding="utf-8") as f:
    f.write(content)

print(f"Refactored {e2e_file}")
print("✓ Refactoring complete!")
