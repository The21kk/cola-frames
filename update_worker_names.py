#!/usr/bin/env python3
"""Update worker names in test file from old ensemble names to new names."""

with open('tests/test_phase2_end_to_end.py', 'r') as f:
    content = f.read()

# Replace worker names
content = content.replace('"yolo_vit"', '"worker_1"')
content = content.replace('"frcnn_rtdetr"', '"worker_2"')

with open('tests/test_phase2_end_to_end.py', 'w') as f:
    f.write(content)

print('✓ Worker names updated in test file')
