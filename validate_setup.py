#!/usr/bin/env python
"""Validation script for detector setup."""

import sys
sys.path.insert(0, '.')

print('=== Validación: Import de GenericDetector ===')
try:
    from workers.generic_detector import GenericDetector
    print('✓ GenericDetector importado correctamente')
except ImportError as e:
    print(f'✗ Error: {e}')
    exit(1)

print('\n=== Validación: Registry ===')
try:
    from workers.detector_registry import list_registered_detectors
    detectors = list_registered_detectors()
    print(f'Detectores registrados: {detectors}')
    if 'GenericDetector' in detectors:
        print('✓ GenericDetector registrado en registry')
    else:
        print('✗ GenericDetector NO está registrado')
        exit(1)
except Exception as e:
    print(f'✗ Error: {e}')
    exit(1)

print('\n=== Validación: Factory can load config ===')
try:
    from workers.detector_factory import DetectorFactory
    factory = DetectorFactory('config/workers_config.yaml')
    num_workers = len(factory.config['workers'])
    print(f'✓ Factory cargó config con {num_workers} workers')
    for worker in factory.config['workers']:
        model_type = worker['model_type']
        model_name = worker['model_name']
        name = worker['name']
        print(f'  - {name} ({model_type}/{model_name})')
except Exception as e:
    print(f'✗ Error: {e}')
    exit(1)

print('\n=== VALIDACIÓN OK ===')
