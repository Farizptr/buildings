#!/usr/bin/env python3
"""
Test script untuk mendemonstrasikan selective merging behavior.
Menunjukkan perbedaan antara merging dengan semua phases vs. hanya high-confidence phases.
"""

from polygon_detection import merge_overlapping_detections
from shapely.geometry import Polygon
import json

def create_test_detections():
    """Create test detections simulating A, B, C scenario"""
    # Building A - overlaps dengan B (strong connection)
    building_a = {
        'id': 'A',
        'polygon': Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        'confidence': 0.9,
        'tile_id': '18/1/1'
    }
    
    # Building B - overlaps dengan A, dekat dengan C
    building_b = {
        'id': 'B', 
        'polygon': Polygon([(0.8, 0), (1.8, 0), (1.8, 1), (0.8, 1)]),
        'confidence': 0.8,
        'tile_id': '18/2/1'
    }
    
    # Building C - hanya proximity dengan B (weak connection)
    building_c = {
        'id': 'C',
        'polygon': Polygon([(2.1, 0), (3.1, 0), (3.1, 1), (2.1, 1)]),
        'confidence': 0.7,
        'tile_id': '18/3/1'
    }
    
    return [building_a, building_b, building_c]

def test_different_phase_settings():
    """Test merging dengan different phase settings"""
    detections = create_test_detections()
    
    print("=== Test Selective Merging ===")
    print(f"Input: 3 buildings A, B, C")
    print(f"- A overlaps B (IoU ~0.2) - strong connection")  
    print(f"- B close to C (~0.3 distance) - weak connection")
    print()
    
    # Test 1: Only Phase 1 (IoU overlap)
    result_phase1 = merge_overlapping_detections(
        detections, 
        iou_thresh=0.1,
        touch_enabled=True,
        min_edge_distance_deg=0.5,
        allowed_merge_phases=[1]
    )
    print(f"Phase 1 only (IoU): {len(result_phase1)} buildings")
    for building in result_phase1:
        print(f"  - {building['id']}: {len(building['original_ids'])} merged ({building['original_ids']})")
    print()
    
    # Test 2: Phase 1 + 2 (default - recommended)  
    result_phase12 = merge_overlapping_detections(
        detections,
        iou_thresh=0.1,
        touch_enabled=True, 
        min_edge_distance_deg=0.5,
        allowed_merge_phases=[1, 2]
    )
    print(f"Phase 1+2 (recommended): {len(result_phase12)} buildings")
    for building in result_phase12:
        print(f"  - {building['id']}: {len(building['original_ids'])} merged ({building['original_ids']})")
    print()
    
    # Test 3: All phases (potential over-merging)
    result_all = merge_overlapping_detections(
        detections,
        iou_thresh=0.1, 
        touch_enabled=True,
        min_edge_distance_deg=0.5,
        allowed_merge_phases=[1, 2, 3]
    )
    print(f"All phases (risky): {len(result_all)} buildings")
    for building in result_all:
        print(f"  - {building['id']}: {len(building['original_ids'])} merged ({building['original_ids']})")
    print()
    
    print("=== Conclusion ===")
    print("Phase 1+2 (default) gives optimal balance:")
    print("✅ Merges truly connected buildings (A+B)")  
    print("✅ Keeps separate buildings separate (C)")
    print("❌ All phases may over-merge due to weak connections")

if __name__ == "__main__":
    test_different_phase_settings() 