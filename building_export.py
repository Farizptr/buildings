import json

def save_buildings_to_json(results_data, output_path="buildings.json"):
    """
    Extract building coordinates from detection results and save to a regular JSON file
    
    Args:
        results_data: Detection results from detect_buildings_in_polygon
        output_path: Path to save the JSON file
        
    Returns:
        Path to the saved JSON file
    """
    # Create a list to store building data
    buildings = []
    
    # Process each tile detection
    building_id = 1
    for tile_detection in results_data['detections']:
        # Get tile bounds
        bounds = tile_detection['bounds']
        west, south, east, north = bounds
        
        # Calculate tile width and height in degrees
        tile_width = east - west
        tile_height = north - south
        
        # Get boxes and confidences
        boxes = tile_detection['boxes']
        confidences = tile_detection['confidences']
        
        # Process each box
        for i, box in enumerate(boxes):
            # Get normalized coordinates (0-1) within the tile
            x1, y1, x2, y2 = box
            
            # Convert to image coordinates (assuming 256x256 images)
            img_width = 256
            img_height = 256
            x1_norm = x1 / img_width
            y1_norm = y1 / img_height
            x2_norm = x2 / img_width
            y2_norm = y2 / img_height
            
            # Convert to geo coordinates
            geo_x1 = west + x1_norm * tile_width
            geo_y1 = north - y1_norm * tile_height  # Flip y-axis
            geo_x2 = west + x2_norm * tile_width
            geo_y2 = north - y2_norm * tile_height  # Flip y-axis
            
            # Calculate center point
            center_lon = (geo_x1 + geo_x2) / 2
            center_lat = (geo_y1 + geo_y2) / 2
            
            # Create a building entry
            building = {
                "building_id": building_id,
                "latitude": center_lat,
                "longitude": center_lon,
                "confidence": confidences[i] if i < len(confidences) else 0.5
            }
            
            # Add to buildings list
            buildings.append(building)
            building_id += 1
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(buildings, f, indent=2)
    
    print(f"Building coordinates saved to {output_path}")
    print(f"Total buildings saved: {len(buildings)}")
    
    return output_path 