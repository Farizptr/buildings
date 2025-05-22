import os
import json
import sys
import tempfile
import time
from tqdm import tqdm
import concurrent.futures
from functools import partial
from shapely.geometry import box
from shapely.ops import unary_union # Added for merging

from detection import load_model, detect_buildings
from geojson_utils import load_geojson, extract_polygon, create_example_geojson
from tile_utils import get_tile_bounds, get_tiles_for_polygon, get_tile_image, process_tile_detections
from visualization import visualize_polygon_detections
from building_export import save_buildings_to_json

# --- Helper function to convert tile detections to geographic Shapely polygons ---
def convert_tile_detections_to_shapely_polygons(all_tile_detections):
    """
    Converts raw tile-based detections into a flat list of Shapely polygons 
    with geographic coordinates, associated confidence scores, and original tile ID.
    """
    shapely_polygons_with_attrs = []
    detection_id_counter = 0
    for tile_detection in all_tile_detections:
        # Each tile_detection should have 'tile': "z/x/y", 'bounds', 'boxes', 'confidences'
        tile_id_str = tile_detection.get('tile', 'unknown_tile') # z/x/y string
        bounds = tile_detection['bounds'] # west, south, east, north
        west, south, east, north = bounds
        tile_width_deg = east - west
        tile_height_deg = north - south
        
        img_width, img_height = 256, 256 # Assuming tile image size used for normalization

        for i, bbox_coords in enumerate(tile_detection['boxes']):
            x1_norm, y1_norm, x2_norm, y2_norm = bbox_coords # Normalized 0-1 within tile (from model output)
            
            # Convert normalized to absolute pixel coordinates if they are not already
            # This step depends on how 'boxes' are stored. If they are [0-1], convert.
            # If they are already in pixel coords [0-255], this can be skipped or adjusted.
            # Assuming 'boxes' from process_tile_detections are [x1,y1,x2,y2] in pixel coords of the tile_image
            # For this example, let's assume bbox_coords are [x_min_pixel, y_min_pixel, x_max_pixel, y_max_pixel]
            # If they are normalized (0-1), then:
            # x1_pixel = x1_norm * img_width
            # y1_pixel = y1_norm * img_height
            # x2_pixel = x2_norm * img_width
            # y2_pixel = y2_norm * img_height
            # For now, assuming bbox_coords are already in pixel values [0-255] for a 256x256 tile
            x1_pixel, y1_pixel, x2_pixel, y2_pixel = bbox_coords

            # Convert pixel coordinates to normalized (0-1) within this tile
            x1_norm_tile = x1_pixel / img_width
            y1_norm_tile = y1_pixel / img_height
            x2_norm_tile = x2_pixel / img_width
            y2_norm_tile = y2_pixel / img_height

            # Convert normalized tile coordinates to geographic coordinates
            geo_x1 = west + x1_norm_tile * tile_width_deg
            geo_y1_bottom_up = south + (1 - y2_norm_tile) * tile_height_deg # y is often from top in images
            geo_x2 = west + x2_norm_tile * tile_width_deg
            geo_y2_bottom_up = south + (1 - y1_norm_tile) * tile_height_deg
            
            # Create Shapely box: box(minx, miny, maxx, maxy)
            # Ensure minx < maxx and miny < maxy
            current_shapely_box = box(
                min(geo_x1, geo_x2), 
                min(geo_y1_bottom_up, geo_y2_bottom_up), 
                max(geo_x1, geo_x2), 
                max(geo_y1_bottom_up, geo_y2_bottom_up)
            )
            
            confidence = tile_detection['confidences'][i] if i < len(tile_detection['confidences']) else 0.0
            
            shapely_polygons_with_attrs.append({
                'id': f"det_{detection_id_counter}",
                'polygon': current_shapely_box,
                'confidence': confidence,
                'tile_id': tile_id_str # Menyimpan ID tile asal
            })
            detection_id_counter += 1
            
    return shapely_polygons_with_attrs

def get_long_axis(polygon):
    """
    Menghitung sumbu panjang dari sebuah poligon.
    Mengembalikan vektor arah sumbu panjang (dinormalisasi).
    """
    # Dapatkan minimum bounding rectangle
    try:
        # Gunakan minimum_rotated_rectangle jika tersedia
        mbr = polygon.minimum_rotated_rectangle
    except AttributeError:
        # Fallback ke envelope jika minimum_rotated_rectangle tidak tersedia
        mbr = polygon.envelope
    
    # Dapatkan koordinat dari MBR
    coords = list(mbr.exterior.coords)
    
    # Hitung panjang sisi-sisi MBR
    sides = []
    for i in range(len(coords) - 1):
        dx = coords[i+1][0] - coords[i][0]
        dy = coords[i+1][1] - coords[i][1]
        sides.append(((dx, dy), (dx**2 + dy**2)**0.5))
    
    # Temukan sisi terpanjang
    longest_side = max(sides, key=lambda x: x[1])
    
    # Normalisasi vektor
    dx, dy = longest_side[0]
    length = longest_side[1]
    if length > 0:
        dx /= length
        dy /= length
    
    return (dx, dy)

def calculate_axis_alignment(axis1, axis2):
    """
    Menghitung keselarasan antara dua sumbu.
    Mengembalikan nilai antara 0 dan 1, di mana 1 berarti sejajar sempurna.
    """
    # Hitung dot product
    dot_product = abs(axis1[0] * axis2[0] + axis1[1] * axis2[1])
    
    # Dot product dari dua vektor unit adalah kosinus dari sudut antara mereka
    # Kita menggunakan nilai absolut untuk mengatasi orientasi yang berlawanan
    return dot_product

def parse_tile_id(tile_id_str):
    """Parse a tile ID string (z/x/y) into its components"""
    if tile_id_str == 'UNKNOWN':
        return None, None, None
    parts = tile_id_str.split('/')
    if len(parts) != 3:
        return None, None, None
    try:
        z, x, y = map(int, parts)
        return z, x, y
    except ValueError:
        return None, None, None

def calculate_boundary_proximity(poly, tile_id_str, other_poly, other_tile_id_str):
    """Calculate how close polygons are to their shared tile boundary"""
    z1, x1, y1 = parse_tile_id(tile_id_str)
    z2, x2, y2 = parse_tile_id(other_tile_id_str)
    
    if None in (z1, x1, y1, z2, x2, y2) or z1 != z2:
        return 0  # Invalid tile IDs or different zoom levels
    
    # Determine if tiles are adjacent
    is_adjacent = (abs(x1 - x2) <= 1 and abs(y1 - y2) <= 1)
    if not is_adjacent:
        return 0  # Tiles are not adjacent
    
    # Determine the direction of the shared boundary
    dx, dy = x2 - x1, y2 - y1
    
    # Get centroids
    c1 = poly.centroid
    c2 = other_poly.centroid
    
    # Calculate boundary proximity score based on how aligned the buildings are
    # with the direction of the tile boundary
    if dx != 0 and dy == 0:  # Horizontal boundary
        # Higher score if buildings are vertically aligned
        vertical_alignment = 1 - abs(c1.y - c2.y) / max(poly.bounds[3] - poly.bounds[1], other_poly.bounds[3] - other_poly.bounds[1])
        return vertical_alignment
    elif dx == 0 and dy != 0:  # Vertical boundary
        # Higher score if buildings are horizontally aligned
        horizontal_alignment = 1 - abs(c1.x - c2.x) / max(poly.bounds[2] - poly.bounds[0], other_poly.bounds[2] - other_poly.bounds[0])
        return horizontal_alignment
    else:  # Diagonal boundary (corner touching)
        # For diagonal, use the minimum of horizontal and vertical alignment
        vertical_alignment = 1 - abs(c1.y - c2.y) / max(poly.bounds[3] - poly.bounds[1], other_poly.bounds[3] - other_poly.bounds[1])
        horizontal_alignment = 1 - abs(c1.x - c2.x) / max(poly.bounds[2] - poly.bounds[0], other_poly.bounds[2] - other_poly.bounds[0])
        return min(vertical_alignment, horizontal_alignment)

def merge_overlapping_detections(individual_detections, 
                                 iou_thresh, 
                                 touch_enabled, 
                                 min_edge_distance_deg):
    """
    Merges overlapping and proximal detections using a multi-phase approach
    with enhanced scoring that considers tile boundaries.
    """
    if not individual_detections:
        return []

    # Pre-compute long axes for each polygon
    long_axes = []
    for det in individual_detections:
        polygon = det['polygon']
        long_axis = get_long_axis(polygon)
        long_axes.append(long_axis)

    # Find all potential connections between detections from different tiles
    # Format: (i, j, score, connection_type)
    # connection_type: 1=high_confidence, 2=boundary_related, 3=other
    all_connections = []
    
    for i in range(len(individual_detections)):
        poly_i_obj = individual_detections[i]
        tile_i_id = poly_i_obj.get('tile_id', 'UNKNOWN')
        
        for j in range(i+1, len(individual_detections)):
            poly_j_obj = individual_detections[j]
            tile_j_id = poly_j_obj.get('tile_id', 'UNKNOWN')
            
            # Only process if from different tiles
            if tile_i_id != tile_j_id and tile_i_id != 'UNKNOWN' and tile_j_id != 'UNKNOWN':
                # Check geometric relationships
                poly_i = poly_i_obj['polygon']
                poly_j = poly_j_obj['polygon']
                
                # Calculate center-to-center distance
                center_i = poly_i.centroid
                center_j = poly_j.centroid
                center_distance = center_i.distance(center_j)
                
                # Calculate boundary proximity score
                boundary_score = calculate_boundary_proximity(
                    poly_i, tile_i_id, poly_j, tile_j_id
                )
                
                # Calculate axis alignment with reduced weight
                axis_alignment = calculate_axis_alignment(long_axes[i], long_axes[j])
                alignment_weight = 5.0  # Reduced from 10.0
                alignment_factor = axis_alignment ** alignment_weight
                
                # Initialize connection variables
                related_geometrically = False
                connection_score = float('inf')
                connection_type = 3  # Default: other
                
                # 1. Check high-confidence connections (IoU)
                intersection = poly_i.intersection(poly_j).area
                if intersection > 0:
                    union_val = poly_i.area + poly_j.area - intersection
                    iou = intersection / union_val if union_val > 0 else 0
                    if iou > iou_thresh:
                        related_geometrically = True
                        connection_type = 1  # High confidence
                        # Score: negative IoU (higher IoU = better score)
                        connection_score = -iou
                
                # 2. Check boundary-related connections
                if not related_geometrically and boundary_score > 0.7:  # High boundary score
                    # For touching polygons near boundaries
                    if touch_enabled and poly_i.touches(poly_j):
                        related_geometrically = True
                        connection_type = 2  # Boundary related
                        # Score based on boundary score and axis alignment
                        connection_score = -boundary_score * alignment_factor
                    
                    # For proximal polygons near boundaries
                    elif min_edge_distance_deg > 0:
                        edge_dist = poly_i.distance(poly_j)
                        if 0 < edge_dist < min_edge_distance_deg:
                            related_geometrically = True
                            connection_type = 2  # Boundary related
                            # Score based on normalized edge distance and boundary score
                            norm_dist = edge_dist / min_edge_distance_deg
                            connection_score = norm_dist - boundary_score
                
                # 3. Check other geometric relationships
                if not related_geometrically:
                    # Check touches
                    if touch_enabled and poly_i.touches(poly_j):
                        related_geometrically = True
                        connection_type = 3  # Other
                        # Score based on touch length and alignment
                        boundary_i = poly_i.boundary
                        boundary_j = poly_j.boundary
                        touch_length = boundary_i.intersection(boundary_j).length
                        connection_score = -touch_length * alignment_factor * 0.5  # Reduced weight
                    
                    # Check edge distance
                    elif min_edge_distance_deg > 0:
                        edge_dist = poly_i.distance(poly_j)
                        if 0 < edge_dist < min_edge_distance_deg:
                            related_geometrically = True
                            connection_type = 3  # Other
                            # Score based on distance, alignment, and center distance
                            epsilon = 1e-10  # Avoid division by zero
                            # Higher center distance = worse score
                            connection_score = edge_dist * (1 + center_distance) / (alignment_factor + epsilon)
                
                # Add valid connection to list
                if related_geometrically:
                    all_connections.append((i, j, connection_score, connection_type))
    
    # Multi-phase pairing process
    used_detections = set()
    best_pairs = []
    
    # Process connections in phases by connection_type (1, 2, then 3)
    for phase in [1, 2, 3]:
        # Filter connections for current phase
        phase_connections = [c for c in all_connections if c[3] == phase]
        # Sort by score (lower is better)
        phase_connections.sort(key=lambda x: x[2])
        
        # Select best pairs for this phase
        for i, j, _, _ in phase_connections:
            if i not in used_detections and j not in used_detections:
                best_pairs.append((i, j))
                used_detections.add(i)
                used_detections.add(j)
    
    # Create components from the best pairs
    components = {}
    for i in range(len(individual_detections)):
        components[i] = {i}
    
    # Merge components based on best pairs
    for i, j in best_pairs:
        components[i].add(j)
        components.pop(j, None)
    
    # Create merged buildings
    merged_buildings = []
    for comp_id, indices in components.items():
        group_polygons = [individual_detections[idx]['polygon'] for idx in indices]
        group_confidences = [individual_detections[idx]['confidence'] for idx in indices]
        group_ids = [individual_detections[idx]['id'] for idx in indices]
        
        if group_polygons:
            combined_polygon_shape = unary_union(group_polygons)
            merged_envelope = combined_polygon_shape.envelope
            merged_buildings.append({
                'id': f"merged_{len(merged_buildings)}",
                'polygon': merged_envelope, 
                'coordinates': list(merged_envelope.exterior.coords),
                'confidence': max(group_confidences) if group_confidences else 0.0,
                'original_ids': sorted(list(group_ids)) 
            })
    
    return merged_buildings

def process_tile_batch(tile_batch, model, conf):
    """Process a batch of tiles and return their detection results"""
    batch_results = []
    
    for tile in tile_batch:
        try:
            # Get tile image (in memory)
            tile_image = get_tile_image(tile)
            
            # Create a temporary file for detection
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
                # Save the image to the temporary file
                tile_image.save(temp_path, format='PNG')
            
            try:
                # Detect buildings using the temporary file path
                results, img = detect_buildings(model, temp_path, conf=conf)
                
                # Process detection results
                boxes, confidences, class_ids = process_tile_detections(results)
                
                # Add to results
                tile_bounds = get_tile_bounds(tile)
                tile_detections = {
                    'tile': f"{tile.z}/{tile.x}/{tile.y}",
                    'bounds': tile_bounds,
                    'detections': len(boxes),
                    'boxes': boxes.tolist() if len(boxes) > 0 else [],
                    'confidences': confidences.tolist() if len(confidences) > 0 else [],
                    'class_ids': class_ids.tolist() if len(class_ids) > 0 else [],
                    'image': tile_image  # Store the image in memory
                }
                batch_results.append(tile_detections)
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            
        except Exception as e:
            print(f"Error processing tile {tile}: {e}")
    
    return batch_results

def create_batches(items, batch_size):
    """Split a list of items into batches of the specified size"""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

def detect_buildings_in_polygon(model, geojson_path, output_dir="polygon_detection_results", zoom=18, conf=0.25, batch_size=5,
                                enable_merging=True,
                                merge_iou_threshold=1.1, 
                                merge_touch_enabled=True, 
                                merge_min_edge_distance_deg=0.00001 # Diubah dari merge_centroid_proximity_deg
                                ):
    """
    Detect buildings within a polygon defined in a GeoJSON file using optimized batch processing.
    Optionally merges fragmented detections.
    
    Args:
        model: Loaded YOLOv8 model
        geojson_path: Path to the GeoJSON file
        output_dir: Directory to save detection results
        zoom: Zoom level for tiles
        conf: Confidence threshold for individual detections
        batch_size: Number of tiles per batch
        enable_merging: Whether to perform post-processing to merge fragmented detections.
        merge_iou_threshold: IoU threshold for considering detections part of the same group for merging.
        merge_touch_enabled: Whether touching polygons are considered for merging.
        merge_min_edge_distance_deg: Max edge distance (degrees) for merging non-touching, non-overlapping detections.
        
    Returns:
        Dictionary with detection results and execution time
    """
    # Fixed optimal worker count based on previous benchmarks
    num_workers = 2
    
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load GeoJSON
    geojson_data = load_geojson(geojson_path)
    
    # Extract polygon
    polygon = extract_polygon(geojson_data)
    
    # Get tiles that intersect with the polygon
    tiles = get_tiles_for_polygon(polygon, zoom=zoom)
    print(f"Found {len(tiles)} tiles that intersect with the polygon")
    
    # Create batches of tiles
    tile_batches = create_batches(tiles, batch_size)
    print(f"Created {len(tile_batches)} batches with batch size {batch_size}")
    
    # Process batches in parallel with 2 workers
    all_detections_raw_per_tile = [] # Renamed from all_detections
    total_buildings = 0
    
    # Create a partial function with fixed arguments
    process_batch = partial(process_tile_batch, model=model, conf=conf)
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all batches and get a list of futures
        future_to_batch = {executor.submit(process_batch, batch): i for i, batch in enumerate(tile_batches)}
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_batch), 
                          total=len(tile_batches), 
                          desc=f"Processing tile batches"):
            batch_idx = future_to_batch[future]
            try:
                batch_detections = future.result()
                if batch_detections:
                    all_detections_raw_per_tile.extend(batch_detections) # Store raw per-tile results
                    # total_buildings += sum(d['detections'] for d in batch_detections) # Don't sum here if merging
            except Exception as exc:
                print(f"Batch {batch_idx} generated an exception: {exc}")
    
    results_path = os.path.join(output_dir, "detection_results.json")
    
    final_detections_for_json = []
    final_merged_shapely_objects = [] # For visualization
    total_buildings_final = 0

    if enable_merging:
        print(f"Converting {sum(len(t['boxes']) for t in all_detections_raw_per_tile)} raw detections to Shapely objects for merging...")
        individual_shapely_detections = convert_tile_detections_to_shapely_polygons(all_detections_raw_per_tile)
        
        print(f"Merging {len(individual_shapely_detections)} individual detections...")
        merged_buildings_list = merge_overlapping_detections(
            individual_shapely_detections,
            merge_iou_threshold,
            merge_touch_enabled,
            merge_min_edge_distance_deg
        )
        total_buildings_final = len(merged_buildings_list)
        print(f"Total buildings after merging: {total_buildings_final}")

        for mb in merged_buildings_list:
            final_detections_for_json.append({
                'id': mb['id'],
                'coordinates': mb['coordinates'], # Already in geojson-friendly format
                'confidence': mb['confidence'],
                'original_count': len(mb['original_ids'])
            })
            final_merged_shapely_objects.append(mb) # Contains the 'polygon' Shapely object
    else:
        # If merging is disabled, use the old logic (or adapt convert_tile_detections_to_shapely_polygons)
        # For now, let's just use the converted individual detections if merging is off
        # This part might need refinement if non-merged output is still desired in the old per-tile format
        print("Merging disabled. Processing individual detections for output.")
        individual_shapely_detections = convert_tile_detections_to_shapely_polygons(all_detections_raw_per_tile)
        total_buildings_final = len(individual_shapely_detections)
        for det in individual_shapely_detections:
             final_detections_for_json.append({
                'id': det['id'],
                'coordinates': list(det['polygon'].exterior.coords),
                'confidence': det['confidence']
            })
             final_merged_shapely_objects.append(det) # Contains the 'polygon' Shapely object


    # Create a copy of the results without the images for JSON serialization
    json_results_payload = {
        'total_buildings': total_buildings_final,
        'total_tiles': len(tiles), # This remains the number of processed tiles
        'zoom': zoom,
        'confidence_threshold': conf, # Original detection confidence
        'merging_enabled': enable_merging,
        'merge_iou_threshold': merge_iou_threshold if enable_merging else None,
        'merge_touch_enabled': merge_touch_enabled if enable_merging else None,
        'merge_min_edge_distance_deg': merge_min_edge_distance_deg if enable_merging else None,
        'detections': final_detections_for_json # This is now a list of merged buildings
    }
    
    with open(results_path, 'w') as f:
        json.dump(json_results_payload, f, indent=2)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Add execution time to results
    json_results_payload['execution_time'] = execution_time
    
    print(f"Processing completed in {execution_time:.2f} seconds")
    print(f"Detection results saved to {results_path}")
    print(f"Total buildings detected: {total_buildings_final}") # Use final count
    
    visualization_path = os.path.join(output_dir, "polygon_visualization.png")
    
    visualization_results_data = {
        'total_buildings': total_buildings_final,
        'total_tiles': len(tiles),
        'zoom': zoom,
        'confidence_threshold': conf, # Original detection confidence for context
        'merged_detections_shapely': final_merged_shapely_objects, 
        'raw_tile_detections_for_background': all_detections_raw_per_tile 
    }
    
    print("Visualizing merged detections...")
    visualize_polygon_detections(
        geojson_path, 
        visualization_results_data, 
        visualization_path, 
        iou_threshold=1.1,  # Threshold for highlighting overlaps *between merged* buildings
        max_proximity_distance=0.00005 # Max distance for highlighting *between merged* buildings (adjust as needed)
    )

    # Save buildings to JSON - this will use json_results_payload
    buildings_json_path = os.path.join(output_dir, "buildings.json")
    save_buildings_to_json(json_results_payload, buildings_json_path) # Pass the new payload
    
    return json_results_payload

if __name__ == "__main__":
    # Check if shapely and mercantile are installed
    try:
        import shapely
        import mercantile
    except ImportError:
        print("This script requires shapely and mercantile packages.")
        print("Please install them with: pip install shapely mercantile")
        sys.exit(1)
    
    # Path to the model
    model_path = "../best.pt"
    
    # Create example GeoJSON if needed
    if len(sys.argv) > 1:
        geojson_path = sys.argv[1]
    else:
        print("No GeoJSON file provided, creating an example...")
        geojson_path = create_example_geojson()
    
    # Output directory
    output_dir = "polygon_detection_results"
    
    # Set batch size (can be specified as command line argument)
    batch_size = 5
    if len(sys.argv) > 2:
        try:
            batch_size = int(sys.argv[2])
        except:
            pass # Keep default if conversion fails
    
    # Load the YOLOv8 model
    model = load_model(model_path)
    
    # Detect buildings in the polygon with optimized batch processing
    # Adjust merging parameters here to be more conservative:
    results = detect_buildings_in_polygon(
        model, geojson_path, output_dir, zoom=18, conf=0.25, batch_size=batch_size,
        enable_merging=True, 
        merge_iou_threshold=1.1,      # Increased from 0.05
        merge_touch_enabled=True,    # Changed from True
        merge_min_edge_distance_deg=0.000001 # Kriteria baru, ~1.1 meter. Set ke 0 jika tidak ingin aktif.
    )
    
    print("\nDetection Summary:")
    print(f"Total buildings detected: {results['total_buildings']}")
    print(f"Total tiles processed: {results['total_tiles']}")
    print(f"Execution time: {results['execution_time']:.2f} seconds")
    print(f"Results saved to {output_dir}/detection_results.json")
    print(f"Visualization saved to {output_dir}/polygon_visualization.png")
    print(f"Building data saved to {output_dir}/buildings.json")