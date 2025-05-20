import mercantile
import requests
from io import BytesIO
from PIL import Image
from shapely.geometry import Polygon

def get_tile_bounds(tile):
    """
    Get the bounds of a tile in [west, south, east, north] format
    
    Args:
        tile: A mercantile Tile object
        
    Returns:
        Bounds as [west, south, east, north]
    """
    bounds = mercantile.bounds(tile)
    return [bounds.west, bounds.south, bounds.east, bounds.north]

def get_tiles_for_polygon(polygon, zoom=18):
    """
    Get all tiles that intersect with a polygon
    
    Args:
        polygon: Shapely polygon
        zoom: Zoom level
        
    Returns:
        List of mercantile Tile objects
    """
    # Get the bounds of the polygon
    minx, miny, maxx, maxy = polygon.bounds
    
    # Get all tiles that intersect with the bounds
    tiles = list(mercantile.tiles(minx, miny, maxx, maxy, zoom))
    
    # Filter tiles to only those that intersect with the polygon
    intersecting_tiles = []
    for tile in tiles:
        tile_bounds = get_tile_bounds(tile)
        tile_polygon = Polygon([
            (tile_bounds[0], tile_bounds[1]),  # SW
            (tile_bounds[2], tile_bounds[1]),  # SE
            (tile_bounds[2], tile_bounds[3]),  # NE
            (tile_bounds[0], tile_bounds[3]),  # NW
            (tile_bounds[0], tile_bounds[1])   # SW (close the polygon)
        ])
        
        if polygon.intersects(tile_polygon):
            intersecting_tiles.append(tile)
    
    return intersecting_tiles

def get_tile_image(tile):
    """
    Get an OSM tile image
    
    Args:
        tile: A mercantile Tile object
        
    Returns:
        PIL Image object of the tile
    """
    # Create URL for the tile
    url = f"https://tile.openstreetmap.org/{tile.z}/{tile.x}/{tile.y}.png"
    
    # Download tile
    headers = {'User-Agent': 'BuildingDetectionBot/1.0'}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        # Convert response to RGB image
        img = Image.open(BytesIO(response.content)).convert('RGB')
        return img
    else:
        raise Exception(f"Failed to download tile: {response.status_code}")

def process_tile_detections(results):
    """
    Process detection results without visualizing or saving images
    
    Args:
        results: YOLOv8 model detection results
        
    Returns:
        Tuple of (boxes, confidences, class_ids)
    """
    # Get detections (boxes, confidence scores, and class IDs)
    boxes = results.boxes.xyxy.cpu().numpy()
    confidences = results.boxes.conf.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy()
    
    return boxes, confidences, class_ids

def create_stitched_image(tile_detections):
    """
    Create a stitched image from individual tile images stored in memory
    
    Args:
        tile_detections: List of tile detection results with bounds information and images
        
    Returns:
        Tuple of (stitched_image, transform_params)
        - stitched_image: PIL Image of the stitched tiles
        - transform_params: Parameters for transforming geo coordinates to pixel coordinates
    """
    if not tile_detections:
        raise ValueError("No tile detections provided")
    
    # Get the bounds of all tiles
    all_bounds = [td['bounds'] for td in tile_detections]
    
    # Calculate the overall bounds
    min_west = min(bounds[0] for bounds in all_bounds)
    min_south = min(bounds[1] for bounds in all_bounds)
    max_east = max(bounds[2] for bounds in all_bounds)
    max_north = max(bounds[3] for bounds in all_bounds)
    
    # Calculate the width and height in degrees
    width_deg = max_east - min_west
    height_deg = max_north - min_south
    
    # Assume all tiles are 256x256 pixels
    tile_size = 256
    
    # Calculate the number of tiles in each direction
    num_tiles_x = len(set(bounds[0] for bounds in all_bounds))
    num_tiles_y = len(set(bounds[3] for bounds in all_bounds))
    
    # Calculate the size of the stitched image
    width_px = num_tiles_x * tile_size
    height_px = num_tiles_y * tile_size
    
    # Create a blank image
    stitched_image = Image.new('RGB', (width_px, height_px), (255, 255, 255))
    
    # Place each tile in the stitched image
    for td in tile_detections:
        # Get the tile bounds
        west, south, east, north = td['bounds']
        
        # Get the tile image from memory
        if 'image' not in td or td['image'] is None:
            print(f"Warning: Tile image for {td['tile']} not found, skipping")
            continue
        
        tile_image = td['image']
        
        # Calculate the position in the stitched image
        x_pos = int((west - min_west) / width_deg * width_px)
        y_pos = int((max_north - north) / height_deg * height_px)
        
        # Paste the tile image
        stitched_image.paste(tile_image, (x_pos, y_pos))
    
    # Create transform parameters for converting geo coordinates to pixel coordinates
    transform_params = {
        'min_west': min_west,
        'max_north': max_north,
        'width_deg': width_deg,
        'height_deg': height_deg,
        'width_px': width_px,
        'height_px': height_px
    }
    
    return stitched_image, transform_params 