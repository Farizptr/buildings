import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon, shape

from geojson_utils import load_geojson, extract_polygon
from tile_utils import create_stitched_image

def visualize_polygon_detections(geojson_path, results_data, output_path=None):
    """
    Visualize all building detections across the entire GeoJSON area on a single map
    
    Args:
        geojson_path: Path to the GeoJSON file
        results_data: Detection results from detect_buildings_in_polygon
        output_path: Path to save the visualization (optional)
        
    Returns:
        None
    """
    # Load GeoJSON
    geojson_data = load_geojson(geojson_path)
    
    # Extract polygon
    polygon = extract_polygon(geojson_data)
    
    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=(15, 15))
    
    # Set bounds based on polygon
    minx, miny, maxx, maxy = polygon.bounds
    
    # Create a stitched image as background if images are available
    if 'detections' in results_data and results_data['detections'] and 'image' in results_data['detections'][0]:
        try:
            # Create stitched image from in-memory tiles
            stitched_image, transform_params = create_stitched_image(results_data['detections'])
            
            # Display the stitched image as background
            ax.imshow(stitched_image, extent=[
                transform_params['min_west'], 
                transform_params['min_west'] + transform_params['width_deg'],
                transform_params['max_north'] - transform_params['height_deg'],
                transform_params['max_north']
            ])
            
            # Set bounds based on the stitched image
            ax.set_xlim(transform_params['min_west'], transform_params['min_west'] + transform_params['width_deg'])
            ax.set_ylim(transform_params['max_north'] - transform_params['height_deg'], transform_params['max_north'])
        except Exception as e:
            print(f"Warning: Failed to create stitched image: {e}")
            print("Falling back to standard visualization")
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
    else:
        # Standard visualization without background image
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
    
    # Plot the polygon(s)
    if geojson_data['type'] == 'FeatureCollection':
        for i, feature in enumerate(geojson_data['features']):
            geom = shape(feature['geometry'])
            if geom.geom_type in ['Polygon', 'MultiPolygon']:
                # Get a color from the tab20 colormap
                color = plt.cm.tab20(i % 20)
                # Plot the polygon
                x, y = geom.exterior.xy
                ax.plot(x, y, color=color, linewidth=2, alpha=0.7)
                # Add a label if name is available
                if 'properties' in feature and 'name' in feature['properties']:
                    ax.text(np.mean(x), np.mean(y), feature['properties']['name'], 
                            fontsize=12, ha='center', va='center', 
                            bbox=dict(facecolor='white', alpha=0.7))
    else:
        # Plot a single polygon
        x, y = polygon.exterior.xy
        ax.plot(x, y, color='blue', linewidth=2, alpha=0.7)
    
    # Plot tile boundaries
    for tile_detection in results_data['detections']:
        bounds = tile_detection['bounds']
        tile_polygon = Polygon([
            (bounds[0], bounds[1]),  # SW
            (bounds[2], bounds[1]),  # SE
            (bounds[2], bounds[3]),  # NE
            (bounds[0], bounds[3]),  # NW
            (bounds[0], bounds[1])   # SW (close the polygon)
        ])
        x, y = tile_polygon.exterior.xy
        ax.plot(x, y, color='gray', linewidth=0.5, alpha=0.3)
    
    # Plot building detections
    building_patches = []
    confidence_values = []
    building_id_counter = 1  # Initialize building ID counter
    
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
            
            # Calculate center point for centroid and ID
            center_lon = (geo_x1 + geo_x2) / 2
            center_lat = (geo_y1 + geo_y2) / 2

            # Plot centroid marker
            ax.plot(center_lon, center_lat, marker='o', color='blue', markersize=2, alpha=0.7)

            # Add building ID text
            ax.text(center_lon, center_lat, str(building_id_counter), color='black', fontsize=6,
                    ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.3, pad=0.1, edgecolor='none'))
            
            # Create rectangle
            rect = patches.Rectangle(
                (geo_x1, geo_y2),  # Lower left corner (x, y)
                geo_x2 - geo_x1,    # Width
                geo_y1 - geo_y2,    # Height
                linewidth=1,
                edgecolor='none',
                facecolor='none'
            )
            
            building_patches.append(rect)
            confidence_values.append(confidences[i] if i < len(confidences) else 0.5)
            building_id_counter += 1 # Increment for the next building
    
    # Add building patches to the plot with color based on confidence
    if building_patches:
        # Create a PatchCollection for better performance
        building_collection = PatchCollection(
            building_patches, 
            cmap=plt.cm.viridis,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )
        
        # Set the color array based on confidence values
        building_collection.set_array(np.array(confidence_values))
        
        # Add the collection to the plot
        ax.add_collection(building_collection)
        
        # Add a colorbar
        cbar = plt.colorbar(building_collection, ax=ax)
        cbar.set_label('Confidence Score')
    
    # Set title and labels
    ax.set_title(f'Building Detections in GeoJSON Area ({len(building_patches)} buildings)', fontsize=16)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    # Add a legend
    ax.plot([], [], color='blue', linewidth=2, label='GeoJSON Polygon')
    ax.plot([], [], color='gray', linewidth=0.5, label='Tile Boundaries')
    ax.legend(loc='upper right')
    
    # Save the figure if output path is provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Visualization saved to {output_path}")
        plt.close()  # Close the figure to avoid displaying it
    else:
        # Only show the plot if no output path is provided
        plt.tight_layout()
        plt.show() 