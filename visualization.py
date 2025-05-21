import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon, shape, box
from shapely.ops import unary_union

from geojson_utils import load_geojson, extract_polygon
from tile_utils import create_stitched_image

# Helper to convert Shapely Polygon to Matplotlib Patch
def shapely_polygon_to_mpl_patch(shapely_polygon, **kwargs):
    """Convert a Shapely Polygon to a Matplotlib Polygon patch."""
    # For Polygons with no interior rings (holes)
    if shapely_polygon.interiors:
        # More complex polygons with holes require Path and PathPatch
        # This is a simplified example assuming simple polygons (envelopes from merge)
        print("Warning: Polygon has interiors (holes), visualization might be simplified.")
        # Fallback to exterior for simplicity, or implement full Path conversion
        path_data = [(patches.Path.MOVETO, list(shapely_polygon.exterior.coords)[0])]
        path_data.extend([(patches.Path.LINETO, pt) for pt in list(shapely_polygon.exterior.coords)[1:]])
        path_data.append((patches.Path.CLOSEPOLY, list(shapely_polygon.exterior.coords)[0]))
        
        # Handle interiors (holes)
        for interior in shapely_polygon.interiors:
            path_data.append((patches.Path.MOVETO, list(interior.coords)[0]))
            path_data.extend([(patches.Path.LINETO, pt) for pt in list(interior.coords)[1:]])
            path_data.append((patches.Path.CLOSEPOLY, list(interior.coords)[0]))
            
        path = patches.Path(np.array([v[1] for v in path_data]), np.array([c[0] for c in path_data]))
        return patches.PathPatch(path, **kwargs)
    else:
        # Simple polygon without holes
        return patches.Polygon(np.array(list(shapely_polygon.exterior.coords)), closed=True, **kwargs)

def visualize_polygon_detections(geojson_path, results_data, output_path=None, 
                                 iou_threshold=0.01, # Threshold for highlighting overlaps between *merged* buildings
                                 max_proximity_distance=0.0): # Max distance for *merged* buildings
    """
    Visualize merged building detections across the entire GeoJSON area on a single map.
    
    Args:
        geojson_path: Path to the GeoJSON file.
        results_data: Dictionary containing detection results, structured for merged detections.
                      Expected keys: 'merged_detections_shapely' (list of dicts with 'polygon', 'confidence'),
                                     'raw_tile_detections_for_background' (optional, for stitched background),
                                     'total_buildings', 'total_tiles', etc.
        output_path: Path to save the visualization (optional).
        iou_threshold: IoU threshold for highlighting overlaps *between merged buildings*.
        max_proximity_distance: Max distance for highlighting proximity *between merged buildings*.
    """
    geojson_data_loaded = load_geojson(geojson_path)
    polygon_area = extract_polygon(geojson_data_loaded)
    
    fig, ax = plt.subplots(1, figsize=(15, 15))
    minx, miny, maxx, maxy = polygon_area.bounds

    # 1. Create stitched image background (if raw tile data is available)
    raw_tile_data = results_data.get('raw_tile_detections_for_background')
    if raw_tile_data and any(d.get('image') for d in raw_tile_data):
        try:
            # create_stitched_image expects a list of dicts, each having 'image' and 'bounds'
            # Ensure raw_tile_data provides this structure if it exists
            # The 'detections' key in raw_tile_data was the old structure, which should work if passed correctly
            stitched_image, transform_params = create_stitched_image(raw_tile_data)
            ax.imshow(stitched_image, extent=[
                transform_params['min_west'], 
                transform_params['min_west'] + transform_params['width_deg'],
                transform_params['max_north'] - transform_params['height_deg'],
                transform_params['max_north']
            ])
            ax.set_xlim(transform_params['min_west'], transform_params['min_west'] + transform_params['width_deg'])
            ax.set_ylim(transform_params['max_north'] - transform_params['height_deg'], transform_params['max_north'])
        except Exception as e:
            print(f"Warning: Failed to create stitched image for background: {e}")
            print("Falling back to GeoJSON polygon bounds.")
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
    else:
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

    # 2. Plot the GeoJSON polygon area
    if geojson_data_loaded['type'] == 'FeatureCollection':
        for i, feature in enumerate(geojson_data_loaded['features']):
            geom = shape(feature['geometry'])
            if geom.geom_type in ['Polygon', 'MultiPolygon']:
                color = plt.cm.tab20(i % 20)
                if geom.geom_type == 'Polygon':
                    x_coords, y_coords = geom.exterior.xy
                    ax.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.7)
                elif geom.geom_type == 'MultiPolygon':
                    for poly_part in geom.geoms:
                        x_coords, y_coords = poly_part.exterior.xy
                        ax.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.7)
                if 'properties' in feature and 'name' in feature['properties']:
                     # Simplified text placement
                    ax.text(geom.centroid.x, geom.centroid.y, feature['properties']['name'], 
                            fontsize=10, ha='center', va='center', 
                            bbox=dict(facecolor='white', alpha=0.5, pad=0.1))
    else: # Single Polygon
        x_coords, y_coords = polygon_area.exterior.xy
        ax.plot(x_coords, y_coords, color='blue', linewidth=2, alpha=0.7)

    # 3. Plot tile boundaries (if raw tile data is available)
    if raw_tile_data:
        for tile_info in raw_tile_data:
            bounds = tile_info.get('bounds')
            if bounds:
                tile_shapely = box(bounds[0], bounds[1], bounds[2], bounds[3])
                x_coords, y_coords = tile_shapely.exterior.xy
                ax.plot(x_coords, y_coords, color='gray', linewidth=0.5, alpha=0.3)

    # 4. Plot merged building detections
    merged_detections = results_data.get('merged_detections_shapely', [])
    
    building_patches_mpl = [] # For Matplotlib PatchCollection
    shapely_polygons_for_eval = [] # List of Shapely polygons for IoU/proximity eval
    confidence_values_for_color = []
    building_ids_for_text = []

    for i, det_data in enumerate(merged_detections):
        shapely_poly = det_data.get('polygon') # This is the merged Shapely Polygon
        confidence = det_data.get('confidence', 0.5)
        det_id = det_data.get('id', f"b_{i}")

        if not shapely_poly or not isinstance(shapely_poly, Polygon) or shapely_poly.is_empty:
            continue

        # Create a Matplotlib patch from the Shapely polygon
        # edgecolor will be set by PatchCollection or individually for intersecting ones
        mpl_patch = shapely_polygon_to_mpl_patch(shapely_poly, linewidth=1, facecolor='none', edgecolor='none')
        building_patches_mpl.append(mpl_patch)
        shapely_polygons_for_eval.append(shapely_poly)
        confidence_values_for_color.append(confidence)
        building_ids_for_text.append(det_id)

        # Plot centroid marker and ID for each merged building
        centroid = shapely_poly.centroid
        ax.plot(centroid.x, centroid.y, marker='o', color='darkblue', markersize=2, alpha=0.6)
        ax.text(centroid.x, centroid.y, str(det_id), color='black', fontsize=5,
                ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.2, pad=0.1, edgecolor='none'))

    # 5. Identify intersecting/proximal *merged* buildings for highlighting
    num_merged_boxes = len(shapely_polygons_for_eval)
    is_highlighted_red = [False] * num_merged_boxes

    if num_merged_boxes > 1:
        for i in range(num_merged_boxes):
            for j in range(i + 1, num_merged_boxes):
                poly_i = shapely_polygons_for_eval[i]
                poly_j = shapely_polygons_for_eval[j]
                
                marked_by_iou = False
                marked_by_touching = False

                intersection_val = poly_i.intersection(poly_j).area
                if intersection_val > 0:
                    union_val = poly_i.union(poly_j).area 
                    if union_val > 0:
                        iou_calc = intersection_val / union_val
                        if iou_calc > iou_threshold:
                            is_highlighted_red[i] = True
                            is_highlighted_red[j] = True
                            marked_by_iou = True
                    elif intersection_val > 0: # e.g. identical polygons
                        is_highlighted_red[i] = True
                        is_highlighted_red[j] = True
                        marked_by_iou = True
                
                if not marked_by_iou:
                    if poly_i.touches(poly_j):
                        is_highlighted_red[i] = True
                        is_highlighted_red[j] = True
                        marked_by_touching = True
                
                if not marked_by_iou and not marked_by_touching and max_proximity_distance > 0:
                    dist = poly_i.centroid.distance(poly_j.centroid)
                    if dist < max_proximity_distance:
                        is_highlighted_red[i] = True
                        is_highlighted_red[j] = True

    # Separate patches for plotting based on highlighting
    highlighted_patches = []
    normal_patches = []
    normal_confidences = []

    for i in range(num_merged_boxes):
        if is_highlighted_red[i]:
            highlighted_patches.append(building_patches_mpl[i])
        else:
            normal_patches.append(building_patches_mpl[i])
            normal_confidences.append(confidence_values_for_color[i])

    # Add highlighted patches (red)
    if highlighted_patches:
        highlight_collection = PatchCollection(
            highlighted_patches, facecolor='red', alpha=0.6, edgecolor='darkred', linewidth=0.7
        )
        ax.add_collection(highlight_collection)

    # Add normal patches (color by confidence)
    if normal_patches:
        normal_collection = PatchCollection(
            normal_patches, cmap=plt.cm.viridis, alpha=0.65, edgecolor='black', linewidth=0.5
        )
        normal_collection.set_array(np.array(normal_confidences))
        ax.add_collection(normal_collection)
        cbar = plt.colorbar(normal_collection, ax=ax, shrink=0.6)
        cbar.set_label('Confidence Score (Non-Highlighted Merged Buildings)')

    # 6. Set title and labels
    ax.set_title(f"Merged Building Detections ({results_data.get('total_buildings', 0)} buildings)", fontsize=16)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    
    # 7. Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='blue', lw=2, label='GeoJSON Area'),
    ]
    if raw_tile_data: # Only add if tile boundaries were potentially plotted
        legend_elements.append(plt.Line2D([0], [0], color='gray', lw=0.5, label='Tile Boundaries'))
    if normal_patches:
        legend_elements.append(patches.Patch(facecolor=plt.cm.viridis(0.5), edgecolor='black', label='Merged Building (Confidence)'))
    if highlighted_patches:
        legend_elements.append(patches.Patch(facecolor='red', edgecolor='darkred', label='Highlighted Merged Building (Overlap/Proximity)'))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize='small')
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout(pad=1.5)

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Visualization of merged detections saved to {output_path}")
        plt.close()
    else:
        plt.show() 