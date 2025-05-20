import json
from shapely.geometry import shape
from shapely.ops import unary_union

def load_geojson(geojson_path):
    """
    Load a GeoJSON file
    
    Args:
        geojson_path: Path to the GeoJSON file
        
    Returns:
        GeoJSON data as a Python dictionary
    """
    try:
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
        print(f"GeoJSON loaded successfully from {geojson_path}")
        return geojson_data
    except Exception as e:
        print(f"Error loading GeoJSON: {e}")
        raise

def extract_polygon(geojson_data):
    """
    Extract polygon(s) from GeoJSON data
    
    Args:
        geojson_data: GeoJSON data as a Python dictionary
        
    Returns:
        Shapely polygon or multipolygon
    """
    polygons = []
    
    # Handle different GeoJSON types
    if geojson_data['type'] == 'FeatureCollection':
        for feature in geojson_data['features']:
            geom = shape(feature['geometry'])
            if geom.geom_type in ['Polygon', 'MultiPolygon']:
                polygons.append(geom)
    elif geojson_data['type'] == 'Feature':
        geom = shape(geojson_data['geometry'])
        if geom.geom_type in ['Polygon', 'MultiPolygon']:
            polygons.append(geom)
    else:
        # Direct geometry
        geom = shape(geojson_data)
        if geom.geom_type in ['Polygon', 'MultiPolygon']:
            polygons.append(geom)
    
    if not polygons:
        raise ValueError("No valid polygons found in GeoJSON")
    
    # Combine all polygons into one
    if len(polygons) == 1:
        return polygons[0]
    else:
        return unary_union(polygons)

def create_example_geojson(output_path="example_area.geojson"):
    """
    Create an example GeoJSON file with a polygon
    
    Args:
        output_path: Path to save the GeoJSON file
        
    Returns:
        Path to the created GeoJSON file
    """
    # Example polygon (Jakarta area)
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "Jakarta Example Area"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [106.8456, -6.2088],  # Center point
                            [106.8476, -6.2088],  # East
                            [106.8476, -6.2068],  # Northeast
                            [106.8456, -6.2068],  # North
                            [106.8436, -6.2068],  # Northwest
                            [106.8436, -6.2088],  # West
                            [106.8436, -6.2108],  # Southwest
                            [106.8456, -6.2108],  # South
                            [106.8476, -6.2108],  # Southeast
                            [106.8476, -6.2088],  # Back to East
                            [106.8456, -6.2088]   # Close the polygon
                        ]
                    ]
                }
            }
        ]
    }
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)
    
    print(f"Example GeoJSON saved to {output_path}")
    return output_path 