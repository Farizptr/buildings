import folium
import json

# Load your buildings
with open('polygon_detection_results/buildings.json', 'r') as f:
    buildings_data = json.load(f)

features = buildings_data["features"]
total_buildings = buildings_data["total_buildings"]

# Create a map centered on the first building's centroid
first_building_coords = features[0]["geometry"]["coordinates"][0]
# Calculate centroid of first building
first_building_lats = [coord[1] for coord in first_building_coords]
first_building_lons = [coord[0] for coord in first_building_coords]
center_lat = sum(first_building_lats) / len(first_building_lats)
center_lon = sum(first_building_lons) / len(first_building_lons)

m = folium.Map(location=[center_lat, center_lon], zoom_start=18, tiles='OpenStreetMap')

# Add building centroids directly to map (no clustering)
for feature in features:
    # Get building info
    building_id = feature["id"]
    confidence = feature["properties"]["confidence"]
    original_count = feature["properties"].get("original_detection_count", 1)
    
    # Get polygon coordinates
    polygon_coords = feature["geometry"]["coordinates"][0]
    
    # Calculate bounding box
    lats = [coord[1] for coord in polygon_coords]
    lons = [coord[0] for coord in polygon_coords]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    
    # Calculate centroid of bounding box
    centroid_lat = (min_lat + max_lat) / 2
    centroid_lon = (min_lon + max_lon) / 2
    
    # Create popup content
    popup_html = f"""
    <b>Building ID:</b> {building_id}<br>
    <b>Confidence:</b> {confidence:.4f}<br>
    <b>Original Detections:</b> {original_count}
    """
    
    # Add centroid marker directly to map
    folium.CircleMarker(
        location=[centroid_lat, centroid_lon],
        radius=3,
        color='red',
        fill=True,
        fill_opacity=0.7,
        popup=folium.Popup(popup_html, max_width=300)
    ).add_to(m)

# Add title with stats
title_html = f'''
<h3 align="center" style="font-size:16px">
    <b>Building Detection Results - Centroids Only</b><br>
    Total Buildings: {total_buildings} | Merging Enabled: {buildings_data["merging_enabled"]}
</h3>
'''
m.get_root().html.add_child(folium.Element(title_html))

# Save the map
m.save('building_validation_map.html')
print(f"Map saved to building_validation_map.html with {total_buildings} building centroids")
print("- Red dots: Building centroids (center points of bounding boxes)")