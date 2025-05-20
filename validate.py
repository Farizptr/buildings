import folium
import json

# Load your buildings
with open('coordinates.json', 'r') as f:
    buildings = json.load(f)

# Create a map centered on the first building
center_lat = buildings[0]['latitude']
center_lon = buildings[0]['longitude']
m = folium.Map(location=[center_lat, center_lon], zoom_start=18, tiles='OpenStreetMap')

# Add building points
for building in buildings:
    folium.CircleMarker(
        location=[building['latitude'], building['longitude']],
        radius=3,
        color='red',
        fill=True,
        fill_opacity=0.7,
        popup=f"ID: {building['building_id']}"
    ).add_to(m)

# Save the map
m.save('building_validation_map.html')
print("Map saved to building_validation_map.html")