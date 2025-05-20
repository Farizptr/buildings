import requests
import json
from time import sleep

# Load coordinates
with open('polygon_detection_results/buildings.json') as f:
    coords = json.load(f)

results = []
seen_addresses = set()

for idx, point in enumerate(coords):
    lat = point['latitude']
    lon = point['longitude']
    building_id = point['building_id']

    try:
        res = requests.get(
            "http://localhost:8080/reverse",
            params={"lat": lat, "lon": lon, "format": "json"},
            timeout=5
        )
        data = res.json()
        address = data.get("display_name", "N/A")

        # If address is new, store it
        if address not in seen_addresses:
            seen_addresses.add(address)
            results.append({
                "building_id": building_id,
                "latitude": lat,
                "longitude": lon,
                "address": address,
                "details": data.get("address", {})
            })
            print(f"[{len(results)}] Added unique address: {address}")
        else:
            print(f"[SKIP] Duplicate address for building {building_id}")

    except Exception as e:
        print(f"[ERROR] {lat},{lon}: {e}")

    sleep(0.1)  # Rate limit

# Save filtered result
with open('unique_buildings_by_address.json', 'w') as f:
    json.dump(results, f, indent=2)
