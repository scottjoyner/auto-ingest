from neo4j import GraphDatabase
import json
from datetime import datetime
import pytz

def fetch_points(tx):
    result = tx.run(query)
    for record in result:
        # Parse UTC timestamp and convert to EST
        utc_time = datetime.strptime(record["timestamp"], "%Y-%m-%dT%H:%M:%SZ")
        utc_time = utc_time.replace(tzinfo=pytz.UTC)
        est_time = utc_time.astimezone(pytz.timezone("US/Eastern"))
        est_str = est_time.strftime("%Y-%m-%d %I:%M:%S %p %Z")

        points.append({
            "timestamp_utc": record["timestamp"],
            "timestamp_est": est_str,
            "lat": record["lat"],
            "lon": record["lon"],
            "speed": record.get("speed", 0),
            "altitude": record.get("altitude", 0)
        })
# Neo4j connection settings
NEO4J_URI = "bolt://localhost:7687"  # or your server URI
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "livelongandprosper"

# Connect to Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Define the query
query = """
MATCH (p:PhoneLog)
WHERE p.timestamp >= '2025-04-25T01:00:00Z'
AND p.timestamp <= '2025-04-25T02:50:00Z'
RETURN p.timestamp AS timestamp, p.latitude AS lat, p.longitude AS lon
ORDER BY p.timestamp DESC
"""
query = """
MATCH (p:PhoneLog)
WHERE p.timestamp >= '2025-04-25T01:00:00Z'
  AND p.timestamp <= '2025-04-25T03:30:00Z'
WITH p
ORDER BY p.timestamp DESC
WITH collect(p) AS logs
UNWIND range(0, size(logs) - 1) AS i
WITH logs[i] AS p, i
WHERE i % 5 = 0  // only every 5th node
RETURN 
  p.timestamp AS timestamp,
  p.latitude AS lat,
  p.longitude AS lon,
  p.speed AS speed,
  p.altitude AS altitude
ORDER BY timestamp DESC
"""

points = []

# def fetch_points(tx):
#     result = tx.run(query)
#     for record in result:
#         points.append({
#             "timestamp": record["timestamp"],
#             "lat": record["lat"],
#             "lon": record["lon"]
#         })

# Run the query
with driver.session() as session:
    session.read_transaction(fetch_points)

driver.close()

# Save points to a JSON file (optional)
with open("points.json", "w") as f:
    json.dump(points, f, indent=2)
# Generate the upgraded HTML map
html_header = """
<!DOCTYPE html>
<html>
<head>
    <title>PhoneLog Map (Dots + Lines)</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        #map { height: 100vh; width: 100%; }
    </style>
    <link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css" />
</head>
<body>
    <div id="map"></div>
    <script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
    <script>
        var map = L.map('map').setView([0, 0], 2);  // Center map initially

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            maxZoom: 19,
            attribution: '© OpenStreetMap'
        }).addTo(map);

        var points = 
"""

html_points = json.dumps(points, indent=2)

html_footer = """
        ;

        var latlngs = [];

        for (var i = 0; i < points.length; i++) {
            var point = points[i];
            latlngs.push([point.lat, point.lon]);

            L.circleMarker([point.lat, point.lon], {
                radius: 4,
                color: 'blue',
                fillColor: 'cyan',
                fillOpacity: 0.7
            }).addTo(map).bindPopup(
                "<b>Time (EST):</b> " + point.timestamp_est + "<br>" +
                "<b>Speed:</b> " + point.speed + " mph<br>" +
                "<b>Altitude:</b> " + point.altitude + " m"
            );

        }

        // Draw a line connecting all points
        if (latlngs.length > 1) {
            var polyline = L.polyline(latlngs, {color: 'red'}).addTo(map);
            map.fitBounds(polyline.getBounds());
        } else if (latlngs.length === 1) {
            map.setView(latlngs[0], 12);
        }
    </script>
</body>
</html>
"""

# Write the final HTML
with open("map.html", "w") as f:
    f.write(html_header + html_points + html_footer)

print("✅ Map with dots and connecting lines generated: map.html")
