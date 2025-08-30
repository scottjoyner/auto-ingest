// 1) Ensure APOC is installed and enabled in neo4j.conf if needed.
// 2) Place trips_2024.csv in the Neo4j import folder.
////////////////////////////////////
////////////////////////////////////
////////////////////////////////////
LOAD CSV WITH HEADERS
FROM 'file:///trips_2024.csv' AS row

MERGE (t:Trip {
  // Combine 'Tracker Name' + 'Start Time' as a unique key. 
  // Adjust if your CSV already has a truly unique field (e.g. "Trip ID").
  uniqueKey: row["Tracker Name"] + "_" + row["Start Time"]
})
ON CREATE SET
    t.tripId             = randomUUID(),
    t.authorized         = row["Authorized"],
    t.endLocation        = row["End Location"],
    t.startLocation      = row["Start Location"],
    t.trackerName        = row["Tracker Name"],
    t.tripMiles          = toInteger(row["Trip Miles"]),
    t.timeAtLocationMins = toInteger(row["Time at Location (Minutes)"]),
    
    // Store as string (optional)
    t.startTimeString    = row["Start Time"],
    t.endTimeString      = row["End Time"],
    
    // Parse into a proper datetime (requires APOC)
    t.startTime = datetime({
      epochMillis: apoc.date.parse(row["Start Time"], "ms", "MM/dd/yyyy hh:mm a")
    }),
    t.endTime = datetime({
      epochMillis: apoc.date.parse(row["End Time"], "ms", "MM/dd/yyyy hh:mm a")
    }),
    
    t.driverName         = row["Driver Name"],
    t.tripTimeMins       = toInteger(row["Trip Time (Minutes)"]),
    t.group              = row["Group"];

////////////////////////////////////
////////////////////////////////////
////////////////////////////////////
LOAD CSV WITH HEADERS
FROM 'file:///rdx_stops_2024.csv' AS row

MERGE (s:Stop {
  // Combine Tracker Name + Begin Date as a unique identifier
  // (Adjust if you have a better unique field)
  uniqueKey: row["Tracker Name"] + "_" + row["Begin Date"]
})
ON CREATE SET
    s.stopId      = randomUUID(),
    s.trackerName = row["Tracker Name"],
    s.device      = row["Device"],
    s.duration    = row["Duration"],
    s.location    = row["Location"],
    s.geofence    = row["Geofence"],
    s.googleMaps  = row["Google Maps"],

    // Optionally store the original strings
    s.beginDateString = row["Begin Date"],
    s.endDateString   = row["End Date"],

    // Convert to Neo4j datetime (APOC)
    s.startTime  = datetime({
      epochMillis: apoc.date.parse(row["Begin Date"], "ms", "MM/dd/yyyy hh:mm a")
    }),
    s.endTime    = datetime({
      epochMillis: apoc.date.parse(row["End Date"], "ms", "MM/dd/yyyy hh:mm a")
    });

////////////////////////////////////
////////////////////////////////////
////////////////////////////////////
MATCH (e) 
WHERE e:Trip OR e:Stop
WITH e
ORDER BY e.trackerName, e.startTime
WITH collect(e) AS allEvents
UNWIND range(0, size(allEvents) - 2) AS i
WITH allEvents[i] AS currentEvent, allEvents[i+1] AS nextEvent
MERGE (currentEvent)-[:NEXT_EVENT]->(nextEvent);

////////////////////////////////////
////////////////////////////////////
////////////////////////////////////

LOAD CSV WITH HEADERS
FROM 'file:///rdx_locations_2024.csv' AS row

// 1) Convert the Date string to a proper datetime using APOC
WITH row,
datetime({
  epochMillis: apoc.date.parse(row["Date"], 'ms', 'MM/dd/yyyy hh:mm a')
}) AS locationTime

// 2) Find a Trip or Stop node with the same Tracker Name and time window
MATCH (e)
WHERE (e:Trip OR e:Stop)
  AND e.trackerName = row["Tracker Name"]
  AND e.startTime <= locationTime <= e.endTime

// 3) Create or find the LocationEvent node for this IMEI + date combination
MERGE (loc:LocationEvent { uniqueKey: row["Device IMEI"] + "_" + row["Date"] })
ON CREATE SET
  loc.model           = row["Model"],
  loc.vin             = row["VIN"],
  loc.geofence        = row["Geofence"],
  loc.mapLink         = row["Location"],
  loc.direction       = row["Direction"],
  loc.trackerName     = row["Tracker Name"],
  loc.speed           = row["Speed"],
  loc.deviceImei      = row["Device IMEI"],
  loc.internalBattery = row["Internal Battery"],
  loc.dateString      = row["Date"], 
  loc.eventTime       = locationTime,            // Proper datetime
  loc.battery         = row["Battery"],
  loc.latitude        = toFloat(row["Latitude"]),
  loc.longitude       = toFloat(row["Longitude"]),
  loc.fuelLevel       = row["Fuel Level"]        // Use toFloat(...) if you need numeric

// 4) Create the relationship from the LocationEvent to the Trip or Stop
MERGE (loc)-[:BELONGS_TO]->(e)

// Return a simple summary
RETURN count(*) AS totalLinked;


////////////////////////////////////
////////////////////////////////////
////////////////////////////////////

MATCH (e) 
WHERE e:Trip OR e:Stop
MATCH (e)<-[:BELONGS_TO]-(loc:LocationEvent)
WITH e, loc
ORDER BY loc.eventTime
WITH e, collect(loc) AS locEvents
UNWIND range(0, size(locEvents)-2) AS i
WITH e, locEvents[i] AS currentLoc, locEvents[i+1] AS nextLoc
MERGE (currentLoc)-[:NEXT]->(nextLoc)
RETURN count(*) AS totalChained;



MATCH (p:PhoneLog)
WITH p,
  // Convert single quotes to double quotes in 'properties' 
  apoc.text.replace(p.properties, "'", "\"") AS jsonStr,
  // Convert single quotes to double quotes in 'geometry'
  apoc.text.replace(p.geometry, "'", "\"") AS geoStr,
  // Convert phone log timestamp to a proper datetime
  datetime(p.timestamp) AS phoneLogTime

// Now parse both JSON strings
WITH p, phoneLogTime,
  apoc.convert.fromJsonMap(jsonStr) AS data,
  apoc.convert.fromJsonMap(geoStr)  AS geo

// 1) Set node-level properties from the parsed JSON
SET p.speed               = data.speed,
    p.battery_state       = data.battery_state,
    p.motion              = data.motion, 
    p.timestampNested     = data.timestamp,    // or rename if you like
    p.horizontal_accuracy = data.horizontal_accuracy,
    p.speed_accuracy      = data.speed_accuracy,
    p.vertical_accuracy   = data.vertical_accuracy,
    p.battery_level       = data.battery_level,
    p.wifi                = data.wifi,
    p.pauses              = data.pauses,
    p.locations_in_payload= data.locations_in_payload,
    p.course              = data.course,
    p.activity            = data.activity,
    p.device_id           = data.device_id,
    p.altitude            = data.altitude,
    p.course_accuracy     = data.course_accuracy,
    p.desired_accuracy    = data.desired_accuracy,
    p.tracking_mode       = data.tracking_mode,

    // 2) Also parse geometry
    p.geoType             = geo.type,
    p.geoCoordinates      = geo.coordinates

// 3) Align to matching Trip or Stop based on timestamp range (and tracker if needed)
WITH p, phoneLogTime
MATCH (e)
WHERE (e:Trip OR e:Stop)
  AND e.startTime <= phoneLogTime <= e.endTime
  // If your phoneLog's device_id should match Trip/Stop trackerName, uncomment:
  // AND e.trackerName = p.device_id

MERGE (p)-[:BELONGS_TO]->(e)
RETURN count(*) AS totalAligned;
