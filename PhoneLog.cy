MATCH (n:PhoneLog)
WITH n,
     n.properties AS props, 
     n.geometry AS geom
WITH n, 
     replace(replace(geom, '"', ''), '{', '') AS geomStr,
     replace(replace(props, '"', ''), '{', '') AS propsStr
WITH n,
     split(geomStr, ', ') AS geomList,
     split(propsStr, ', ') AS propsList
RETURN n.user_id AS user_id,
       split(geomList[0], ': ')[1] AS geom_type,
       [toFloat(split(split(geomList[1], ': [')[1], ',')[0]), toFloat(split(split(geomList[1], ': [')[1], ',')[1])] AS coordinates,
       toInteger(split(propsList[0], ': ')[1]) AS speed,
       split(propsList[1], ': ')[1] AS battery_state,
       split(propsList[2], ': [')[1] AS motion,
       split(propsList[3], ': ')[1] AS timestamp,
       toFloat(split(propsList[4], ': ')[1]) AS battery_level,
       toInteger(split(propsList[5], ': ')[1]) AS vertical_accuracy,
       split(propsList[6], ': ')[1] AS pauses,
       toInteger(split(propsList[7], ': ')[1]) AS horizontal_accuracy,
       split(propsList[8], ': ')[1] AS wifi,
       toInteger(split(propsList[9], ': ')[1]) AS deferred,
       toInteger(split(propsList[10], ': ')[1]) AS significant_change,
       toInteger(split(propsList[11], ': ')[1]) AS locations_in_payload,
       split(propsList[12], ': ')[1] AS activity,
       split(propsList[13], ': ')[1] AS device_id,
       toFloat(split(propsList[14], ': ')[1]) AS altitude,
       toInteger(split(propsList[15], ': ')[1]) AS desired_accuracy



// Apoc Version that might work better?
MATCH (n:PhoneLog)
WITH n,
     n.properties AS props, 
     n.geometry AS geom
WITH n, 
     replace(replace(replace(replace(geom, "'", "\""), "True", "true"), "False", "false"), "None", "null") AS geomStr,
     replace(replace(replace(replace(props, "'", "\""), "True", "true"), "False", "false"), "None", "null") AS propsStr
WITH n,
     CASE WHEN geomStr IS NOT NULL THEN apoc.convert.fromJsonMap(geomStr) ELSE null END AS geomMap,
     CASE WHEN propsStr IS NOT NULL THEN apoc.convert.fromJsonMap(propsStr) ELSE null END AS propMap
RETURN n.user_id AS user_id,
       CASE WHEN geomMap IS NOT NULL THEN geomMap.type ELSE null END AS geom_type,
       CASE WHEN geomMap IS NOT NULL THEN geomMap.coordinates ELSE [null, null] END AS coordinates,
       CASE WHEN propMap IS NOT NULL THEN propMap.speed ELSE null END AS speed,
       CASE WHEN propMap IS NOT NULL THEN propMap.battery_state ELSE null END AS battery_state,
       CASE WHEN propMap IS NOT NULL THEN propMap.motion ELSE null END AS motion,
       CASE WHEN propMap IS NOT NULL THEN propMap.timestamp ELSE null END AS timestamp,
       CASE WHEN propMap IS NOT NULL THEN propMap.battery_level ELSE null END AS battery_level,
       CASE WHEN propMap IS NOT NULL THEN propMap.vertical_accuracy ELSE null END AS vertical_accuracy,
       CASE WHEN propMap IS NOT NULL THEN propMap.pauses ELSE null END AS pauses,
       CASE WHEN propMap IS NOT NULL THEN propMap.horizontal_accuracy ELSE null END AS horizontal_accuracy,
       CASE WHEN propMap IS NOT NULL THEN propMap.wifi ELSE null END AS wifi,
       CASE WHEN propMap IS NOT NULL THEN propMap.deferred ELSE null END AS deferred,
       CASE WHEN propMap IS NOT NULL THEN propMap.significant_change ELSE null END AS significant_change,
       CASE WHEN propMap IS NOT NULL THEN propMap.locations_in_payload ELSE null END AS locations_in_payload,
       CASE WHEN propMap IS NOT NULL THEN propMap.activity ELSE null END AS activity,
       CASE WHEN propMap IS NOT NULL THEN propMap.device_id ELSE null END AS device_id,
       CASE WHEN propMap IS NOT NULL THEN propMap.altitude ELSE null END AS altitude,
       CASE WHEN propMap IS NOT NULL THEN propMap.desired_accuracy ELSE null END AS desired_accuracy


// Final Version that works well but requires apoc
// Set the batch size
WITH 100 AS batchSize

// Use apoc.periodic.iterate to process nodes in batches
CALL apoc.periodic.iterate(
  '
  MATCH (n:PhoneLog)
  RETURN n
  ',
  '
  WITH n,
       n.properties AS props, 
       n.geometry AS geom
  WITH n, 
       replace(replace(replace(replace(geom, "\'", "\\\""), "True", "true"), "False", "false"), "None", "null") AS geomStr,
       replace(replace(replace(replace(props, "\'", "\\\""), "True", "true"), "False", "false"), "None", "null") AS propsStr
  WITH n,
       CASE WHEN geomStr IS NOT NULL THEN apoc.convert.fromJsonMap(geomStr) ELSE null END AS geomMap,
       CASE WHEN propsStr IS NOT NULL THEN apoc.convert.fromJsonMap(propsStr) ELSE null END AS propMap,
       CASE WHEN geomStr IS NOT NULL THEN apoc.convert.fromJsonMap(geomStr).coordinates[0] ELSE null END AS longitude,
       CASE WHEN geomStr IS NOT NULL THEN apoc.convert.fromJsonMap(geomStr).coordinates[1] ELSE null END AS latitude
  SET n.geom_type = CASE WHEN geomMap IS NOT NULL THEN geomMap.type ELSE null END,
      n.coordinates = CASE WHEN geomMap IS NOT NULL THEN geomMap.coordinates ELSE [null, null] END,
      n.longitude = toFloat(longitude),
      n.latitude = toFloat(latitude),
      n.speed = CASE WHEN propMap IS NOT NULL THEN propMap.speed ELSE null END,
      n.battery_state = CASE WHEN propMap IS NOT NULL THEN propMap.battery_state ELSE null END,
      n.motion = CASE WHEN propMap IS NOT NULL THEN propMap.motion ELSE null END,
      n.timestamp = CASE WHEN propMap IS NOT NULL THEN propMap.timestamp ELSE null END,
      n.battery_level = CASE WHEN propMap IS NOT NULL THEN propMap.battery_level ELSE null END,
      n.vertical_accuracy = CASE WHEN propMap IS NOT NULL THEN propMap.vertical_accuracy ELSE null END,
      n.pauses = CASE WHEN propMap IS NOT NULL THEN propMap.pauses ELSE null END,
      n.horizontal_accuracy = CASE WHEN propMap IS NOT NULL THEN propMap.horizontal_accuracy ELSE null END,
      n.wifi = CASE WHEN propMap IS NOT NULL THEN propMap.wifi ELSE null END,
      n.deferred = CASE WHEN propMap IS NOT NULL THEN propMap.deferred ELSE null END,
      n.significant_change = CASE WHEN propMap IS NOT NULL THEN propMap.significant_change ELSE null END,
      n.locations_in_payload = CASE WHEN propMap IS NOT NULL THEN propMap.locations_in_payload ELSE null END,
      n.activity = CASE WHEN propMap IS NOT NULL THEN propMap.activity ELSE null END,
      n.device_id = CASE WHEN propMap IS NOT NULL THEN propMap.device_id ELSE null END,
      n.altitude = CASE WHEN propMap IS NOT NULL THEN propMap.altitude ELSE null END,
      n.desired_accuracy = CASE WHEN propMap IS NOT NULL THEN propMap.desired_accuracy ELSE null END
  RETURN count(*)
  ',
  {batchSize: batchSize, parallel: false}
)
YIELD batches, total AS processed, errorMessages
RETURN batches, processed, errorMessages

// Chains together the nodes that have not been included as of yet
MATCH (p:PhoneLog)
WHERE NOT (p)-[:NEXT]->()  // p has no outgoing NEXT
WITH p
ORDER BY p.timestamp ASC
WITH collect(p) AS logs
UNWIND range(0, size(logs) - 2) AS i
WITH logs[i] AS fromLog, logs[i + 1] AS toLog
WHERE NOT ()-[:NEXT]->(toLog)  // toLog has no incoming NEXT
MERGE (fromLog)-[:NEXT]->(toLog)

///////////////////////////////////////////////////////////GPT5
// Constraints
CREATE CONSTRAINT phone_id IF NOT EXISTS FOR (p:PhoneLog) REQUIRE p.elementId IS UNIQUE; // internal safeguard
CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE;
CREATE CONSTRAINT device_id IF NOT EXISTS FOR (d:Device) REQUIRE d.id IS UNIQUE;

// Helpful indexes for joins / sort
CREATE INDEX phone_ts IF NOT EXISTS FOR (p:PhoneLog) ON (p.timestamp);
CREATE INDEX phone_epoch IF NOT EXISTS FOR (p:PhoneLog) ON (p.epoch_millis);
CREATE INDEX phone_user IF NOT EXISTS FOR (p:PhoneLog) ON (p.user_id);
CREATE INDEX phone_device IF NOT EXISTS FOR (p:PhoneLog) ON (p.device_id);
CREATE INDEX phone_loc IF NOT EXISTS FOR (p:PhoneLog) ON (p.loc);  // point index



////////-------------------------------------////////////////
////////-------------------------------------////////////////
////////-------------------------------------////////////////
WITH 500 AS batchSize, 1 AS SCHEMA_VERSION
CALL apoc.periodic.iterate(
  // Only new or outdated records
  '
  MATCH (n:PhoneLog)
  WHERE coalesce(n.schema_version,0) < $SCHEMA_VERSION
  RETURN n
  ',
  '
  // --- 1) Try to coerce python-dict-like strings to JSON strings
  WITH n,
       n.properties AS props,
       n.geometry   AS geom

  WITH n,
       // Convert python repr -> JSON
       // - single quotes -> double quotes
       // - True/False/None -> true/false/null
       replace(replace(replace(replace(geom, "\'", "\\\""), "True", "true"), "False", "false"), "None", "null") AS geomStr,
       replace(replace(replace(replace(props, "\'", "\\\""), "True", "true"), "False", "false"), "None", "null") AS propsStr

  // --- 2) Parse to maps (guarded)
  WITH n,
       CASE WHEN geomStr IS NOT NULL AND geomStr STARTS WITH "{" THEN apoc.convert.fromJsonMap(geomStr) ELSE NULL END AS geomMap,
       CASE WHEN propsStr IS NOT NULL AND propsStr STARTS WITH "{" THEN apoc.convert.fromJsonMap(propsStr) ELSE NULL END AS propMap

  // --- 3) Pull fields with types
  WITH n, geomMap, propMap,
       CASE WHEN geomMap IS NOT NULL THEN geomMap.coordinates ELSE NULL END AS coords

  SET n.geom_type  = CASE WHEN geomMap IS NOT NULL THEN geomMap.type ELSE n.geom_type END,
      n.coordinates = CASE WHEN coords IS NOT NULL THEN coords ELSE n.coordinates END,
      n.longitude   = CASE WHEN coords IS NOT NULL THEN toFloat(coords[0]) ELSE n.longitude END,
      n.latitude    = CASE WHEN coords IS NOT NULL THEN toFloat(coords[1]) ELSE n.latitude END,

      n.speed               = CASE WHEN propMap IS NOT NULL THEN toFloat(propMap.speed)               ELSE n.speed END,
      n.battery_state       = CASE WHEN propMap IS NOT NULL THEN propMap.battery_state               ELSE n.battery_state END,
      n.motion              = CASE WHEN propMap IS NOT NULL THEN propMap.motion                       ELSE n.motion END,
      n.timestamp           = CASE WHEN propMap IS NOT NULL THEN propMap.timestamp                    ELSE n.timestamp END,
      n.battery_level       = CASE WHEN propMap IS NOT NULL THEN toFloat(propMap.battery_level)       ELSE n.battery_level END,
      n.vertical_accuracy   = CASE WHEN propMap IS NOT NULL THEN toFloat(propMap.vertical_accuracy)   ELSE n.vertical_accuracy END,
      n.horizontal_accuracy = CASE WHEN propMap IS NOT NULL THEN toFloat(propMap.horizontal_accuracy) ELSE n.horizontal_accuracy END,
      n.pauses              = CASE WHEN propMap IS NOT NULL THEN toBooleanOrNull(propMap.pauses)      ELSE n.pauses END,
      n.wifi                = CASE WHEN propMap IS NOT NULL THEN propMap.wifi                         ELSE n.wifi END,
      n.deferred            = CASE WHEN propMap IS NOT NULL THEN toBooleanOrNull(propMap.deferred)    ELSE n.deferred END,
      n.significant_change  = CASE WHEN propMap IS NOT NULL THEN toBooleanOrNull(propMap.significant_change) ELSE n.significant_change END,
      n.locations_in_payload= CASE WHEN propMap IS NOT NULL THEN toInteger(propMap.locations_in_payload) ELSE n.locations_in_payload END,
      n.activity            = CASE WHEN propMap IS NOT NULL THEN propMap.activity                     ELSE n.activity END,
      n.device_id           = CASE WHEN propMap IS NOT NULL THEN propMap.device_id                    ELSE n.device_id END,
      n.altitude            = CASE WHEN propMap IS NOT NULL THEN toFloat(propMap.altitude)            ELSE n.altitude END,
      n.desired_accuracy    = CASE WHEN propMap IS NOT NULL THEN toInteger(propMap.desired_accuracy)  ELSE n.desired_accuracy END

  // --- 4) Derived fields: loc (point), typed ts + epoch_millis
  WITH n
  SET n.loc = CASE
                WHEN n.latitude IS NOT NULL AND n.longitude IS NOT NULL
                THEN point({latitude: n.latitude, longitude: n.longitude, crs: "wgs-84"})
                ELSE n.loc
              END
  WITH n, CASE WHEN n.timestamp IS NOT NULL THEN datetime(n.timestamp) END AS ts
  SET n.ts = CASE WHEN ts IS NOT NULL THEN ts ELSE n.ts END,
      n.epoch_millis = CASE WHEN ts IS NOT NULL THEN ts.epochMillis ELSE n.epoch_millis END,
      n.schema_version = $SCHEMA_VERSION,
      n.normalized = true

  // --- 5) Link to User and Device (safe idempotent MERGEs)
  WITH n
  FOREACH (_ IN CASE WHEN n.user_id   IS NOT NULL THEN [1] ELSE [] END |
    MERGE (u:User {id: n.user_id})
    MERGE (n)-[:BY_USER]->(u)
  )
  FOREACH (_ IN CASE WHEN n.device_id IS NOT NULL THEN [1] ELSE [] END |
    MERGE (d:Device {id: n.device_id})
    MERGE (n)-[:FROM_DEVICE]->(d)
  )

  RETURN count(*)
  ',
  {batchSize: batchSize, parallel: false, params: {SCHEMA_VERSION: 1}}
)
YIELD batches, total, errorMessages
RETURN batches, total AS processed, errorMessages;
////////-------------------------------------////////////////
////////-------------------------------------////////////////
////////-------------------------------------////////////////

WITH 1000 AS batchSize
CALL apoc.periodic.iterate(
  '
  MATCH (p:PhoneLog)
  WHERE p.normalized = true AND (p.linked IS NULL OR p.linked = false)
  RETURN p
  ',
  '
  WITH p
  MATCH (prev:PhoneLog {device_id: p.device_id})
  WHERE prev.epoch_millis < p.epoch_millis
    AND NOT (prev)-[:NEXT]->()   // last known node for that device
  WITH p, prev
  ORDER BY prev.epoch_millis DESC
  LIMIT 1
  MERGE (prev)-[:NEXT]->(p)
  SET p.linked = true
  RETURN count(*)
  ',
  {batchSize: batchSize, parallel: false}
)
YIELD batches, total, errorMessages
RETURN batches, total, errorMessages;

////////-------------------------------------////////////////
////////-------------------------------------////////////////
////////-------------------------------------////////////////
Save this to: $NEO4J_HOME/import/phlog_normalize.cypher

2) Schedule it at 3:00 AM daily

    Cron here is seconds minute hour day month day-of-week.
    “0 0 3 * * *” → every day at 03:00:00 (uses Neo4j/JVM server timezone).

// Remove if it already exists
CALL apoc.scheduler.remove('normalizePhoneLogDaily') YIELD name
RETURN name;

// Schedule daily at 3:00 AM
CALL apoc.scheduler.schedule(
  'normalizePhoneLogDaily',
  "CALL apoc.cypher.runFile('phlog_normalize.cypher', {})",
  {cron: '0 0 3 * * *', paused: false}
);

// Inspect
CALL apoc.scheduler.jobs();
////////phlog_normalize.cypher--------------////////////////
// Daily PhoneLog normalization pass (idempotent)
WITH 500 AS batchSize, 1 AS SCHEMA_VERSION
CALL apoc.periodic.iterate(
  '
  MATCH (n:PhoneLog)
  WHERE coalesce(n.schema_version,0) < $SCHEMA_VERSION
  RETURN n
  ',
  '
  WITH n, n.properties AS props, n.geometry AS geom
  WITH n,
       replace(replace(replace(replace(geom, "\'", "\\\""), "True", "true"), "False", "false"), "None", "null") AS geomStr,
       replace(replace(replace(replace(props, "\'", "\\\""), "True", "true"), "False", "false"), "None", "null") AS propsStr
  WITH n,
       CASE WHEN geomStr IS NOT NULL AND geomStr STARTS WITH "{" THEN apoc.convert.fromJsonMap(geomStr) ELSE NULL END AS geomMap,
       CASE WHEN propsStr IS NOT NULL AND propsStr STARTS WITH "{" THEN apoc.convert.fromJsonMap(propsStr) ELSE NULL END AS propMap,
       CASE WHEN geomStr IS NOT NULL AND geomStr STARTS WITH "{" THEN apoc.convert.fromJsonMap(geomStr).coordinates[0] ELSE NULL END AS longitude,
       CASE WHEN geomStr IS NOT NULL AND geomStr STARTS WITH "{" THEN apoc.convert.fromJsonMap(geomStr).coordinates[1] ELSE NULL END AS latitude
  SET n.geom_type  = CASE WHEN geomMap IS NOT NULL THEN geomMap.type ELSE n.geom_type END,
      n.coordinates = CASE WHEN geomMap IS NOT NULL THEN geomMap.coordinates ELSE n.coordinates END,
      n.longitude   = CASE WHEN longitude IS NOT NULL THEN toFloat(longitude) ELSE n.longitude END,
      n.latitude    = CASE WHEN latitude  IS NOT NULL THEN toFloat(latitude)  ELSE n.latitude  END,
      n.speed               = CASE WHEN propMap IS NOT NULL THEN toFloat(propMap.speed)               ELSE n.speed END,
      n.battery_state       = CASE WHEN propMap IS NOT NULL THEN propMap.battery_state               ELSE n.battery_state END,
      n.motion              = CASE WHEN propMap IS NOT NULL THEN propMap.motion                       ELSE n.motion END,
      n.timestamp           = CASE WHEN propMap IS NOT NULL THEN propMap.timestamp                    ELSE n.timestamp END,
      n.battery_level       = CASE WHEN propMap IS NOT NULL THEN toFloat(propMap.battery_level)       ELSE n.battery_level END,
      n.vertical_accuracy   = CASE WHEN propMap IS NOT NULL THEN toFloat(propMap.vertical_accuracy)   ELSE n.vertical_accuracy END,
      n.horizontal_accuracy = CASE WHEN propMap IS NOT NULL THEN toFloat(propMap.horizontal_accuracy) ELSE n.horizontal_accuracy END,
      n.pauses              = CASE WHEN propMap IS NOT NULL THEN toBooleanOrNull(propMap.pauses)      ELSE n.pauses END,
      n.wifi                = CASE WHEN propMap IS NOT NULL THEN propMap.wifi                         ELSE n.wifi END,
      n.deferred            = CASE WHEN propMap IS NOT NULL THEN toBooleanOrNull(propMap.deferred)    ELSE n.deferred END,
      n.significant_change  = CASE WHEN propMap IS NOT NULL THEN toBooleanOrNull(propMap.significant_change) ELSE n.significant_change END,
      n.locations_in_payload= CASE WHEN propMap IS NOT NULL THEN toInteger(propMap.locations_in_payload) ELSE n.locations_in_payload END,
      n.activity            = CASE WHEN propMap IS NOT NULL THEN propMap.activity                     ELSE n.activity END,
      n.device_id           = CASE WHEN propMap IS NOT NULL THEN propMap.device_id                    ELSE n.device_id END,
      n.altitude            = CASE WHEN propMap IS NOT NULL THEN toFloat(propMap.altitude)            ELSE n.altitude END,
      n.desired_accuracy    = CASE WHEN propMap IS NOT NULL THEN toInteger(propMap.desired_accuracy)  ELSE n.desired_accuracy END
  WITH n
  SET n.loc = CASE
                WHEN n.latitude IS NOT NULL AND n.longitude IS NOT NULL
                THEN point({latitude: n.latitude, longitude: n.longitude, crs: "wgs-84"})
                ELSE n.loc
              END
  WITH n, CASE WHEN n.timestamp IS NOT NULL THEN datetime(n.timestamp) END AS ts
  SET n.ts = CASE WHEN ts IS NOT NULL THEN ts ELSE n.ts END,
      n.epoch_millis = CASE WHEN ts IS NOT NULL THEN ts.epochMillis ELSE n.epoch_millis END,
      n.schema_version = $SCHEMA_VERSION,
      n.normalized = true
  RETURN count(*)
  ',
  {batchSize: batchSize, parallel: false, params: {SCHEMA_VERSION: SCHEMA_VERSION}}
)
YIELD batches, total, errorMessages
RETURN batches, total, errorMessages;
