#!/usr/bin/env python3
import csv
import math

INPUT_CSV = "rdx_dashcam_metadata_filled_5.csv"
OUTPUT_CSV = "rdx_dashcam_metadata_filled_6.csv"

# Threshold for a "significant jump" in lat/long. Adjust to your data scale.
LAT_LON_DIFF_THRESHOLD = 0.01

def is_single_outlier(prev_val, current_val, next_val, threshold):
    """
    Check if 'current_val' is a single-row outlier compared to neighbors:
      1) |current - prev| > threshold
      2) |current - next| > threshold
      3) |prev - next| <= threshold  (neighbors are close to each other)
    """
    if None in (prev_val, current_val, next_val):
        return False

    diff_pc = abs(current_val - prev_val)
    diff_nc = abs(current_val - next_val)
    diff_pn = abs(prev_val - next_val)

    return (diff_pc > threshold) and (diff_nc > threshold) and (diff_pn <= threshold)

def is_double_outlier(prev_val, val_i, val_i1, next_val, threshold):
    """
    Check if 'val_i' and 'val_i1' are two consecutive outliers between
    'prev_val' and 'next_val':
      1) Both val_i and val_i1 differ from prev_val by > threshold
      2) Both val_i and val_i1 differ from next_val by > threshold
      3) prev_val and next_val are fairly close (<= threshold)
    In other words, we have a 2-row "spike" or "dip" between stable neighbors.
    """
    if None in (prev_val, val_i, val_i1, next_val):
        return False

    diff_i_prev = abs(val_i - prev_val)
    diff_i_next = abs(val_i - next_val)
    diff_i1_prev = abs(val_i1 - prev_val)
    diff_i1_next = abs(val_i1 - next_val)
    diff_pn = abs(prev_val - next_val)

    # Both i, i+1 are far from prev and next
    if (diff_i_prev > threshold and diff_i_next > threshold and
        diff_i1_prev > threshold and diff_i1_next > threshold and
        diff_pn <= threshold):
        return True
    return False

def midpoint(a, b):
    """Utility to return midpoint of a and b."""
    return (a + b) / 2.0

def main():
    # 1) Read CSV
    with open(INPUT_CSV, newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)

    if not rows:
        print("No data rows found.")
        return

    # Convert lat/long to floats
    for r in rows:
        try:
            r["Lat"] = float(r["Lat"])
        except ValueError:
            r["Lat"] = None
        try:
            r["Long"] = float(r["Long"])
        except ValueError:
            r["Long"] = None

    n = len(rows)
    i = 1  # We'll iterate from 1 up to n-2 so we can reference neighbors

    while i < n - 1:
        # We'll check double-row outliers first, so we also need to ensure i+2 < n
        # for referencing row[i+2] when we do a double check
        if i < n - 2:
            # We have enough rows to test a 2-row outlier: row[i], row[i+1]
            lat_prev = rows[i - 1]["Lat"]
            lat_i    = rows[i]["Lat"]
            lat_i1   = rows[i + 1]["Lat"]
            lat_next = rows[i + 2]["Lat"]

            lon_prev = rows[i - 1]["Long"]
            lon_i    = rows[i]["Long"]
            lon_i1   = rows[i + 1]["Long"]
            lon_next = rows[i + 2]["Long"]

            # Check if lat is a double outlier for row i, i+1
            lat_double = is_double_outlier(lat_prev, lat_i, lat_i1, lat_next, LAT_LON_DIFF_THRESHOLD)
            # Check if long is a double outlier
            lon_double = is_double_outlier(lon_prev, lon_i, lon_i1, lon_next, LAT_LON_DIFF_THRESHOLD)

            if lat_double:
                # Fix lat for row i, i+1
                corrected_lat = midpoint(lat_prev, lat_next)
                print(f"Double outlier (lat) at rows {i}, {i+1}: {lat_i}, {lat_i1} => {corrected_lat}")
                rows[i]["Lat"] = corrected_lat
                rows[i + 1]["Lat"] = corrected_lat

            if lon_double:
                # Fix long for row i, i+1
                corrected_long = midpoint(lon_prev, lon_next)
                print(f"Double outlier (long) at rows {i}, {i+1}: {lon_i}, {lon_i1} => {corrected_long}")
                rows[i]["Long"] = corrected_long
                rows[i + 1]["Long"] = corrected_long

            # If we corrected for double outlier, skip i+1 since it’s already handled
            if lat_double or lon_double:
                i += 2
                continue

        # If not double outlier, check single outlier for row i
        lat_prev = rows[i - 1]["Lat"]
        lat_curr = rows[i]["Lat"]
        lat_next = rows[i + 1]["Lat"]

        lon_prev = rows[i - 1]["Long"]
        lon_curr = rows[i]["Long"]
        lon_next = rows[i + 1]["Long"]

        # Single outlier check for lat
        if is_single_outlier(lat_prev, lat_curr, lat_next, LAT_LON_DIFF_THRESHOLD):
            new_lat = midpoint(lat_prev, lat_next)
            print(f"Single outlier (lat) at row {i}: {lat_curr} => {new_lat}")
            rows[i]["Lat"] = new_lat

        # Single outlier check for long
        if is_single_outlier(lon_prev, lon_curr, lon_next, LAT_LON_DIFF_THRESHOLD):
            new_long = midpoint(lon_prev, lon_next)
            print(f"Single outlier (long) at row {i}: {lon_curr} => {new_long}")
            rows[i]["Long"] = new_long

        # Move to next row
        i += 1

    # 3) Convert lat/long back to strings
    for r in rows:
        if r["Lat"] is None:
            r["Lat"] = ""
        else:
            r["Lat"] = f"{r['Lat']:.4f}"

        if r["Long"] is None:
            r["Long"] = ""
        else:
            r["Long"] = f"{r['Long']:.4f}"

    # 4) Write corrected CSV
    fieldnames = rows[0].keys()
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone! Outliers replaced where found. Output: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
