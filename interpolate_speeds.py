#!/usr/bin/env python3
import csv
import re

INPUT_CSV = "rdx_dashcam_metadata_filled_2.csv"
OUTPUT_CSV = "rdx_dashcam_metadata_filled_3.csv"

def is_invalid_coordinate(value):
    """
    Returns True if `value` ends with '.0000' or is '0.0000' (i.e. suspicious).
    Adjust if you only want to treat '0.0000' as invalid, or a subset.
    """
    val = value.strip()
    return val.endswith(".0000")

def is_missing_mph(value):
    """
    Returns True if MPH is '0', '000', or cannot be parsed as a positive int.
    Adjust as needed (e.g., if '00' or '' should also count).
    """
    val = value.strip()
    if not val or val == '0' or val == '000':
        return True
    # If we want to ensure it's a valid integer > 0:
    try:
        ival = int(val)
        if ival == 0:
            return True
        # If you consider, e.g. 999 as suspicious, you could handle that here too.
        return False
    except ValueError:
        return True

def fill_lat_long(rows):
    """
    In-place modification:
      - If Lat/Long is invalid (ends with '.0000'), replace with the last known valid.
    """
    last_valid_lat = None
    last_valid_long = None

    for row in rows:
        lat = row["Lat"].strip()
        lon = row["Long"].strip()

        # Fix lat if invalid
        if is_invalid_coordinate(lat):
            if last_valid_lat is not None:
                row["Lat"] = last_valid_lat
        
        # Fix long if invalid
        if is_invalid_coordinate(lon):
            if last_valid_long is not None:
                row["Long"] = last_valid_long
        
        # Update last valid if what we have now is valid
        if not is_invalid_coordinate(row["Lat"]):
            last_valid_lat = row["Lat"]
        if not is_invalid_coordinate(row["Long"]):
            last_valid_long = row["Long"]

def interpolate_mph(rows):
    """
    - Parse MPH as integers (where possible).
    - Identify runs of missing MPH (is_missing_mph == True).
    - For each consecutive run, if we have known MPH before & after, do a linear interpolation.
      Otherwise, fallback to the nearest known or 0.

    The result is stored back into row["MPH"] as an integer (converted to string).
    """

    # 1. Convert MPH to a list of (is_known, value) for easier processing
    mph_list = []
    for row in rows:
        mph_str = row["MPH"].strip()
        if is_missing_mph(mph_str):
            mph_list.append((False, 0))  # not known, store 0 as placeholder
        else:
            # parse as int
            try:
                mph_val = int(mph_str)
                mph_list.append((True, mph_val))
            except ValueError:
                # If parse fails, treat as missing
                mph_list.append((False, 0))

    # 2. Interpolate over consecutive runs of missing data
    #    We'll do it in one pass. For each run [start..end], if there's known mph
    #    at start-1 and end+1, do linear interpolation. Otherwise fallback to a simpler guess.

    n = len(mph_list)
    i = 0
    while i < n:
        if mph_list[i][0]:
            # This index has a known mph, skip
            i += 1
            continue

        # We found a missing run starting at i
        start = i
        # Move `j` until we hit the end or a known mph
        j = i
        while j < n and mph_list[j][0] == False:
            j += 1
        # Now j is the first known after the missing run, or j == n

        run_length = j - start  # number of missing points

        # The known mph before this run (if any)
        prev_idx = start - 1
        prev_mph = mph_list[prev_idx][1] if prev_idx >= 0 and mph_list[prev_idx][0] else None

        # The known mph after this run (if any)
        next_idx = j
        next_mph = mph_list[next_idx][1] if next_idx < n and mph_list[next_idx][0] else None

        if prev_mph is not None and next_mph is not None:
            # We can do a linear interpolation from prev_mph -> next_mph across run_length+1 intervals
            step = (next_mph - prev_mph) / float(run_length + 1)
            for offset in range(run_length):
                # mph at index (start + offset) is prev_mph + step*(offset+1)
                fill_val = prev_mph + step * (offset + 1)
                mph_list[start + offset] = (True, int(round(fill_val)))
        elif prev_mph is not None and next_mph is None:
            # We only know the mph before the run. Fill with that mph or a simple downward guess
            # For simplicity, let's just copy `prev_mph`.
            for offset in range(run_length):
                mph_list[start + offset] = (True, prev_mph)
        elif prev_mph is None and next_mph is not None:
            # We only know the mph after the run. Fill with that mph
            for offset in range(run_length):
                mph_list[start + offset] = (True, next_mph)
        else:
            # We have no known mph before or after => set them all to 0
            for offset in range(run_length):
                mph_list[start + offset] = (True, 0)

        # Move i to j
        i = j

    # 3. Store back into the rows
    for idx, row in enumerate(rows):
        # mph_list[idx] is now guaranteed (True, some_value)
        final_mph = mph_list[idx][1]
        row["MPH"] = str(final_mph)

def main():
    # Read rows
    with open(INPUT_CSV, newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)

    if not rows:
        print("No data found in input CSV.")
        return

    # 1) Fill lat/long if ends with .0000
    fill_lat_long(rows)

    # 2) Interpolate MPH if 0 or '000'
    interpolate_mph(rows)

    # Write out
    fieldnames = rows[0].keys()
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done! Created '{OUTPUT_CSV}' with lat/long fixed and MPH interpolated.")

if __name__ == "__main__":
    main()
