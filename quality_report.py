#!/usr/bin/env python3

import csv
import math

INPUT_FILE = "rdx_dashcam_metadata_filled_6.csv"          # Change to your input CSV file
OUTPUT_FILE = "validation_report.txt"  # Where we log validation issues

# Tolerance / thresholds (customize as needed)
MAX_MPH_CHANGE = 20        # e.g. no more than 10 mph difference between lines
MAX_LAT_CHANGE = 1
MAX_LONG_CHANGE = 1
EXPECT_FRAME_INCREMENT = 60  # expected difference in frame from row to row

def is_float(value):
    """Check if the value can be converted to float."""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        # csv.DictReader yields None for a missing trailing cell.
        return False

def validate_rows(rows):
    """
    Compare each row to the previous row based on certain consistency rules.
    Return a list of validation messages for any rows that violate the rules.
    """
    issues = []
    prev_row = None

    for i, row in enumerate(rows):
        current_key = row["Key"]
        current_lat = row["Lat"]
        current_long = row["Long"]
        current_frame = row["Frame"]

        # We'll collect any issues in this row here. DictReader has already
        # consumed the header, so i == 0 is the first *data* row: skipping it
        # used to omit the first record entirely (bad values on a one-row CSV
        # produced "No issues found") and misattributed its bad values to row 2.
        row_issues = []
        if not is_float(current_lat):
            row_issues.append(f"Lat is not numeric (value={current_lat})")
        if not is_float(current_long):
            row_issues.append(f"Long is not numeric (value={current_long})")

        # 1. Check Frame increment
        # if is_float(current_frame) and is_float(prev_frame):
        #     frame_diff = float(current_frame) - float(prev_frame)
        #     if frame_diff != EXPECT_FRAME_INCREMENT:
        #         row_issues.append(f"Frame should be +{EXPECT_FRAME_INCREMENT}, got {frame_diff}")
        # else:
        #     row_issues.append(f"Frame is not numeric (current={current_frame}, prev={prev_frame})")

        # 2. Check MPH difference
        # (If one isn't numeric, that's an issue in itself)
        # if is_float(current_mph) and is_float(prev_mph):
        #     mph_diff = abs(float(current_mph) - float(prev_mph))
        #     if mph_diff > MAX_MPH_CHANGE:
        #         row_issues.append(f"MPH changed by {mph_diff}, exceeds threshold of {MAX_MPH_CHANGE}")
        # else:
        #     row_issues.append(f"MPH is not numeric (current={current_mph}, prev={prev_mph})")

        if prev_row is not None:
            prev_lat = prev_row["Lat"]
            prev_long = prev_row["Long"]
            # 3. Check Lat/Long difference against the preceding row. Numeric
            # validity is reported above for this row itself, so a bad value
            # is no longer silently charged to the following row.
            if is_float(current_lat) and is_float(prev_lat):
                lat_diff = abs(float(current_lat) - float(prev_lat))
                if lat_diff > MAX_LAT_CHANGE:
                    row_issues.append(f"Lat changed by {lat_diff}, exceeds threshold of {MAX_LAT_CHANGE}")

            if is_float(current_long) and is_float(prev_long):
                long_diff = abs(float(current_long) - float(prev_long))
                if long_diff > MAX_LONG_CHANGE:
                    row_issues.append(f"Long changed by {long_diff}, exceeds threshold of {MAX_LONG_CHANGE}")

        # 4. Check Key changes (optional / example)
        # If you expect the same Key for a sequence of frames, you might flag if it changes.
        # Conversely, if you expect the Key to change per row, you'd do the opposite check.
        # if current_key != prev_key:
        #     # For example: if we normally expect the Key to remain the same from one row to the next,
        #     # you can log an issue if it changes.
        #     row_issues.append(f"Key changed from {prev_key} to {current_key} unexpectedly.")

        if row_issues:
            # Add a summary of the row number (i+1 or so) and the raw data
            # i is 0-based; if you want lines 1-based, use i+1
            issue_msg = (
                f"Row {i+1} (Key={current_key}, Frame={current_frame}):\n  "
                + "\n  ".join(row_issues)
            )
            issues.append(issue_msg)

        # Update prev_row for the next iteration
        prev_row = row

    return issues


def main():
    with open(INPUT_FILE, newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)  # load all rows

    issues = validate_rows(rows)

    # Write issues to a report file (or print them)
    if issues:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for issue in issues:
                f.write(issue + "\n\n")
        print(f"Validation complete. Found {len(issues)} issues. See '{OUTPUT_FILE}' for details.")
    else:
        print("No issues found. Data appears consistent with the defined rules.")


if __name__ == "__main__":
    main()
